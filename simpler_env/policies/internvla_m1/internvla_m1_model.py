import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle

from simpler_env.policies.internvla_m1.adaptive_ensemble import AdaptiveEnsembler


def _resolve_internvla_repo_root(explicit_repo_path: Optional[str] = None) -> Path:
    candidates = []
    if explicit_repo_path:
        candidates.append(Path(explicit_repo_path).expanduser())

    env_repo_path = os.environ.get("INTERNVLA_REPO_PATH")
    if env_repo_path:
        candidates.append(Path(env_repo_path).expanduser())

    current_file = Path(__file__).resolve()
    simpler_env_root = current_file.parents[3]
    candidates.extend(
        [
            simpler_env_root / "InternVLA-M1",
            simpler_env_root.parent / "InternVLA-M1",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and (candidate / "InternVLA").exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the InternVLA-M1 repository. "
        "Set `INTERNVLA_REPO_PATH` or pass `internvla_repo_path`. "
        f"Searched: {searched}"
    )

class InternVLAM1Inference:
    def __init__(
        self,
        saved_model_path: str,
        policy_setup: str = "widowx_bridge",
        unnorm_key: Optional[str] = None,
        image_size: tuple[int, int] = (224, 224),
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
        use_bf16: bool = True,
        internvla_repo_path: Optional[str] = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_dataset" if unnorm_key is None else unnorm_key
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for InternVLA-M1.")

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.action_scale = action_scale
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.task_description = None

        repo_root = _resolve_internvla_repo_root(internvla_repo_path)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from InternVLA.model.framework.M1 import InternVLA_M1
        from InternVLA.model.framework.share_tools import read_mode_config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InternVLA_M1.from_pretrained(saved_model_path)
        if use_bf16 and self.device == "cuda":
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()
        _, norm_stats = read_mode_config(saved_model_path)
        resolved_unnorm_key = InternVLA_M1._check_unnorm_key(norm_stats, self.unnorm_key)
        self.action_norm_stats = norm_stats[resolved_unnorm_key]["action"]
        self.unnorm_key = resolved_unnorm_key

        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        del args, kwargs
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        image = Image.fromarray(image)
        prediction = self.model.predict_action(
            batch_images=[[image]],
            instructions=[self.task_description],
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )

        normalized_actions = prediction["normalized_actions"][0]
        raw_actions = self.model.unnormalize_actions(normalized_actions, self.action_norm_stats)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        else:
            raise NotImplementedError()

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        action_dim_labels = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(action_dim_labels), action_dim_labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(action_dim_labels):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
