import os

import numpy as np
# import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
# from simpler_env.policies.octo.octo_server_model import OctoServerInference
# from simpler_env.policies.rt1.rt1_model import RT1Inference

# try:
#     from simpler_env.policies.octo.octo_model import OctoInference
# except ImportError as e:
#     print("Octo is not correctly imported.")
#     print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # gpus = tf.config.list_physical_devices("GPU")
    # if len(gpus) > 0:
    #     # prevent a single tf process from taking up all the GPU memory
    #     tf.config.set_logical_device_configuration(
    #         gpus[0],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
    #     )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model in {"internvla-m1", "internvla_m1"}:
        assert args.ckpt_path not in {None, "None"}
        if args.internvla_repo_path is not None:
            os.environ["INTERNVLA_REPO_PATH"] = args.internvla_repo_path
        from simpler_env.policies.internvla_m1.internvla_m1_model import InternVLAM1Inference

        model = InternVLAM1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            unnorm_key=args.internvla_unnorm_key,
            image_size=tuple(args.internvla_image_size),
            action_scale=args.action_scale,
            cfg_scale=args.internvla_cfg_scale,
            use_ddim=not args.internvla_disable_ddim,
            num_ddim_steps=args.internvla_num_ddim_steps,
            action_ensemble_horizon=args.internvla_action_ensemble_horizon,
            adaptive_ensemble_alpha=args.internvla_adaptive_ensemble_alpha,
            use_bf16=not args.internvla_no_bf16,
            internvla_repo_path=args.internvla_repo_path,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
