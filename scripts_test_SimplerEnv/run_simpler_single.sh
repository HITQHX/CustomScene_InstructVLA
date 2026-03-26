#!/bin/bash
# 【纯净版】单任务 + 前台运行 + 终端实时输出 + Ctrl+C一键停止
# 无后台 | 无日志文件 | 无隐藏进程 | 只跑一个任务
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 阻止 TensorFlow 抢占显存
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

# 如果底层的 cuDNN 报错，强行指定后端
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# 仿真器无头渲染必备
export MANISKILL2_HEADLESS=1
export SIMPLER_ENV_HEADLESS=1
export DISPLAY=""
export SAPIEN_DISABLE_X11=1
export SAPIEN_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json


# ===================== 【仅改这里】你的模型路径 =====================
ckpt_path="/root/autodl-tmp/InstructVLA/model/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

# ===================== 安全环境变量（无坑）=====================
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 关闭FlashAttention + 无头渲染（服务器必开）
export TORCH_FLASH_ATTN=0
export DISABLE_FLASH_ATTN=1
export SVULKAN2_HEADLESS=1
export MANISKILL2_HEADLESS=1
export SIMPLER_ENV_HEADLESS=1
export DISPLAY=:1

# ===================== 固定端口（不用算，直接用）=====================
PORT_START=50000
PORT_END=51999

# 进入任务目录
cd ./SimplerEnv

# ===================== 【核心】只跑1个任务！前台运行！实时打印！=====================
# 选最简单的任务：pick_coke_can 捡可乐瓶
echo "============================================="
echo "✅ 开始运行：单任务 - pick_coke_can"
echo "✅ 前台运行 | 实时输出 | Ctrl+C 立即停止"
echo "============================================="

# 关键：无 & 无重定向 → 所有日志直接打印在终端
bash ./scripts_self/server_speed_scripts/1_cogact_pick_coke_can_variant_agg-12.sh $ckpt_path 0 $PORT_START $PORT_END

echo "============================================="
echo "🏁 任务运行完成！"
echo "============================================="