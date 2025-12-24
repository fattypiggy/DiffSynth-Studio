#!/bin/bash
# 示例推理脚本

# ============ 配置 ============
CHECKPOINT="./output/delit_exp1/checkpoint_best.pth"  # 训练好的模型
INPUT="/path/to/VFHQ_videos"  # 输入视频/图像/目录
OUTPUT_DIR="./inference_output"

# 模型参数（必须与训练时一致）
BASE_DIM=64
ENV_HEIGHT=128
ENV_WIDTH=256
LOG_SCALE=10.0
IMAGE_SIZE=512

# ============ 推理 ============

python inference_delit.py \
    --checkpoint $CHECKPOINT \
    --input $INPUT \
    --output_dir $OUTPUT_DIR \
    --base_dim $BASE_DIM \
    --env_height $ENV_HEIGHT \
    --env_width $ENV_WIDTH \
    --log_scale $LOG_SCALE \
    --image_size $IMAGE_SIZE

echo "Inference completed! Results saved to $OUTPUT_DIR"
