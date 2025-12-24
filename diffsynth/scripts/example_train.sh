#!/bin/bash
# 示例训练脚本
# 根据你的数据路径和硬件配置修改参数

# ============ 配置 ============
DATA_ROOT="/path/to/your/FaceOLAT_data"  # 修改为你的数据路径
OUTPUT_DIR="./output/delit_exp1"
NUM_GPUS=4  # 可用的 GPU 数量

# 数据参数
IMAGE_SIZE=512
ENV_HEIGHT=128
ENV_WIDTH=256

# 模型参数
BASE_DIM=64  # 可选: 32, 64, 128

# 训练参数
BATCH_SIZE=4  # 每个 GPU 的 batch size
NUM_EPOCHS=100
LEARNING_RATE=1e-4

# ============ 训练 ============

if [ $NUM_GPUS -gt 1 ]; then
    echo "Multi-GPU training with $NUM_GPUS GPUs"
    torchrun --nproc_per_node=$NUM_GPUS train_delit.py \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --image_size $IMAGE_SIZE \
        --env_height $ENV_HEIGHT \
        --env_width $ENV_WIDTH \
        --base_dim $BASE_DIM \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --use_mask \
        --use_lpips \
        --use_augmentation \
        --num_workers 4
else
    echo "Single-GPU training"
    python train_delit.py \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --image_size $IMAGE_SIZE \
        --env_height $ENV_HEIGHT \
        --env_width $ENV_WIDTH \
        --base_dim $BASE_DIM \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --use_mask \
        --use_lpips \
        --use_augmentation \
        --num_workers 4
fi

echo "Training completed! Results saved to $OUTPUT_DIR"
echo "View training logs with: tensorboard --logdir $OUTPUT_DIR/logs"
