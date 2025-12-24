#!/bin/bash

# Activate Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate relit

# Enable OpenEXR
export OPENCV_IO_ENABLE_OPENEXR=1

# Configuration
DATA_ROOT="./testdata"
OUTPUT_DIR="output/delit_run_001"
MODEL_ROOT="/home/wei/GitHub/Wan2.2-I2V-A14B"

# Placeholders - PLEASE UPDATE THESE PATHS
WAN_DIT_PATH="${MODEL_ROOT}/high_noise_model"
WAN_VAE_PATH="${MODEL_ROOT}/Wan2.1_VAE.pth"

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root $DATA_ROOT does not exist."
    exit 1
fi

echo "Starting Delit Training..."
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"

# Run Training
# Use python from current environment (relit)
python diffsynth/scripts/train_delit_lora.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --wan_dit_path "$WAN_DIT_PATH" \
    --wan_vae_path "$WAN_VAE_PATH" \
    --batch_size 1 \
    --epochs 100 \
    --save_interval 500 \
    --mixed_precision bf16 \
    --lora_rank 16

