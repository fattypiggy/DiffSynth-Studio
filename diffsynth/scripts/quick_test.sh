#!/bin/bash
# 快速测试脚本 - 完整验证整个流程

echo "=========================================="
echo "Quick Test - FaceOLAT Delit System"
echo "=========================================="
echo ""

# 设置错误时退出
set -e

# ===== Step 1: 验证模块 =====
echo "Step 1: Verifying modules..."
echo "----------------------------"
python verify_modules.py

echo ""
echo "✓ Module verification completed"
echo ""

# ===== Step 2: 生成测试数据集 =====
echo "Step 2: Generating toy dataset..."
echo "----------------------------"
python generate_toy_dataset.py \
    --output_dir ./toy_dataset \
    --num_subjects 2 \
    --num_envs 5 \
    --image_size 512

echo ""
echo "✓ Toy dataset generated"
echo ""

# ===== Step 3: 快速训练测试 =====
echo "Step 3: Quick training test (10 epochs)..."
echo "----------------------------"
python train_delit.py \
    --data_root ./toy_dataset \
    --output_dir ./output/quick_test \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --base_dim 32 \
    --num_workers 2 \
    --use_mask

echo ""
echo "✓ Training test completed"
echo ""

# ===== Step 4: 推理测试 =====
echo "Step 4: Inference test..."
echo "----------------------------"

# 使用数据集中的一张图像进行测试
TEST_IMAGE=$(find ./toy_dataset/subjects/*/relit/*.exr 2>/dev/null | head -1)

if [ -z "$TEST_IMAGE" ]; then
    # Fallback: 创建一个测试图像
    python -c "
import numpy as np
import cv2
img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
cv2.imwrite('./test_image.png', img)
"
    TEST_IMAGE="./test_image.png"
fi

python inference_delit.py \
    --checkpoint ./output/quick_test/checkpoint_latest.pth \
    --input "$TEST_IMAGE" \
    --output_dir ./inference_output \
    --base_dim 32

echo ""
echo "✓ Inference test completed"
echo ""

# ===== 总结 =====
echo "=========================================="
echo "Quick Test Summary"
echo "=========================================="
echo ""
echo "✓ All tests passed successfully!"
echo ""
echo "Generated files:"
echo "  - Toy dataset: ./toy_dataset/"
echo "  - Training output: ./output/quick_test/"
echo "  - Inference output: ./inference_output/"
echo "  - Verification output: ./verify_output/"
echo ""
echo "Next steps:"
echo "  1. Check TensorBoard: tensorboard --logdir ./output/quick_test/logs"
echo "  2. View inference results in ./inference_output/"
echo "  3. Prepare your real FaceOLAT data"
echo "  4. Run full training with real data"
echo ""
echo "=========================================="
