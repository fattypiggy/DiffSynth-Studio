"""
测试所有组件是否正常工作
运行此脚本来验证安装和代码正确性
"""

import sys
from pathlib import Path

def test_imports():
    """测试所有模块是否能正常导入"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from hdr_codec import HDRCodec, HDRCodecTorch
        print("✓ HDR Codec imported successfully")
    except Exception as e:
        print(f"✗ Failed to import HDR Codec: {e}")
        return False

    try:
        from delit_model import SimplifiedDelitModel
        print("✓ Delit Model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Delit Model: {e}")
        return False

    try:
        from delit_loss import SimplifiedDelitLoss
        print("✓ Delit Loss imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Delit Loss: {e}")
        return False

    try:
        from delit_dataset import FaceOLATDelitDataset
        print("✓ Delit Dataset imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Delit Dataset: {e}")
        return False

    print()
    return True


def test_hdr_codec():
    """测试 HDR 编解码器"""
    print("=" * 60)
    print("Testing HDR Codec...")
    print("=" * 60)

    try:
        from hdr_codec import HDRCodec, HDRCodecTorch
        import numpy as np
        import torch

        # NumPy 版本测试
        hdr = np.random.rand(128, 256, 3).astype(np.float32) * 100.0
        encoded = HDRCodec.encode_to_4channel(hdr)
        recovered = HDRCodec.decode_from_4channel(encoded)
        error = np.abs(hdr - recovered).max()

        if error < 1e-4:
            print(f"✓ NumPy HDR Codec test passed (error: {error:.6f})")
        else:
            print(f"✗ NumPy HDR Codec test failed (error: {error:.6f})")
            return False

        # PyTorch 版本测试
        hdr_torch = torch.from_numpy(hdr).permute(2, 0, 1).unsqueeze(0)
        encoded_torch = HDRCodecTorch.encode_to_4channel(hdr_torch)
        recovered_torch = HDRCodecTorch.decode_from_4channel(encoded_torch)
        error_torch = (hdr_torch - recovered_torch).abs().max().item()

        if error_torch < 1e-4:
            print(f"✓ PyTorch HDR Codec test passed (error: {error_torch:.6f})")
        else:
            print(f"✗ PyTorch HDR Codec test failed (error: {error_torch:.6f})")
            return False

    except Exception as e:
        print(f"✗ HDR Codec test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_model():
    """测试模型前向传播"""
    print("=" * 60)
    print("Testing Delit Model...")
    print("=" * 60)

    try:
        from delit_model import SimplifiedDelitModel
        import torch

        # 创建模型
        model = SimplifiedDelitModel(base_dim=32)
        print(f"✓ Model created successfully")

        # 测试前向传播
        batch_size = 2
        relit = torch.randn(batch_size, 3, 512, 512)

        with torch.no_grad():
            flat_lit, env_encoded = model(relit)

        # 检查输出形状
        if flat_lit.shape == (batch_size, 3, 512, 512):
            print(f"✓ Flat-lit output shape correct: {flat_lit.shape}")
        else:
            print(f"✗ Flat-lit output shape incorrect: {flat_lit.shape}")
            return False

        if env_encoded.shape == (batch_size, 4, 128, 256):
            print(f"✓ Env encoded output shape correct: {env_encoded.shape}")
        else:
            print(f"✗ Env encoded output shape incorrect: {env_encoded.shape}")
            return False

        # 检查输出范围
        if flat_lit.min() >= -1.5 and flat_lit.max() <= 1.5:
            print(f"✓ Flat-lit output range reasonable: [{flat_lit.min():.2f}, {flat_lit.max():.2f}]")
        else:
            print(f"⚠ Flat-lit output range unusual: [{flat_lit.min():.2f}, {flat_lit.max():.2f}]")

        if env_encoded.min() >= -0.1 and env_encoded.max() <= 1.1:
            print(f"✓ Env encoded output range reasonable: [{env_encoded.min():.2f}, {env_encoded.max():.2f}]")
        else:
            print(f"⚠ Env encoded output range unusual: [{env_encoded.min():.2f}, {env_encoded.max():.2f}]")

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_loss():
    """测试损失函数"""
    print("=" * 60)
    print("Testing Loss Function...")
    print("=" * 60)

    try:
        from delit_loss import SimplifiedDelitLoss
        import torch

        # 创建损失函数
        criterion = SimplifiedDelitLoss()
        print(f"✓ Loss function created successfully")

        # 创建测试数据
        batch_size = 2
        pred_flat = torch.randn(batch_size, 3, 512, 512)
        gt_flat = torch.randn(batch_size, 3, 512, 512)
        pred_env = torch.rand(batch_size, 4, 128, 256)
        gt_env = torch.rand(batch_size, 4, 128, 256)
        mask = torch.rand(batch_size, 1, 512, 512) > 0.5

        # 计算损失
        total_loss, loss_dict = criterion(
            pred_flat, pred_env,
            gt_flat, gt_env,
            mask.float()
        )

        # 检查损失
        if torch.isfinite(total_loss) and total_loss.item() > 0:
            print(f"✓ Total loss computed: {total_loss.item():.4f}")
        else:
            print(f"✗ Total loss invalid: {total_loss.item()}")
            return False

        # 检查各项损失
        for key, value in loss_dict.items():
            if key != 'total' and torch.isfinite(value) and value.item() >= 0:
                print(f"✓ {key}: {value.item():.4f}")
            elif key != 'total':
                print(f"✗ {key}: {value.item()}")
                return False

    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_backward():
    """测试反向传播"""
    print("=" * 60)
    print("Testing Backward Propagation...")
    print("=" * 60)

    try:
        from delit_model import SimplifiedDelitModel
        from delit_loss import SimplifiedDelitLoss
        import torch

        # 创建模型和损失函数
        model = SimplifiedDelitModel(base_dim=32)
        criterion = SimplifiedDelitLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 创建测试数据
        relit = torch.randn(2, 3, 512, 512)
        gt_flat = torch.randn(2, 3, 512, 512)
        gt_env = torch.rand(2, 4, 128, 256)

        # 前向传播
        flat_lit, env_encoded = model(relit)
        loss, loss_dict = criterion(flat_lit, env_encoded, gt_flat, gt_env)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Backward propagation successful")
        print(f"✓ Gradient update successful")

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break

        if has_grad:
            print(f"✓ Gradients computed correctly")
        else:
            print(f"⚠ Warning: No non-zero gradients found")

    except Exception as e:
        print(f"✗ Backward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("FaceOLAT Delit System - Component Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("HDR Codec", test_hdr_codec),
        ("Model", test_model),
        ("Loss", test_loss),
        ("Backward", test_backward),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 打印总结
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {test_name}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
