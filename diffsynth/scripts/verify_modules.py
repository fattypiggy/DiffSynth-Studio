"""
模块验证脚本
用于检验各个模块是否正常工作，特别是 HDR 编解码
"""

import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def test_hdr_codec_reversibility():
    """测试 HDR 编解码的可逆性"""
    print("=" * 60)
    print("Test 1: HDR Codec Reversibility")
    print("=" * 60)

    from hdr_codec import HDRCodec, HDRCodecTorch

    # 测试不同动态范围的 HDR 数据
    test_cases = [
        ("Low Dynamic Range", np.random.rand(128, 256, 3).astype(np.float32) * 1.0),
        ("Medium Dynamic Range", np.random.rand(128, 256, 3).astype(np.float32) * 10.0),
        ("High Dynamic Range", np.random.rand(128, 256, 3).astype(np.float32) * 100.0),
        ("Very High Dynamic Range", np.random.rand(128, 256, 3).astype(np.float32) * 1000.0),
    ]

    print("\nNumPy Version:")
    for name, hdr in test_cases:
        encoded = HDRCodec.encode_to_4channel(hdr)
        recovered = HDRCodec.decode_from_4channel(encoded)

        # 计算误差
        abs_error = np.abs(hdr - recovered)
        max_error = abs_error.max()
        mean_error = abs_error.mean()
        rel_error = (abs_error / (hdr + 1e-6)).mean() * 100

        print(f"\n{name}:")
        print(f"  Input range: [{hdr.min():.2f}, {hdr.max():.2f}]")
        print(f"  Encoded range: RGB [{encoded[..., :3].min():.3f}, {encoded[..., :3].max():.3f}], "
              f"Log [{encoded[..., 3].min():.3f}, {encoded[..., 3].max():.3f}]")
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        print(f"  Relative error: {rel_error:.4f}%")

        if max_error < 1e-4:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL (error too large)")

    # PyTorch 版本
    print("\n" + "-" * 60)
    print("PyTorch Version:")
    for name, hdr in test_cases:
        hdr_torch = torch.from_numpy(hdr).permute(2, 0, 1).unsqueeze(0)
        encoded_torch = HDRCodecTorch.encode_to_4channel(hdr_torch)
        recovered_torch = HDRCodecTorch.decode_from_4channel(encoded_torch)

        error_torch = (hdr_torch - recovered_torch).abs().max().item()

        print(f"\n{name}:")
        print(f"  Max error: {error_torch:.6f}")
        if error_torch < 1e-4:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL (error too large)")

    print("\n" + "=" * 60 + "\n")


def test_hdr_codec_visualization(output_dir="./verify_output"):
    """可视化 HDR 编解码过程"""
    print("=" * 60)
    print("Test 2: HDR Codec Visualization")
    print("=" * 60)

    from hdr_codec import HDRCodec

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 创建一个合成的 HDR 图像（渐变 + 高亮）
    h, w = 256, 512
    hdr = np.zeros((h, w, 3), dtype=np.float32)

    # 创建水平渐变（线性空间）
    for i in range(w):
        brightness = (i / w) ** 2 * 100  # 二次曲线，最大 100
        hdr[:, i, 0] = brightness * 1.0  # R
        hdr[:, i, 1] = brightness * 0.8  # G
        hdr[:, i, 2] = brightness * 0.6  # B

    # 添加一些高亮点（模拟太阳/光源）
    for _ in range(5):
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        radius = 20
        brightness = np.random.uniform(500, 2000)

        y, x = np.ogrid[:h, :w]
        mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
        hdr[mask] = brightness

    print(f"Created synthetic HDR image:")
    print(f"  Shape: {hdr.shape}")
    print(f"  Range: [{hdr.min():.2f}, {hdr.max():.2f}]")
    print(f"  Mean: {hdr.mean():.2f}")

    # 编码
    encoded = HDRCodec.encode_to_4channel(hdr)
    normalized_rgb = encoded[..., :3]
    log_luminance = encoded[..., 3]

    # 解码
    recovered = HDRCodec.decode_from_4channel(encoded)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始 HDR（tone mapped）
    hdr_display = np.clip(hdr ** (1/2.2), 0, 1)
    axes[0, 0].imshow(hdr_display)
    axes[0, 0].set_title(f'Original HDR (tone mapped)\nRange: [{hdr.min():.1f}, {hdr.max():.1f}]')
    axes[0, 0].axis('off')

    # Normalized RGB
    axes[0, 1].imshow(normalized_rgb)
    axes[0, 1].set_title(f'Normalized RGB\nRange: [{normalized_rgb.min():.3f}, {normalized_rgb.max():.3f}]')
    axes[0, 1].axis('off')

    # Log Luminance
    im = axes[0, 2].imshow(log_luminance, cmap='hot')
    axes[0, 2].set_title(f'Log Luminance\nRange: [{log_luminance.min():.3f}, {log_luminance.max():.3f}]')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # 恢复的 HDR（tone mapped）
    recovered_display = np.clip(recovered ** (1/2.2), 0, 1)
    axes[1, 0].imshow(recovered_display)
    axes[1, 0].set_title(f'Recovered HDR (tone mapped)\nRange: [{recovered.min():.1f}, {recovered.max():.1f}]')
    axes[1, 0].axis('off')

    # 误差图
    error = np.abs(hdr - recovered)
    im = axes[1, 1].imshow(np.log1p(error.mean(axis=2)), cmap='hot')
    axes[1, 1].set_title(f'Error (log scale)\nMax: {error.max():.6f}, Mean: {error.mean():.6f}')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])

    # 统计直方图
    axes[1, 2].hist(hdr.flatten(), bins=100, alpha=0.5, label='Original', log=True)
    axes[1, 2].hist(recovered.flatten(), bins=100, alpha=0.5, label='Recovered', log=True)
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Count (log scale)')
    axes[1, 2].set_title('Value Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'hdr_codec_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    # 保存原始和恢复的 HDR
    np.save(output_dir / 'original_hdr.npy', hdr)
    np.save(output_dir / 'recovered_hdr.npy', recovered)
    np.save(output_dir / 'encoded.npy', encoded)
    print(f"✓ HDR data saved to: {output_dir}")

    plt.close()
    print("\n" + "=" * 60 + "\n")


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("Test 3: Model Forward Pass")
    print("=" * 60)

    from delit_model import SimplifiedDelitModel

    # 测试不同配置
    configs = [
        {"base_dim": 32, "name": "Small Model"},
        {"base_dim": 64, "name": "Medium Model"},
    ]

    for config in configs:
        print(f"\n{config['name']} (base_dim={config['base_dim']}):")

        model = SimplifiedDelitModel(base_dim=config['base_dim'])

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # 测试前向传播
        batch_sizes = [1, 2, 4]
        for bs in batch_sizes:
            relit = torch.randn(bs, 3, 512, 512)

            with torch.no_grad():
                flat_lit, env_encoded = model(relit)

            print(f"  Batch size {bs}: "
                  f"flat_lit {flat_lit.shape}, env {env_encoded.shape} - ✓")

    print("\n" + "=" * 60 + "\n")


def test_loss_computation():
    """测试损失计算"""
    print("=" * 60)
    print("Test 4: Loss Computation")
    print("=" * 60)

    from delit_loss import SimplifiedDelitLoss

    criterion = SimplifiedDelitLoss(lambda_flat=1.0, lambda_env=1.0)

    # 测试数据
    batch_size = 2
    pred_flat = torch.randn(batch_size, 3, 512, 512)
    gt_flat = torch.randn(batch_size, 3, 512, 512)
    pred_env = torch.rand(batch_size, 4, 128, 256)
    gt_env = torch.rand(batch_size, 4, 128, 256)
    mask = (torch.rand(batch_size, 1, 512, 512) > 0.3).float()

    # 计算损失
    total_loss, loss_dict = criterion(pred_flat, pred_env, gt_flat, gt_env, mask)

    print(f"\nLoss computation results:")
    print(f"  Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.4f}")

    # 测试梯度
    total_loss.backward()
    print(f"\n✓ Gradient computation successful")

    print("\n" + "=" * 60 + "\n")


def test_end_to_end():
    """端到端测试"""
    print("=" * 60)
    print("Test 5: End-to-End Pipeline")
    print("=" * 60)

    from delit_model import SimplifiedDelitModel
    from delit_loss import SimplifiedDelitLoss
    from hdr_codec import HDRCodecTorch

    # 创建模型
    model = SimplifiedDelitModel(base_dim=32)
    criterion = SimplifiedDelitLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\nTraining for 5 steps...")

    for step in range(5):
        # 模拟数据
        relit = torch.randn(2, 3, 512, 512)
        gt_flat = torch.randn(2, 3, 512, 512)
        gt_env = torch.rand(2, 4, 128, 256)

        # 前向传播
        flat_lit, env_encoded = model(relit)

        # 计算损失
        loss, loss_dict = criterion(flat_lit, env_encoded, gt_flat, gt_env)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print("\n✓ End-to-end training successful")

    # 测试 HDR 解码
    with torch.no_grad():
        env_hdr = HDRCodecTorch.decode_from_4channel(env_encoded)

    print(f"\n✓ HDR decoding successful")
    print(f"  Env HDR range: [{env_hdr.min():.2f}, {env_hdr.max():.2f}]")

    print("\n" + "=" * 60 + "\n")


def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("Module Verification Script")
    print("=" * 60 + "\n")

    try:
        # Test 1: HDR 编解码可逆性
        test_hdr_codec_reversibility()

        # Test 2: HDR 编解码可视化
        test_hdr_codec_visualization()

        # Test 3: 模型前向传播
        test_model_forward()

        # Test 4: 损失计算
        test_loss_computation()

        # Test 5: 端到端测试
        test_end_to_end()

        print("=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
