"""
加载 Wan2.2 预训练 VAE 的辅助脚本
"""

import sys
import torch
from pathlib import Path

# 添加 DiffSynth 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.wan_video_vae import WanVideoVAE
except ImportError:
    print("Error: Cannot import WanVideoVAE from diffsynth.models")
    print("Make sure you are in the DiffSynth-Studio repository")
    sys.exit(1)


def load_wan_vae(
    checkpoint_path: str,
    device: str = 'cuda',
    z_dim: int = 16
):
    """
    加载预训练的 Wan Video VAE

    Args:
        checkpoint_path: Wan VAE checkpoint 路径
        device: 设备 ('cuda' 或 'cpu')
        z_dim: VAE latent 维度，默认 16

    Returns:
        vae: 加载好的 WanVideoVAE 模型
    """
    print(f"Loading Wan Video VAE from: {checkpoint_path}")

    # 创建 VAE 模型
    vae = WanVideoVAE(z_dim=z_dim)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 处理不同格式的 checkpoint
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 使用 state dict converter（如果需要）
    converter = vae.state_dict_converter()
    if hasattr(converter, 'from_civitai'):
        # 假设是 civitai 格式
        state_dict = converter.from_civitai({'model_state': state_dict})

    # 加载权重
    missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")  # 只显示前5个
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")

    # 移到设备
    vae = vae.to(device)
    vae.eval()

    # 冻结 VAE 参数（推荐）
    for param in vae.parameters():
        param.requires_grad = False

    print(f"✓ Wan Video VAE loaded successfully")
    print(f"  Device: {device}")
    print(f"  Z-dim: {z_dim}")
    print(f"  Upsampling factor: {vae.upsampling_factor}")
    print(f"  Parameters frozen: True (recommended)")

    return vae


def test_vae(vae, device='cuda'):
    """测试 VAE 是否正常工作"""
    print("\nTesting VAE...")

    # 创建测试视频
    test_video = torch.randn(1, 3, 5, 512, 512).to(device)  # [B, C, T, H, W]

    with torch.no_grad():
        # Encode
        latent = vae.model.encode(test_video, vae.scale)
        print(f"  Encode: {test_video.shape} -> {latent.shape}")

        # Decode
        reconstructed = vae.model.decode(latent, vae.scale)
        print(f"  Decode: {latent.shape} -> {reconstructed.shape}")

    print("✓ VAE test passed")


def main():
    """示例：加载和测试 VAE"""
    import argparse

    parser = argparse.ArgumentParser(description='Load Wan Video VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Wan VAE checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--test', action='store_true',
                        help='Run test after loading')

    args = parser.parse_args()

    # 加载 VAE
    vae = load_wan_vae(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 测试
    if args.test:
        test_vae(vae, device=args.device)

    print("\n" + "=" * 60)
    print("VAE loaded successfully!")
    print("You can now use this VAE in your Delit model:")
    print("=" * 60)
    print("""
from load_wan_vae import load_wan_vae
from delit_model import DelitDiT

# Load VAE
vae = load_wan_vae('path/to/wan_vae.pth')

# Create Delit model
model = DelitDiT(
    vae=vae,
    env_resolution=(128, 256),
    log_scale=10.0
)
    """)


if __name__ == '__main__':
    main()
