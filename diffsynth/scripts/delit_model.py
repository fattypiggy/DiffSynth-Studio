"""
DelitDiT - 基于 Wan Video VAE 的 Delit 模型
用于从 relit 图像中分离出 flat-lit 和 environment map

要求：
- 使用预训练的 Wan Video VAE
- VAE 参数建议冻结
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 添加父目录到路径，以便导入 diffsynth 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wan_video_vae import WanVideoVAE
from .hdr_codec import HDRCodecTorch


class EnvMapDecoder(nn.Module):
    """
    环境图解码器
    从 VAE latent space 解码到 HDR 环境图
    """

    def __init__(
        self,
        latent_dim: int = 16,  # Wan VAE latent 维度
        latent_size: Tuple[int, int] = (64, 64),  # Latent 空间大小
        env_resolution: Tuple[int, int] = (128, 256)
    ):
        """
        Args:
            latent_dim: VAE latent 通道数，Wan VAE 默认 16
            latent_size: Latent 空间大小，512 图像对应 (64, 64)
            env_resolution: 目标环境图分辨率 (height, width)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.env_h, self.env_w = env_resolution

        # Decoder: latent (16, 64, 64) -> env (4, 128, 256)
        self.decoder = nn.Sequential(
            # 64×64 -> 64×64 (增加通道)
            nn.Conv2d(latent_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),

            # 64×64 -> 128×128
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 128×128 -> 128×256 (调整纵横比)
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Upsample(size=(self.env_h, self.env_w), mode='bilinear', align_corners=False),

            # 细化特征
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),

            # 输出 4 通道: normalized_rgb (3) + log_luminance (1)
            nn.Conv2d(32, 4, 3, padding=1),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim, H, W] VAE latent，通常是 [B, 16, 64, 64]

        Returns:
            env_encoded: [B, 4, env_h, env_w] 编码的环境图
        """
        return self.decoder(latent)


class DelitDiT(nn.Module):
    """
    基于 Wan Video VAE 的 Delit 模型

    架构：
    1. 使用预训练的 Wan Video VAE encoder 编码 relit 图像
    2. 通过特征提取网络处理 latent
    3. 分两个分支：
       - Flat-lit 分支：通过 VAE decoder 解码
       - Env Map 分支：通过专门的 decoder 解码为 HDR 环境图

    要求：
    - 输入：Relit 图像 [B, 3, H, W]
    - 输出：Flat-lit [B, 3, H, W] + Env Map [B, 4, env_h, env_w]
    """

    def __init__(
        self,
        vae: WanVideoVAE,
        env_resolution: Tuple[int, int] = (128, 256),
        log_scale: float = 10.0,
        freeze_vae: bool = True
    ):
        """
        Args:
            vae: 预训练的 WanVideoVAE 模型
            env_resolution: 环境图分辨率
            log_scale: HDR 编码的 log 缩放因子
            freeze_vae: 是否冻结 VAE 参数（推荐）
        """
        super().__init__()
        self.vae = vae
        self.log_scale = log_scale
        self.env_resolution = env_resolution

        # 冻结 VAE
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        # VAE latent 维度
        vae_latent_dim = self.vae.z_dim  # 通常是 16

        # 特征提取网络：处理 VAE latent
        # 将 relit latent 转换为适合两个分支的特征
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(vae_latent_dim, vae_latent_dim * 2, 3, padding=1),
            nn.GroupNorm(16, vae_latent_dim * 2),
            nn.SiLU(),

            nn.Conv2d(vae_latent_dim * 2, vae_latent_dim * 2, 3, padding=1),
            nn.GroupNorm(16, vae_latent_dim * 2),
            nn.SiLU(),

            nn.Conv2d(vae_latent_dim * 2, vae_latent_dim * 2, 3, padding=1),
            nn.GroupNorm(16, vae_latent_dim * 2),
            nn.SiLU(),
        )

        # 两个分支的投影层
        self.flat_proj = nn.Sequential(
            nn.Conv2d(vae_latent_dim * 2, vae_latent_dim, 3, padding=1),
            nn.GroupNorm(8, vae_latent_dim),
            nn.SiLU(),
            nn.Conv2d(vae_latent_dim, vae_latent_dim, 3, padding=1),
        )

        self.env_proj = nn.Sequential(
            nn.Conv2d(vae_latent_dim * 2, vae_latent_dim, 3, padding=1),
            nn.GroupNorm(8, vae_latent_dim),
            nn.SiLU(),
            nn.Conv2d(vae_latent_dim, vae_latent_dim, 3, padding=1),
        )

        # Env Map decoder
        self.env_decoder = EnvMapDecoder(
            latent_dim=vae_latent_dim,
            env_resolution=env_resolution
        )

    def forward(self, relit_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            relit_image: [B, 3, H, W] relit 图像，范围通常 [-1, 1] 或 [0, 1]

        Returns:
            flat_lit: [B, 3, H, W] flat-lit 图像
            env_encoded: [B, 4, env_h, env_w] 编码的环境图
        """
        B, C, H, W = relit_image.shape

        # 1. VAE Encode
        # 需要添加时间维度给 VAE（它期望 5D 输入）
        relit_video = relit_image.unsqueeze(2)  # [B, 3, 1, H, W]

        # 使用 VAE encode
        with torch.set_grad_enabled(not self.training):  # 如果冻结 VAE，不计算梯度
            z = self.vae.model.encode(relit_video, self.vae.scale)  # [B, C, 1, h, w]

        # 移除时间维度
        z = z.squeeze(2)  # [B, C, h, w]

        # 2. 特征提取
        features = self.feature_extractor(z)  # [B, C*2, h, w]

        # 3. 分成两个分支
        flat_latent = self.flat_proj(features)  # [B, C, h, w]
        env_latent = self.env_proj(features)    # [B, C, h, w]

        # 4. Flat-lit: 通过 VAE decoder
        flat_latent_video = flat_latent.unsqueeze(2)  # [B, C, 1, h, w]

        with torch.set_grad_enabled(not self.training):
            flat_lit_video = self.vae.model.decode(flat_latent_video, self.vae.scale)

        flat_lit = flat_lit_video.squeeze(2)  # [B, 3, H, W]

        # 5. Env map: 通过专门的 decoder
        env_encoded = self.env_decoder(env_latent)  # [B, 4, env_h, env_w]

        return flat_lit, env_encoded

    def decode_env_to_hdr(self, env_encoded: torch.Tensor) -> torch.Tensor:
        """
        将编码的 env map 解码为 HDR

        Args:
            env_encoded: [B, 4, env_h, env_w]

        Returns:
            env_hdr: [B, 3, env_h, env_w]
        """
        return HDRCodecTorch.decode_from_4channel(env_encoded, self.log_scale)


if __name__ == "__main__":
    print("Testing DelitDiT model...")
    print("Note: This requires a pretrained Wan Video VAE")
    print()

    # 创建一个 dummy VAE 用于测试
    print("Creating Wan Video VAE...")
    vae = WanVideoVAE(z_dim=16)
    print(f"  VAE latent dim: {vae.z_dim}")
    print(f"  VAE upsampling factor: {vae.upsampling_factor}")

    # 创建 Delit 模型
    print("\nCreating DelitDiT model...")
    model = DelitDiT(
        vae=vae,
        env_resolution=(128, 256),
        log_scale=10.0,
        freeze_vae=True
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # 测试前向传播
    print("\nTesting forward pass...")
    batch_size = 2
    relit = torch.randn(batch_size, 3, 512, 512)

    with torch.no_grad():
        flat_lit, env_encoded = model(relit)

    print(f"  Input: {relit.shape}")
    print(f"  Output flat-lit: {flat_lit.shape}")
    print(f"  Output env encoded: {env_encoded.shape}")

    # 测试 HDR 解码
    env_hdr = model.decode_env_to_hdr(env_encoded)
    print(f"  Output env HDR: {env_hdr.shape}")

    print("\n✓ DelitDiT model test passed!")
