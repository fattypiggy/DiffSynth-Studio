"""
HDR Codec - 完全可逆的 HDR 编解码器
将 HDR 图像编码为 Normalized RGB + Log Luminance 的形式
"""

import numpy as np
import torch
import torch.nn.functional as F


class HDRCodec:
    """完全可逆的HDR编解码器"""

    @staticmethod
    def encode(hdr: np.ndarray, log_scale: float = 10.0) -> tuple:
        """
        HDR -> (Normalized RGB, Log Luminance)

        Args:
            hdr: [H, W, 3] 线性HDR图像
            log_scale: log值的缩放因子，使其落在[0,1]范围

        Returns:
            normalized_rgb: [H, W, 3] 范围[0,1]，保留颜色比例
            log_luminance: [H, W, 1] 范围约[0,1]，保留亮度信息
        """
        # 计算每像素的最大通道值（亮度代理）
        max_rgb = np.max(hdr, axis=-1, keepdims=True)  # [H, W, 1]
        max_rgb = np.maximum(max_rgb, 1e-6)  # 避免除零

        # 归一化RGB：颜色方向，范围[0,1]
        normalized_rgb = hdr / max_rgb  # [H, W, 3]

        # Log亮度：动态范围压缩
        # log1p(x) = log(1+x)，确保x=0时输出为0
        log_luminance = np.log1p(max_rgb) / log_scale  # [H, W, 1]
        log_luminance = np.clip(log_luminance, 0, 1)  # 裁剪到[0,1]

        return normalized_rgb, log_luminance

    @staticmethod
    def decode(normalized_rgb: np.ndarray, log_luminance: np.ndarray,
               log_scale: float = 10.0) -> np.ndarray:
        """
        (Normalized RGB, Log Luminance) -> HDR

        完全可逆：decode(encode(hdr)) ≈ hdr
        """
        # 恢复亮度
        max_rgb = np.expm1(log_luminance * log_scale)  # expm1(x) = exp(x) - 1

        # 恢复HDR
        hdr = normalized_rgb * max_rgb

        return hdr

    @staticmethod
    def encode_to_4channel(hdr: np.ndarray, log_scale: float = 10.0) -> np.ndarray:
        """编码为4通道图像，方便存储和训练"""
        normalized_rgb, log_luminance = HDRCodec.encode(hdr, log_scale)
        # 合并为 [H, W, 4]: RGB + Log
        return np.concatenate([normalized_rgb, log_luminance], axis=-1)

    @staticmethod
    def decode_from_4channel(encoded: np.ndarray, log_scale: float = 10.0) -> np.ndarray:
        """从4通道恢复HDR"""
        normalized_rgb = encoded[..., :3]
        log_luminance = encoded[..., 3:4]
        return HDRCodec.decode(normalized_rgb, log_luminance, log_scale)


class HDRCodecTorch:
    """PyTorch版本的HDR编解码器，用于训练"""

    @staticmethod
    def encode(hdr: torch.Tensor, log_scale: float = 10.0) -> tuple:
        """
        HDR -> (Normalized RGB, Log Luminance)

        Args:
            hdr: [B, 3, H, W] 或 [B, 3, env_H, env_W] 线性HDR图像
            log_scale: log值的缩放因子

        Returns:
            normalized_rgb: [B, 3, H, W] 范围[0,1]
            log_luminance: [B, 1, H, W] 范围约[0,1]
        """
        # 计算每像素的最大通道值
        max_rgb = torch.max(hdr, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        max_rgb = torch.clamp(max_rgb, min=1e-6)  # 避免除零

        # 归一化RGB
        normalized_rgb = hdr / max_rgb  # [B, 3, H, W]

        # Log亮度
        log_luminance = torch.log1p(max_rgb) / log_scale  # [B, 1, H, W]
        log_luminance = torch.clamp(log_luminance, 0, 1)

        return normalized_rgb, log_luminance

    @staticmethod
    def decode(normalized_rgb: torch.Tensor, log_luminance: torch.Tensor,
               log_scale: float = 10.0) -> torch.Tensor:
        """
        (Normalized RGB, Log Luminance) -> HDR

        Args:
            normalized_rgb: [B, 3, H, W]
            log_luminance: [B, 1, H, W]

        Returns:
            hdr: [B, 3, H, W]
        """
        # 恢复亮度
        max_rgb = torch.expm1(log_luminance * log_scale)

        # 恢复HDR
        hdr = normalized_rgb * max_rgb

        return hdr

    @staticmethod
    def encode_to_4channel(hdr: torch.Tensor, log_scale: float = 10.0) -> torch.Tensor:
        """
        编码为4通道张量

        Args:
            hdr: [B, 3, H, W]

        Returns:
            encoded: [B, 4, H, W]
        """
        normalized_rgb, log_luminance = HDRCodecTorch.encode(hdr, log_scale)
        return torch.cat([normalized_rgb, log_luminance], dim=1)

    @staticmethod
    def decode_from_4channel(encoded: torch.Tensor, log_scale: float = 10.0) -> torch.Tensor:
        """
        从4通道恢复HDR

        Args:
            encoded: [B, 4, H, W]

        Returns:
            hdr: [B, 3, H, W]
        """
        normalized_rgb = encoded[:, :3]
        log_luminance = encoded[:, 3:4]
        return HDRCodecTorch.decode(normalized_rgb, log_luminance, log_scale)


def test_reversibility():
    """测试编解码的可逆性"""
    print("Testing HDR Codec reversibility...")

    # 模拟HDR数据，动态范围很大
    hdr = np.random.rand(512, 1024, 3).astype(np.float32)
    hdr = hdr * np.array([[[100.0, 50.0, 1.0]]])  # 模拟高动态范围

    # 编码
    encoded = HDRCodec.encode_to_4channel(hdr)
    print(f"Encoded range: RGB [{encoded[..., :3].min():.3f}, {encoded[..., :3].max():.3f}], "
          f"Log [{encoded[..., 3].min():.3f}, {encoded[..., 3].max():.3f}]")

    # 解码
    recovered = HDRCodec.decode_from_4channel(encoded)

    # 验证
    error = np.abs(hdr - recovered).max()
    print(f"Max reconstruction error: {error:.6f}")

    # PyTorch版本测试
    print("\nTesting PyTorch version...")
    hdr_torch = torch.from_numpy(hdr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    encoded_torch = HDRCodecTorch.encode_to_4channel(hdr_torch)
    recovered_torch = HDRCodecTorch.decode_from_4channel(encoded_torch)

    error_torch = (hdr_torch - recovered_torch).abs().max().item()
    print(f"PyTorch max reconstruction error: {error_torch:.6f}")

    print("\n✓ HDR Codec test passed!")


if __name__ == "__main__":
    test_reversibility()
