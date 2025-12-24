"""
DelitLoss - Delit 模型的损失函数
包括 flat-lit 损失、env map 损失和可选的重建损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from .hdr_codec import HDRCodecTorch


try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")


class DelitLoss(nn.Module):
    """
    Delit 模型的完整损失函数

    损失组成：
    1. Flat-lit 损失：L1 + LPIPS（感知损失）
    2. Env Map 损失：Normalized RGB L1 + Log Luminance MSE
    3. 重建损失（可选）：用预测的 env + GT OLAT 重建 relit 图像
    """

    def __init__(
        self,
        lambda_flat_l1: float = 1.0,
        lambda_flat_lpips: float = 0.1,
        lambda_env_rgb: float = 1.0,
        lambda_env_log: float = 1.0,
        lambda_recon: float = 0.5,
        use_lpips: bool = True,
        log_scale: float = 10.0,
    ):
        """
        Args:
            lambda_flat_l1: Flat-lit L1 损失权重
            lambda_flat_lpips: Flat-lit LPIPS 损失权重
            lambda_env_rgb: Env map RGB 损失权重
            lambda_env_log: Env map log luminance 损失权重
            lambda_recon: 重建损失权重
            use_lpips: 是否使用 LPIPS
            log_scale: HDR 编码的 log 缩放因子
        """
        super().__init__()
        self.lambda_flat_l1 = lambda_flat_l1
        self.lambda_flat_lpips = lambda_flat_lpips
        self.lambda_env_rgb = lambda_env_rgb
        self.lambda_env_log = lambda_env_log
        self.lambda_recon = lambda_recon
        self.log_scale = log_scale

        # LPIPS 模型
        if use_lpips and LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='vgg').eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
            self.use_lpips = True
        else:
            self.use_lpips = False
            if use_lpips:
                print("Warning: LPIPS not available, skipping perceptual loss")

    def compute_flat_loss(
        self,
        pred_flat: torch.Tensor,
        gt_flat: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算 flat-lit 损失

        Args:
            pred_flat: [B, 3, H, W] 预测的 flat-lit，范围 [-1, 1]
            gt_flat: [B, 3, H, W] GT flat-lit，范围 [-1, 1]
            mask: [B, 1, H, W] 可选的 mask，只在人物区域计算损失

        Returns:
            losses: 损失字典
        """
        losses = {}

        # 应用 mask
        if mask is not None:
            pred_flat_masked = pred_flat * mask
            gt_flat_masked = gt_flat * mask
            # 归一化：只在 mask 区域计算
            mask_sum = mask.sum() + 1e-6
        else:
            pred_flat_masked = pred_flat
            gt_flat_masked = gt_flat
            mask_sum = pred_flat.numel() / pred_flat.shape[1]  # 总像素数 / 通道数

        # L1 损失
        flat_l1 = F.l1_loss(pred_flat_masked, gt_flat_masked, reduction='sum') / mask_sum
        losses['flat_l1'] = flat_l1 * self.lambda_flat_l1

        # LPIPS 感知损失
        if self.use_lpips:
            # LPIPS 需要范围 [-1, 1] 的输入
            # 将 mask 应用到整个图像，背景设为 0
            if mask is not None:
                pred_for_lpips = pred_flat * mask
                gt_for_lpips = gt_flat * mask
            else:
                pred_for_lpips = pred_flat
                gt_for_lpips = gt_flat

            with torch.no_grad():
                lpips_loss = self.lpips_model(pred_for_lpips, gt_for_lpips).mean()
            losses['flat_lpips'] = lpips_loss * self.lambda_flat_lpips

        return losses

    def compute_env_loss(
        self,
        pred_env_encoded: torch.Tensor,
        gt_env_encoded: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算 env map 损失

        Args:
            pred_env_encoded: [B, 4, env_H, env_W] 预测的编码 env map
            gt_env_encoded: [B, 4, env_H, env_W] GT 编码 env map

        Returns:
            losses: 损失字典
        """
        losses = {}

        # 分离 normalized RGB 和 log luminance
        pred_rgb = pred_env_encoded[:, :3]  # [B, 3, H, W]
        pred_log = pred_env_encoded[:, 3:4]  # [B, 1, H, W]
        gt_rgb = gt_env_encoded[:, :3]
        gt_log = gt_env_encoded[:, 3:4]

        # RGB 损失：L1
        env_rgb_loss = F.l1_loss(pred_rgb, gt_rgb)
        losses['env_rgb'] = env_rgb_loss * self.lambda_env_rgb

        # Log luminance 损失：MSE（log 空间更稳定）
        env_log_loss = F.mse_loss(pred_log, gt_log)
        losses['env_log'] = env_log_loss * self.lambda_env_log

        return losses

    def compute_reconstruction_loss(
        self,
        pred_flat: torch.Tensor,
        pred_env_encoded: torch.Tensor,
        relit_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        olat_images: Optional[torch.Tensor] = None,
        indexmap: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算重建损失（物理约束）

        用预测的 env map 和 GT 的 OLAT 图像重建 relit 图像，
        这确保预测的 env map 是"物理正确的"

        Args:
            pred_flat: [B, 3, H, W] 预测的 flat-lit
            pred_env_encoded: [B, 4, env_H, env_W] 预测的编码 env map
            relit_image: [B, 3, H, W] 输入的 relit 图像
            mask: [B, 1, H, W] 人物 mask
            olat_images: [331, 3, H, W] OLAT 基图像（可选）
            indexmap: [env_H, env_W] Voronoi 索引图（可选）

        Returns:
            losses: 损失字典
        """
        losses = {}

        if olat_images is None or indexmap is None:
            # 如果没有 OLAT 数据，跳过重建损失
            return losses

        # 1. 解码 env map 到 HDR
        pred_env_hdr = HDRCodecTorch.decode_from_4channel(
            pred_env_encoded, self.log_scale
        )  # [B, 3, env_H, env_W]

        # 2. 将 HDR env map 投影到 OLAT 系数
        olat_coeffs = self.project_env_to_olat(
            pred_env_hdr, indexmap
        )  # [B, 331, 3]

        # 3. 用 OLAT 系数和基图像渲染
        rerendered = self.render_with_olat(
            olat_images, olat_coeffs
        )  # [B, 3, H, W]

        # 4. 计算重建损失
        if mask is not None:
            rerendered_masked = rerendered * mask
            relit_masked = relit_image * mask
            mask_sum = mask.sum() + 1e-6
        else:
            rerendered_masked = rerendered
            relit_masked = relit_image
            mask_sum = relit_image.numel() / relit_image.shape[1]

        recon_loss = F.l1_loss(rerendered_masked, relit_masked, reduction='sum') / mask_sum
        losses['recon'] = recon_loss * self.lambda_recon

        return losses

    def project_env_to_olat(
        self,
        env_hdr: torch.Tensor,
        indexmap: torch.Tensor
    ) -> torch.Tensor:
        """
        将 HDR 环境图投影到 OLAT 系数

        Args:
            env_hdr: [B, 3, env_H, env_W] HDR 环境图
            indexmap: [env_H, env_W] Voronoi 索引图，值为 0-330

        Returns:
            coeffs: [B, 331, 3] OLAT 系数
        """
        B, C, env_H, env_W = env_hdr.shape
        device = env_hdr.device

        # 计算 solid angle weights（等距柱状投影）
        theta = (torch.arange(env_H, device=device) + 0.5) * (np.pi / env_H)
        solid_angle = torch.sin(theta) * 2.0 * np.pi / env_W  # [env_H]
        solid_angle = solid_angle.view(1, 1, env_H, 1)  # [1, 1, env_H, 1]

        # 应用权重
        weighted_env = env_hdr * solid_angle  # [B, 3, env_H, env_W]

        # 按 Voronoi 区域累加
        coeffs = torch.zeros(B, 331, 3, device=device, dtype=env_hdr.dtype)

        # 将 indexmap 移到正确的设备
        indexmap = indexmap.to(device)

        for i in range(331):
            mask = (indexmap == i).float()  # [env_H, env_W]
            # 在空间维度上求和
            # weighted_env: [B, 3, env_H, env_W]
            # mask: [env_H, env_W]
            coeffs[:, i] = (weighted_env * mask[None, None]).sum(dim=(2, 3))  # [B, 3]

        return coeffs

    def render_with_olat(
        self,
        olat_images: torch.Tensor,
        olat_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        使用 OLAT 图像和系数渲染

        Args:
            olat_images: [331, 3, H, W] OLAT 基图像
            olat_coeffs: [B, 331, 3] OLAT 系数

        Returns:
            rendered: [B, 3, H, W] 渲染的图像
        """
        # olat_images: [331, 3, H, W]
        # olat_coeffs: [B, 331, 3]

        # 使用 einsum 进行加权求和
        # n: 331个方向, c: 3个颜色通道, h,w: 空间维度, b: batch
        rendered = torch.einsum('nchw,bnc->bchw', olat_images, olat_coeffs)
        rendered = rendered / np.sqrt(331)  # 归一化

        return rendered

    def forward(
        self,
        pred_flat: torch.Tensor,
        pred_env_encoded: torch.Tensor,
        gt_flat: torch.Tensor,
        gt_env_encoded: torch.Tensor,
        relit_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        olat_images: Optional[torch.Tensor] = None,
        indexmap: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        计算总损失

        Args:
            pred_flat: [B, 3, H, W] 预测的 flat-lit
            pred_env_encoded: [B, 4, env_H, env_W] 预测的编码 env map
            gt_flat: [B, 3, H, W] GT flat-lit
            gt_env_encoded: [B, 4, env_H, env_W] GT 编码 env map
            relit_image: [B, 3, H, W] 输入的 relit 图像（用于重建损失）
            mask: [B, 1, H, W] 人物 mask
            olat_images: [331, 3, H, W] OLAT 基图像（用于重建损失）
            indexmap: [env_H, env_W] Voronoi 索引图（用于重建损失）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. Flat-lit 损失
        flat_losses = self.compute_flat_loss(pred_flat, gt_flat, mask)
        loss_dict.update(flat_losses)

        # 2. Env map 损失
        env_losses = self.compute_env_loss(pred_env_encoded, gt_env_encoded)
        loss_dict.update(env_losses)

        # 3. 重建损失（如果有 OLAT 数据）
        if relit_image is not None and olat_images is not None:
            recon_losses = self.compute_reconstruction_loss(
                pred_flat, pred_env_encoded, relit_image,
                mask, olat_images, indexmap
            )
            loss_dict.update(recon_losses)

        # 总损失
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss

        return total_loss, loss_dict


class SimplifiedDelitLoss(nn.Module):
    """
    简化版本的 Delit 损失，只包含 flat-lit 和 env map 损失
    不需要 OLAT 数据，更容易使用
    """

    def __init__(
        self,
        lambda_flat: float = 1.0,
        lambda_env: float = 1.0,
        use_lpips: bool = False,
    ):
        super().__init__()
        self.lambda_flat = lambda_flat
        self.lambda_env = lambda_env

        if use_lpips and LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='vgg').eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
            self.use_lpips = True
        else:
            self.use_lpips = False

    def forward(
        self,
        pred_flat: torch.Tensor,
        pred_env: torch.Tensor,
        gt_flat: torch.Tensor,
        gt_env: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            pred_flat: [B, 3, H, W] 预测的 flat-lit，范围 [-1, 1]
            pred_env: [B, 4, env_H, env_W] 预测的 env，范围 [0, 1]
            gt_flat: [B, 3, H, W] GT flat-lit
            gt_env: [B, 4, env_H, env_W] GT env
            mask: [B, 1, H, W] 可选的 mask

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        loss_dict = {}

        # Flat-lit 损失
        if mask is not None:
            flat_loss = F.l1_loss(pred_flat * mask, gt_flat * mask)
        else:
            flat_loss = F.l1_loss(pred_flat, gt_flat)

        loss_dict['flat_l1'] = flat_loss * self.lambda_flat

        # Env map 损失
        env_loss = F.l1_loss(pred_env, gt_env)
        loss_dict['env_l1'] = env_loss * self.lambda_env

        # LPIPS（如果启用）
        if self.use_lpips:
            if mask is not None:
                pred_for_lpips = pred_flat * mask
                gt_for_lpips = gt_flat * mask
            else:
                pred_for_lpips = pred_flat
                gt_for_lpips = gt_flat

            with torch.no_grad():
                lpips_loss = self.lpips_model(pred_for_lpips, gt_for_lpips).mean()
            loss_dict['flat_lpips'] = lpips_loss * 0.1

        # 总损失
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss

        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试损失函数
    print("Testing SimplifiedDelitLoss...")

    loss_fn = SimplifiedDelitLoss()

    # 创建测试数据
    B, H, W = 2, 512, 512
    env_H, env_W = 128, 256

    pred_flat = torch.randn(B, 3, H, W)
    gt_flat = torch.randn(B, 3, H, W)
    pred_env = torch.rand(B, 4, env_H, env_W)
    gt_env = torch.rand(B, 4, env_H, env_W)
    mask = torch.rand(B, 1, H, W) > 0.5

    # 计算损失
    total_loss, loss_dict = loss_fn(pred_flat, pred_env, gt_flat, gt_env, mask.float())

    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.4f}")

    print("\n✓ Loss test passed!")
