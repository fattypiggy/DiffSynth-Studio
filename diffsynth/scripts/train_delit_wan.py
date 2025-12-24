"""
训练 Delit 模型的主脚本（使用预训练 Wan Video VAE）
支持单 GPU 和多 GPU (DDP) 训练
"""

import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wan_video_vae import WanVideoVAE
from load_wan_vae import load_wan_vae
from delit_model import DelitDiT
from delit_loss import SimplifiedDelitLoss
from delit_dataset import FaceOLATDelitDataset
from hdr_codec import HDRCodecTorch


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    save_path: Path,
    is_best: bool = False
):
    """保存 checkpoint"""
    # 如果是 DDP 模型，保存原始模型
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    # 保存最新的 checkpoint
    torch.save(checkpoint, save_path / 'checkpoint_latest.pth')

    # 每10个epoch保存一次
    if epoch % 10 == 0:
        torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pth')

    # 保存最佳模型
    if is_best:
        torch.save(checkpoint, save_path / 'checkpoint_best.pth')


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    writer: SummaryWriter = None
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()

    total_loss = 0.0
    loss_dict_total = {}
    num_batches = len(dataloader)

    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备
        relit = batch['relit'].to(device)
        flat_lit_gt = batch['flat_lit'].to(device)
        env_encoded_gt = batch['env_encoded'].to(device)
        mask = batch['mask'].to(device)

        # 前向传播
        flat_lit_pred, env_encoded_pred = model(relit)

        # 计算损失
        loss, loss_dict = criterion(
            flat_lit_pred, env_encoded_pred,
            flat_lit_gt, env_encoded_gt,
            mask
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 累积损失
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_dict_total:
                loss_dict_total[key] = 0.0
            loss_dict_total[key] += value.item()

        # 更新进度条
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})

        # 记录到 tensorboard
        if rank == 0 and writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)

    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_total.items()}

    # 在多 GPU 情况下同步损失
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

    return {'total': avg_loss, **avg_loss_dict}


def main(args):
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只在主进程创建 tensorboard writer
    writer = None
    if rank == 0:
        log_dir = output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)

        # 保存配置
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # 加载预训练的 Wan Video VAE
    if rank == 0:
        print("=" * 60)
        print("Loading pretrained Wan Video VAE...")
        print("=" * 60)

    vae = load_wan_vae(
        checkpoint_path=args.wan_vae_checkpoint,
        device=device,
        z_dim=16
    )

    # 创建数据集
    if rank == 0:
        print(f"\nLoading dataset from {args.data_root}")

    train_dataset = FaceOLATDelitDataset(
        data_root=args.data_root,
        env_resolution=(args.env_height, args.env_width),
        image_size=args.image_size,
        log_scale=args.log_scale,
        use_mask=args.use_mask,
        use_augmentation=args.use_augmentation,
    )

    # 创建数据加载器
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if rank == 0:
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches per epoch: {len(train_loader)}")

    # 创建模型
    if rank == 0:
        print("\n" + "=" * 60)
        print("Creating DelitDiT model...")
        print("=" * 60)

    model = DelitDiT(
        vae=vae,
        env_resolution=(args.env_height, args.env_width),
        log_scale=args.log_scale,
        freeze_vae=args.freeze_vae
    ).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters (VAE): {frozen_params:,}")

    # 创建 DDP 模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 创建损失函数
    criterion = SimplifiedDelitLoss(
        lambda_flat=args.lambda_flat,
        lambda_env=args.lambda_env,
        use_lpips=args.use_lpips
    ).to(device)

    # 创建优化器（只优化 trainable 参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )

    # 加载 checkpoint（如果存在）
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            if rank == 0:
                print(f"\nLoading checkpoint from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('loss', float('inf'))

            if rank == 0:
                print(f"Resumed from epoch {start_epoch}")

    # 训练循环
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        print(f"Training for {args.num_epochs} epochs")
        print()

    for epoch in range(start_epoch, args.num_epochs):
        # 设置 epoch（用于 DistributedSampler）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, rank, writer
        )

        # 更新学习率
        scheduler.step()

        # 记录
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.num_epochs}")
            print(f"  Train Loss: {train_loss_dict['total']:.4f}")
            for key, value in train_loss_dict.items():
                if key != 'total':
                    print(f"    {key}: {value:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # TensorBoard
            if writer is not None:
                writer.add_scalar('train/loss_epoch', train_loss_dict['total'], epoch)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
                for key, value in train_loss_dict.items():
                    if key != 'total':
                        writer.add_scalar(f'train/{key}', value, epoch)

            # 保存 checkpoint
            is_best = train_loss_dict['total'] < best_loss
            if is_best:
                best_loss = train_loss_dict['total']

            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss_dict['total'],
                output_dir, is_best
            )

    # 清理
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        if writer is not None:
            writer.close()

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Delit Model with Wan Video VAE')

    # 预训练模型
    parser.add_argument('--wan_vae_checkpoint', type=str, required=True,
                        help='Path to pretrained Wan Video VAE checkpoint')
    parser.add_argument('--freeze_vae', action='store_true', default=True,
                        help='Freeze VAE parameters (recommended)')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--image_size', type=int, default=512, help='图像分辨率')
    parser.add_argument('--env_height', type=int, default=128, help='环境图高度')
    parser.add_argument('--env_width', type=int, default=256, help='环境图宽度')
    parser.add_argument('--use_mask', action='store_true', help='是否使用 mask')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')

    # 模型参数
    parser.add_argument('--log_scale', type=float, default=10.0, help='HDR log 缩放因子')

    # 损失函数参数
    parser.add_argument('--lambda_flat', type=float, default=1.0, help='Flat-lit 损失权重')
    parser.add_argument('--lambda_env', type=float, default=1.0, help='Env map 损失权重')
    parser.add_argument('--use_lpips', action='store_true', help='是否使用 LPIPS')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的 checkpoint 路径')

    args = parser.parse_args()

    main(args)
