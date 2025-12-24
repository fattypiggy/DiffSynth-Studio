
# 设置 Wan Video VAE 指南

本项目需要使用预训练的 Wan Video VAE。以下是完整的设置步骤。

## 📥 步骤 1: 下载 Wan Video VAE

### 方法 1: 从 Hugging Face 下载（推荐）

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 VAE 权重
# 注意：需要检查 Wan2.2 的实际 HuggingFace 仓库名称
huggingface-cli download Wan-Video/Wan2.2 --include "vae*.pth" --local-dir ./pretrained
```

### 方法 2: 从官方 GitHub 下载

访问 Wan2.2 官方仓库：
- https://github.com/Wan-Video/Wan2.2

查找 Release 或 Model Zoo 部分，下载 VAE checkpoint。

### 方法 3: 手动下载链接

如果上述方法不可用，查找官方提供的直接下载链接（通常在 README 或 release notes 中）。

## 📍 步骤 2: 放置权重文件

将下载的 VAE checkpoint 放在合适的位置：

```bash
# 推荐目录结构
DiffSynth-Studio/
├── diffsynth/
│   └── scripts/
└── pretrained/
    └── wan_vae.pth  # 或其他名称
```

## ✅ 步骤 3: 验证 VAE

使用提供的脚本验证 VAE 是否正确加载：

```bash
cd diffsynth/scripts

python load_wan_vae.py \
  --checkpoint ../../pretrained/wan_vae.pth \
  --test
```

**期望输出**:
```
Loading Wan Video VAE from: ../../pretrained/wan_vae.pth
✓ Wan Video VAE loaded successfully
  Device: cuda
  Z-dim: 16
  Upsampling factor: 8
  Parameters frozen: True (recommended)

Testing VAE...
  Encode: torch.Size([1, 3, 5, 512, 512]) -> torch.Size([1, 16, 2, 64, 64])
  Decode: torch.Size([1, 16, 2, 64, 64]) -> torch.Size([1, 3, 5, 512, 512])
✓ VAE test passed
```

## 🔍 常见问题排查

### Q1: 找不到 Wan Video VAE 下载链接

**A**: Wan2.2 可能还在开发中，或者权重在不同的地方。尝试：

1. 检查 Wan2.2 GitHub Issues
2. 查看 DiffSynth-Studio 主仓库的文档
3. 联系 Wan2.2 作者

### Q2: 加载 checkpoint 时出现 key mismatch

**A**: 这是正常的，因为我们只使用 VAE 部分。检查输出：

- `Missing keys`: 如果是 DiT 相关的 key，可以忽略
- `Unexpected keys`: 如果数量不多，通常可以忽略

只要最后显示 "VAE loaded successfully"，就说明加载成功。

### Q3: checkpoint 文件很大（> 10GB）

**A**: 完整的 Wan2.2 模型包含 DiT + VAE。我们只需要 VAE 部分。

如果没有单独的 VAE checkpoint：

1. 下载完整模型
2. 使用我们的脚本会自动提取 VAE 部分
3. （可选）之后可以删除完整模型，节省空间

### Q4: CUDA out of memory

**A**: VAE 可能比较大。尝试：

```python
# 在 load_wan_vae.py 中，使用 CPU 加载
vae = load_wan_vae(
    checkpoint_path="...",
    device='cpu'  # 使用 CPU
)
```

然后在训练时，VAE 会被冻结，内存占用较小。

## 🎯 下一步

VAE 设置完成后，你可以：

### 1. 测试完整流程

```bash
# 生成测试数据
python generate_toy_dataset.py \
  --output_dir ./toy_dataset \
  --num_subjects 2 \
  --num_envs 5

# 运行验证
python verify_modules.py
```

### 2. 开始训练

```bash
python train_delit_wan.py \
  --wan_vae_checkpoint ../../pretrained/wan_vae.pth \
  --data_root ./toy_dataset \
  --output_dir ./output/test_run \
  --batch_size 2 \
  --num_epochs 10 \
  --use_mask
```

## 📊 VAE 信息

### Wan Video VAE 规格

- **Latent 维度**: 16 通道
- **压缩率**: 8x (空间维度)
- **时间压缩**: 4x
- **输入**: [B, 3, T, 512, 512]
- **Latent**: [B, 16, T/4, 64, 64]

### 在 Delit 中的使用

```python
# Delit 使用单帧图像
Input: [B, 3, 512, 512]  # relit 图像

# 内部转换为视频格式（添加时间维度）
Internal: [B, 3, 1, 512, 512]

# VAE Encode
Latent: [B, 16, 1, 64, 64]

# 移除时间维度
Latent: [B, 16, 64, 64]  # 用于 Delit 处理
```

## 🔐 许可和引用

使用 Wan Video VAE 时，请遵守其许可证并引用相关论文。

---

**完成设置后，返回 QUICK_START.md 继续训练流程。**
