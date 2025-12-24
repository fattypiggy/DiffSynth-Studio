# FaceOLAT Delit 训练系统

这是一个完整的 Delit (De-lighting) 模型训练和推理系统，用于从 relit 图像中分离出 flat-lit 图像和环境光照图（environment map）。

## 概述

### 三阶段 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 1: Delit Training                     │
│  FaceOLAT: Relit(512×512) → Flat-lit + Env Map (HDR)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 2: Pseudo Labeling                       │
│  VFHQ videos → Delit Model → Flat-lit + Env Map (pseudo GT)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 3: Relit Training                       │
│  FaceOLAT + VFHQ: Flat-lit + Env Map → Relit Video             │
└─────────────────────────────────────────────────────────────────┘
```

### 核心技术

1. **HDR 编码方案**: Normalized RGB + Log Luminance (4 通道)
   - 完全可逆
   - 输出范围 [0, 1]，适合神经网络训练

2. **双分支架构**:
   - Flat-lit 分支: 输出去光照的人物图像
   - Env Map 分支: 输出环境光照的 HDR 表示

3. **损失函数**:
   - Flat-lit L1 + LPIPS（感知损失）
   - Env Map RGB L1 + Log Luminance MSE
   - 可选: 物理重建损失（需要 OLAT 数据）

## 文件结构

```
scripts/
├── README.md                    # 本文档
├── hdr_codec.py                # HDR 编解码器
├── delit_model.py              # Delit 模型定义
├── delit_loss.py               # 损失函数
├── delit_dataset.py            # 数据集
├── train_delit.py              # 训练脚本
├── inference_delit.py          # 推理脚本
└── requirements.txt            # 依赖
```

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐)

### 安装依赖

```bash
cd scripts
pip install -r requirements.txt
```

主要依赖：
- `torch`: PyTorch 深度学习框架
- `opencv-python`: 图像处理
- `numpy`: 数值计算
- `tqdm`: 进度条
- `tensorboard`: 训练可视化
- `lpips`: 感知损失（可选，推荐）
- `OpenEXR`: EXR 文件读写（可选，如果使用 EXR 格式）

## 数据准备

### 数据结构

将数据组织成以下结构：

```
data_root/
├── subjects/
│   ├── ID001/
│   │   ├── flat_lit.exr          # Flat-lit 图像（OLAT 全亮）
│   │   ├── mask.png               # 人物 mask
│   │   └── relit/
│   │       ├── env_001.exr        # 用 env_001 环境光照渲染的图像
│   │       ├── env_002.exr
│   │       └── ...
│   ├── ID002/
│   └── ...
└── env_maps/
    ├── env_001.hdr                # 环境图（HDR 格式）
    ├── env_002.hdr
    └── ...
```

### 数据格式

- **Relit 图像**: `.exr` 格式（线性 HDR）
- **Flat-lit 图像**: `.exr` 或 `.png` 格式
- **Mask**: `.png` 格式（灰度图，255 = 人物，0 = 背景）
- **Environment Maps**: `.hdr` 或 `.exr` 格式（线性 HDR）

### 从 FaceOLAT 生成数据

如果你已经使用 FaceOLAT 进行了 relit，确保：

1. 使用 `render_reference_envmap_relit.py` 融合人物和环境
2. 使用 RMBG-2-Studio 分离人物得到 mask
3. 保留 OLAT 全亮图像作为 flat-lit

## 训练

### 基础训练命令

```bash
cd scripts

python train_delit.py \
  --data_root /path/to/your/data \
  --output_dir ./output/exp1 \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --use_mask \
  --use_lpips
```

### 多 GPU 训练（推荐）

使用 PyTorch DDP 进行多 GPU 训练：

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train_delit.py \
  --data_root /path/to/your/data \
  --output_dir ./output/exp1 \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --use_mask \
  --use_lpips
```

### 恢复训练

```bash
python train_delit.py \
  --data_root /path/to/your/data \
  --output_dir ./output/exp1 \
  --resume ./output/exp1/checkpoint_latest.pth
```

### 主要参数说明

#### 数据参数
- `--data_root`: 数据根目录（必需）
- `--image_size`: 图像分辨率，默认 512
- `--env_height`: 环境图高度，默认 128
- `--env_width`: 环境图宽度，默认 256
- `--use_mask`: 是否使用 mask（推荐）
- `--use_augmentation`: 是否使用数据增强

#### 模型参数
- `--base_dim`: 模型基础维度，默认 64（可以调整为 32/64/128）
- `--log_scale`: HDR log 缩放因子，默认 10.0

#### 损失参数
- `--lambda_flat`: Flat-lit 损失权重，默认 1.0
- `--lambda_env`: Env map 损失权重，默认 1.0
- `--use_lpips`: 是否使用 LPIPS 感知损失（推荐）

#### 训练参数
- `--batch_size`: Batch size，默认 4
- `--num_epochs`: 训练轮数，默认 100
- `--learning_rate`: 学习率，默认 1e-4
- `--weight_decay`: 权重衰减，默认 1e-4
- `--num_workers`: 数据加载线程数，默认 4

### 监控训练

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir ./output/exp1/logs
```

在浏览器中打开 `http://localhost:6006` 查看：
- 训练损失曲线
- 学习率变化
- 各项损失的详细信息

## 推理

### 单张图像推理

```bash
python inference_delit.py \
  --checkpoint ./output/exp1/checkpoint_best.pth \
  --input /path/to/image.jpg \
  --output_dir ./inference_output
```

输出：
- `image_flat_lit.png`: Flat-lit 图像（LDR）
- `image_env.hdr`: 环境图（HDR）
- `image_env_preview.png`: 环境图预览（LDR）

### 视频推理

```bash
python inference_delit.py \
  --checkpoint ./output/exp1/checkpoint_best.pth \
  --input /path/to/video.mp4 \
  --output_dir ./inference_output
```

输出：
- `video_flat_lit.mp4`: Flat-lit 视频
- `video_env_frame_XXXXX.png`: 每秒的环境图预览
- `video_env_avg.hdr`: 平均环境图

### 批量推理（目录）

```bash
python inference_delit.py \
  --checkpoint ./output/exp1/checkpoint_best.pth \
  --input /path/to/image_folder \
  --output_dir ./inference_output
```

## 高级使用

### 测试 HDR 编解码器

```bash
cd scripts
python hdr_codec.py
```

这会测试 HDR 编解码器的可逆性，确保 HDR 数据可以准确还原。

### 测试模型

```bash
cd scripts
python delit_model.py
```

这会测试模型的前向传播，确保模型结构正确。

### 测试损失函数

```bash
cd scripts
python delit_loss.py
```

这会测试损失函数的计算，确保所有损失项都正常工作。

## 常见问题

### Q1: 训练时 GPU 内存不足

**A**: 尝试以下方法：
1. 减小 `--batch_size`（例如从 4 降到 2）
2. 减小 `--image_size`（例如从 512 降到 256）
3. 减小 `--base_dim`（例如从 64 降到 32）
4. 使用梯度累积（需要修改代码）

### Q2: LPIPS 损失导致训练不稳定

**A**:
1. 移除 `--use_lpips` 参数
2. 或者降低 LPIPS 权重（需要修改代码中的 `lambda_flat_lpips`）

### Q3: 环境图预测不准确

**A**:
1. 增加 `--lambda_env` 权重
2. 增加环境图分辨率 `--env_height` 和 `--env_width`
3. 确保训练数据的环境图质量良好
4. 增加训练数据量

### Q4: Flat-lit 图像有伪影

**A**:
1. 确保 mask 质量良好
2. 使用 `--use_mask` 参数
3. 检查训练数据是否对齐（relit 和 flat-lit）

### Q5: OpenEXR 安装失败

**A**:
OpenEXR 是可选的。如果安装失败：
1. 使用 `.hdr` 格式代替 `.exr`
2. 或者使用 conda 安装：`conda install -c conda-forge openexr-python`

## 下一步

训练完 Delit 模型后，你可以：

1. **对 VFHQ 进行推理**:
   ```bash
   python inference_delit.py \
     --checkpoint ./output/exp1/checkpoint_best.pth \
     --input /path/to/VFHQ_videos \
     --output_dir ./vfhq_delit
   ```

2. **使用生成的 pseudo GT 训练 Relit 模型**:
   - 将 Delit 输出的 flat-lit 和 env map 作为输入
   - 训练一个 relit 模型（Phase 3）
   - 这需要额外的 relit 模型代码（未包含在此脚本中）

## 技术细节

### HDR 编码

我们使用 Normalized RGB + Log Luminance 方案：

```
max_rgb = max(R, G, B)
normalized_rgb = [R, G, B] / max_rgb           # [0, 1]
log_luminance = log(1 + max_rgb) / log_scale   # [0, 1]
```

这样 HDR 图像被编码为 4 个通道，每个通道都在 [0, 1] 范围内，适合神经网络训练。

解码时：
```
max_rgb = exp(log_luminance * log_scale) - 1
hdr = normalized_rgb * max_rgb
```

### 模型架构

SimplifiedDelitModel 使用简单的 encoder-decoder 架构：

- **Encoder**: 4 层卷积下采样，512×512 → 64×64
- **Flat-lit Decoder**: 4 层卷积上采样，64×64 → 512×512
- **Env Map Decoder**: 2 层卷积上采样，64×64 → 128×256

### 损失函数

1. **Flat-lit L1**: 像素级重建损失
2. **Flat-lit LPIPS**: 感知损失，保证视觉质量
3. **Env RGB L1**: 环境图颜色重建
4. **Env Log MSE**: 环境图亮度重建（log 空间更稳定）

## 引用

如果你使用了这个代码，请引用相关论文：

- FaceOLAT (待发布)
- Wan2.2: https://github.com/Wan-Video/Wan2.2

## 许可

本代码遵循 MIT 许可证。

## 联系

如有问题，请通过 GitHub Issues 联系。
