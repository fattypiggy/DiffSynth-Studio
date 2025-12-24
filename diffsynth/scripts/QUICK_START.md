# Quick Start Guide - FaceOLAT Delit 训练系统

## 🎉 已创建的文件

所有文件都在 `diffsynth/scripts/` 文件夹中：

### 核心代码
1. **hdr_codec.py** - HDR 编解码器
   - 实现 Normalized RGB + Log Luminance 编码方案
   - 支持 NumPy 和 PyTorch 版本
   - 完全可逆

2. **delit_model.py** - Delit 模型
   - `SimplifiedDelitModel`: 简化版模型（推荐）
   - `DelitDiT`: 基于 Wan Video VAE 的版本
   - 双分支架构：Flat-lit + Env Map

3. **delit_loss.py** - 损失函数
   - `SimplifiedDelitLoss`: 简化版损失（推荐）
   - `DelitLoss`: 完整版损失（包含 OLAT 重建）

4. **delit_dataset.py** - 数据集
   - `FaceOLATDelitDataset`: 数据加载和预处理
   - 支持 EXR 和 HDR 格式

### 训练和推理
5. **train_delit.py** - 训练脚本
   - 支持单 GPU 和多 GPU (DDP) 训练
   - TensorBoard 可视化
   - 自动保存 checkpoint

6. **inference_delit.py** - 推理脚本
   - 支持单张图像、视频和批量推理
   - 输出 flat-lit 和 env map

### 示例脚本
7. **example_train.sh** - 训练示例脚本
8. **example_inference.sh** - 推理示例脚本

### 文档和测试
9. **README.md** - 完整文档
10. **requirements.txt** - 依赖列表
11. **test_all.py** - 组件测试
12. **__init__.py** - Python 包初始化

## 🚀 快速开始

### 1. 安装依赖

```bash
cd diffsynth/scripts
pip install -r requirements.txt
```

如果安装失败，至少需要这些核心依赖：
```bash
pip install torch torchvision numpy opencv-python tqdm tensorboard
```

### 2. 准备数据

按照以下结构组织你的 FaceOLAT 数据：

```
your_data/
├── subjects/
│   ├── ID001/
│   │   ├── flat_lit.exr       # Flat-lit 图像
│   │   ├── mask.png            # 人物 mask
│   │   └── relit/
│   │       ├── env_001.exr     # Relit 图像
│   │       ├── env_002.exr
│   │       └── ...
│   └── ...
└── env_maps/
    ├── env_001.hdr             # 环境图
    ├── env_002.hdr
    └── ...
```

### 3. 测试组件（可选）

```bash
cd diffsynth/scripts
python test_all.py
```

如果所有测试通过，说明系统安装正确。

### 4. 训练模型

#### 方法 1: 使用示例脚本（推荐）

编辑 `example_train.sh`，修改数据路径：
```bash
nano example_train.sh
# 修改 DATA_ROOT="/path/to/your/FaceOLAT_data"
```

运行训练：
```bash
bash example_train.sh
```

#### 方法 2: 直接使用 Python

单 GPU:
```bash
python train_delit.py \
  --data_root /path/to/your/data \
  --output_dir ./output/exp1 \
  --batch_size 4 \
  --num_epochs 100 \
  --use_mask
```

多 GPU (4 GPUs):
```bash
torchrun --nproc_per_node=4 train_delit.py \
  --data_root /path/to/your/data \
  --output_dir ./output/exp1 \
  --batch_size 4 \
  --num_epochs 100 \
  --use_mask
```

### 5. 监控训练

```bash
tensorboard --logdir ./output/exp1/logs
```

在浏览器中打开 http://localhost:6006

### 6. 进行推理

对 VFHQ 或其他视频进行推理：

```bash
python inference_delit.py \
  --checkpoint ./output/exp1/checkpoint_best.pth \
  --input /path/to/VFHQ_videos \
  --output_dir ./inference_output
```

## 📊 Pipeline 概览

### Phase 1: Delit Training (当前)
```
FaceOLAT Relit → [Delit Model] → Flat-lit + Env Map
```

**输入**: Relit 图像 (512×512)
**输出**:
- Flat-lit 图像 (512×512)
- Env Map HDR (128×256)

### Phase 2: Pseudo Labeling (下一步)
```
VFHQ Videos → [Trained Delit] → Pseudo GT (Flat-lit + Env Map)
```

使用训练好的 Delit 模型对 VFHQ 进行推理，生成 pseudo ground truth。

### Phase 3: Relit Training (最终目标)
```
Flat-lit + Env Map → [Relit Model] → Relit Video
```

使用 FaceOLAT + VFHQ (pseudo GT) 训练 video relit 模型。

## 🔧 常见问题

### Q: 训练时 GPU 内存不足
**A**:
- 减小 `--batch_size` (例如从 4 降到 2)
- 减小 `--image_size` (例如从 512 降到 256)
- 减小 `--base_dim` (例如从 64 降到 32)

### Q: 找不到 torch 模块
**A**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q: OpenEXR 安装失败
**A**:
OpenEXR 是可选的，可以使用 `.hdr` 格式代替 `.exr`。如果需要安装：
```bash
conda install -c conda-forge openexr-python
```

### Q: 数据集找不到样本
**A**:
- 检查数据结构是否正确
- 查看终端输出的警告信息
- 确保文件扩展名正确 (.exr, .hdr, .png)

## 📚 更多信息

详细文档请查看 **README.md**

## 🎯 下一步

1. ✅ 训练 Delit 模型（当前阶段）
2. ⏭️ 对 VFHQ 进行推理，生成 pseudo GT
3. ⏭️ 训练 Relit 视频模型（需要额外实现）

---

**提示**: 如果遇到任何问题，请先查看 README.md 中的"常见问题"部分。
