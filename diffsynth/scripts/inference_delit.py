"""
使用训练好的 Delit 模型进行推理
用于对 VFHQ 或其他视频/图像进行 delit
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

from delit_model import SimplifiedDelitModel
from hdr_codec import HDRCodec, HDRCodecTorch


def load_model(checkpoint_path: str, device: torch.device, **model_kwargs):
    """加载训练好的模型"""
    model = SimplifiedDelitModel(**model_kwargs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理 DDP 保存的模型
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}")

    return model


def normalize_image(img: np.ndarray) -> np.ndarray:
    """将图像归一化到 [-1, 1]"""
    # 假设输入是 [0, 1] 范围
    img = img * 2.0 - 1.0
    return img


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """将图像从 [-1, 1] 反归一化到 [0, 1]"""
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)


def save_hdr_image(filepath: str, img: np.ndarray):
    """保存 HDR 图像"""
    ext = Path(filepath).suffix.lower()

    if ext == '.hdr':
        # Radiance HDR format
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
    elif ext == '.exr':
        # OpenEXR format
        try:
            import OpenEXR
            import Imath

            height, width = img.shape[:2]
            header = OpenEXR.Header(width, height)

            # 转换为 float32
            img = img.astype(np.float32)

            # 写入通道
            channels = {'R': img[:, :, 0].tobytes(),
                       'G': img[:, :, 1].tobytes(),
                       'B': img[:, :, 2].tobytes()}

            exr_file = OpenEXR.OutputFile(filepath, header)
            exr_file.writePixels(channels)
            exr_file.close()
        except ImportError:
            print("Warning: OpenEXR not available, saving as .hdr instead")
            filepath = str(Path(filepath).with_suffix('.hdr'))
            save_hdr_image(filepath, img)
    else:
        raise ValueError(f"Unsupported HDR format: {ext}")


@torch.no_grad()
def inference_single_image(
    model: SimplifiedDelitModel,
    image_path: str,
    output_dir: Path,
    device: torch.device,
    image_size: int = 512
):
    """对单张图像进行推理"""
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Resize
    original_size = img.shape[:2]
    img = cv2.resize(img, (image_size, image_size))

    # 归一化
    img_normalized = normalize_image(img)

    # 转为 tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 推理
    flat_lit, env_encoded = model(img_tensor)

    # 转回 numpy
    flat_lit = flat_lit.squeeze(0).permute(1, 2, 0).cpu().numpy()
    env_encoded = env_encoded.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 反归一化
    flat_lit = denormalize_image(flat_lit)

    # 解码环境图
    env_hdr = HDRCodec.decode_from_4channel(env_encoded, model.log_scale)

    # Resize 回原始尺寸（flat-lit）
    flat_lit = cv2.resize(flat_lit, (original_size[1], original_size[0]))

    # 保存结果
    image_name = Path(image_path).stem

    # 保存 flat-lit（LDR）
    flat_lit_ldr = (flat_lit * 255).astype(np.uint8)
    flat_lit_bgr = cv2.cvtColor(flat_lit_ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{image_name}_flat_lit.png"), flat_lit_bgr)

    # 保存环境图（HDR）
    save_hdr_image(str(output_dir / f"{image_name}_env.hdr"), env_hdr)

    # 也保存一个 LDR 预览
    env_ldr = np.clip(env_hdr ** (1/2.2), 0, 1)  # 简单的 tone mapping
    env_ldr = (env_ldr * 255).astype(np.uint8)
    env_ldr_bgr = cv2.cvtColor(env_ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{image_name}_env_preview.png"), env_ldr_bgr)

    print(f"Saved results for {image_name}")


@torch.no_grad()
def inference_video(
    model: SimplifiedDelitModel,
    video_path: str,
    output_dir: Path,
    device: torch.device,
    image_size: int = 512
):
    """对视频进行推理"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    # 创建输出视频
    video_name = Path(video_path).stem

    # Flat-lit 视频
    flat_lit_path = output_dir / f"{video_name}_flat_lit.mp4"
    flat_lit_writer = cv2.VideoWriter(
        str(flat_lit_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # 保存所有帧的环境图
    env_maps = []

    # 逐帧处理
    pbar = tqdm(total=total_frames, desc="Processing video")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        # Resize
        frame_resized = cv2.resize(frame_rgb, (image_size, image_size))

        # 归一化
        frame_normalized = normalize_image(frame_resized)

        # 转为 tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # 推理
        flat_lit, env_encoded = model(frame_tensor)

        # 转回 numpy
        flat_lit = flat_lit.squeeze(0).permute(1, 2, 0).cpu().numpy()
        env_encoded_np = env_encoded.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 反归一化
        flat_lit = denormalize_image(flat_lit)

        # 解码环境图
        env_hdr = HDRCodec.decode_from_4channel(env_encoded_np, model.log_scale)
        env_maps.append(env_hdr)

        # Resize flat-lit 回原始尺寸
        flat_lit = cv2.resize(flat_lit, (width, height))

        # 写入视频
        flat_lit_bgr = cv2.cvtColor((flat_lit * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        flat_lit_writer.write(flat_lit_bgr)

        # 保存每隔 N 帧的环境图预览
        if frame_idx % 30 == 0:  # 每秒保存一次（假设30fps）
            env_ldr = np.clip(env_hdr ** (1/2.2), 0, 1)
            env_ldr = (env_ldr * 255).astype(np.uint8)
            env_ldr_bgr = cv2.cvtColor(env_ldr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{video_name}_env_frame_{frame_idx:05d}.png"), env_ldr_bgr)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    flat_lit_writer.release()
    pbar.close()

    # 保存平均环境图
    avg_env = np.mean(env_maps, axis=0)
    save_hdr_image(str(output_dir / f"{video_name}_env_avg.hdr"), avg_env)

    print(f"Saved video results to {output_dir}")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = load_model(
        args.checkpoint,
        device,
        base_dim=args.base_dim,
        env_resolution=(args.env_height, args.env_width),
        log_scale=args.log_scale
    )

    # 推理
    input_path = Path(args.input)

    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 视频
            inference_video(model, str(input_path), output_dir, device, args.image_size)
        else:
            # 图像
            inference_single_image(model, str(input_path), output_dir, device, args.image_size)
    elif input_path.is_dir():
        # 目录中的所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))

        print(f"Found {len(image_files)} images")

        for image_file in tqdm(image_files, desc="Processing images"):
            inference_single_image(model, str(image_file), output_dir, device, args.image_size)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    print("Inference completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delit Model Inference')

    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像/视频/目录路径')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='输出目录')

    # 模型参数（需要与训练时一致）
    parser.add_argument('--base_dim', type=int, default=64, help='模型基础维度')
    parser.add_argument('--env_height', type=int, default=128, help='环境图高度')
    parser.add_argument('--env_width', type=int, default=256, help='环境图宽度')
    parser.add_argument('--log_scale', type=float, default=10.0, help='HDR log 缩放因子')
    parser.add_argument('--image_size', type=int, default=512, help='处理图像分辨率')

    args = parser.parse_args()

    main(args)
