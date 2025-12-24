"""
生成小型测试数据集
用于快速验证训练流程，无需真实的 FaceOLAT 数据
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def create_synthetic_face(size=512):
    """创建合成的人脸图像（简单的几何形状）"""
    img = np.zeros((size, size, 3), dtype=np.float32)

    # 脸部（椭圆）
    center = (size // 2, size // 2)
    axes = (size // 3, size // 2)
    color = (0.8, 0.7, 0.6)  # 肤色
    cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)

    # 眼睛
    eye_y = size // 3
    eye_radius = size // 20
    cv2.circle(img, (size // 3, eye_y), eye_radius, (0.1, 0.1, 0.1), -1)
    cv2.circle(img, (2 * size // 3, eye_y), eye_radius, (0.1, 0.1, 0.1), -1)

    # 嘴巴
    mouth_center = (size // 2, 2 * size // 3)
    mouth_axes = (size // 8, size // 16)
    cv2.ellipse(img, mouth_center, mouth_axes, 0, 0, 180, (0.5, 0.2, 0.2), -1)

    # 鼻子
    nose_pts = np.array([
        [size // 2, size // 2 - 20],
        [size // 2 - 10, size // 2 + 10],
        [size // 2 + 10, size // 2 + 10]
    ], np.int32)
    cv2.fillPoly(img, [nose_pts], (0.7, 0.6, 0.5))

    return img


def create_mask_from_face(face_img, threshold=0.01):
    """从人脸图像创建 mask"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    return mask


def create_environment_map(env_id, resolution=(128, 256)):
    """创建合成的环境图（HDR）"""
    h, w = resolution

    # 创建基础渐变
    env_map = np.zeros((h, w, 3), dtype=np.float32)

    # 类型1: 天空（上方亮，下方暗）
    if env_id % 3 == 0:
        for i in range(h):
            brightness = ((h - i) / h) ** 2 * 50
            env_map[i, :] = brightness * np.array([0.8, 0.9, 1.0])

        # 添加太阳
        sun_x = np.random.randint(w // 4, 3 * w // 4)
        sun_y = np.random.randint(0, h // 3)
        sun_brightness = np.random.uniform(500, 1500)

        y, x = np.ogrid[:h, :w]
        sun_mask = ((x - sun_x)**2 + (y - sun_y)**2) <= (h // 10)**2
        env_map[sun_mask] = sun_brightness

    # 类型2: 侧光（左右不对称）
    elif env_id % 3 == 1:
        for j in range(w):
            brightness = ((j / w) ** 2) * 30
            env_map[:, j] = brightness * np.array([1.0, 0.9, 0.7])

        # 添加窗户效果
        window_x = 3 * w // 4
        window_brightness = 200
        env_map[:, window_x - 20:window_x + 20] = window_brightness

    # 类型3: 多点光源
    else:
        # 背景
        env_map[:] = 2.0

        # 添加多个光源
        num_lights = np.random.randint(2, 5)
        for _ in range(num_lights):
            light_x = np.random.randint(0, w)
            light_y = np.random.randint(0, h)
            light_brightness = np.random.uniform(100, 800)
            light_radius = np.random.randint(10, 30)

            y, x = np.ogrid[:h, :w]
            light_mask = ((x - light_x)**2 + (y - light_y)**2) <= light_radius**2
            env_map[light_mask] = light_brightness * np.array([1.0, 0.95, 0.8])

    return env_map


def apply_lighting(face_img, env_map, mask):
    """
    模拟光照效果
    这是一个简化的光照模型，不使用真实的 OLAT

    Args:
        face_img: 基础人脸图像
        env_map: 环境图
        mask: 人脸 mask

    Returns:
        relit_img: 光照后的图像
    """
    # 计算平均光照强度
    avg_light = env_map.mean(axis=(0, 1))

    # 归一化到合理范围
    avg_light = np.clip(avg_light / 100.0, 0.1, 10.0)

    # 应用光照
    relit_img = face_img * avg_light[np.newaxis, np.newaxis, :]

    # 添加一些随机的阴影效果
    h, w = face_img.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)

    # 根据环境图的主要光源方向添加阴影
    light_center_x = np.argmax(env_map.mean(axis=0).mean(axis=1))
    light_direction = (light_center_x / env_map.shape[1]) * 2 - 1  # -1 到 1

    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                # 简单的方向性阴影
                offset = int((j - w // 2) * light_direction * 0.3)
                if 0 <= j + offset < w:
                    distance = abs(offset) / w
                    shadow[i, j] = 1.0 - distance * 0.5

    # 应用阴影
    relit_img = relit_img * shadow[:, :, np.newaxis]

    # 添加高光
    highlight = np.random.rand(h, w) < 0.01  # 1% 的像素
    relit_img[highlight] *= 2.0

    return relit_img


def save_hdr_as_exr_fallback(filepath, img):
    """
    保存 HDR 图像为 EXR（如果 OpenEXR 不可用，保存为 .hdr）
    """
    try:
        import OpenEXR
        import Imath

        height, width = img.shape[:2]
        header = OpenEXR.Header(width, height)

        img = img.astype(np.float32)

        channels = {
            'R': img[:, :, 0].tobytes(),
            'G': img[:, :, 1].tobytes(),
            'B': img[:, :, 2].tobytes()
        }

        exr_file = OpenEXR.OutputFile(str(filepath), header)
        exr_file.writePixels(channels)
        exr_file.close()
    except ImportError:
        # Fallback to .hdr format
        print(f"  OpenEXR not available, saving as .hdr instead")
        filepath = Path(str(filepath).replace('.exr', '.hdr'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), img_bgr)


def generate_toy_dataset(
    output_dir: str,
    num_subjects: int = 3,
    num_envs: int = 10,
    image_size: int = 512
):
    """
    生成小型测试数据集

    Args:
        output_dir: 输出目录
        num_subjects: 生成的人物数量
        num_envs: 每个人物的环境图数量
        image_size: 图像分辨率
    """
    output_dir = Path(output_dir)
    subjects_dir = output_dir / "subjects"
    env_maps_dir = output_dir / "env_maps"

    subjects_dir.mkdir(parents=True, exist_ok=True)
    env_maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating toy dataset:")
    print(f"  Output: {output_dir}")
    print(f"  Subjects: {num_subjects}")
    print(f"  Environments per subject: {num_envs}")
    print(f"  Image size: {image_size}")
    print()

    # 生成环境图
    print("Generating environment maps...")
    env_maps = []
    for env_id in tqdm(range(num_envs), desc="Env maps"):
        env_map = create_environment_map(env_id)
        env_maps.append(env_map)

        # 保存环境图
        env_name = f"env_{env_id:03d}"

        # 保存 HDR
        save_hdr_as_exr_fallback(
            env_maps_dir / f"{env_name}.exr",
            env_map
        )

        # 保存预览（tone mapped）
        env_preview = np.clip(env_map ** (1/2.2), 0, 1)
        env_preview = (env_preview * 255).astype(np.uint8)
        env_preview_bgr = cv2.cvtColor(env_preview, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(env_maps_dir / f"{env_name}_preview.png"), env_preview_bgr)

    # 生成人物数据
    print("\nGenerating subjects...")
    for subject_id in tqdm(range(num_subjects), desc="Subjects"):
        subject_name = f"ID{subject_id:03d}"
        subject_dir = subjects_dir / subject_name
        relit_dir = subject_dir / "relit"

        subject_dir.mkdir(exist_ok=True)
        relit_dir.mkdir(exist_ok=True)

        # 创建基础人脸（添加一些变化）
        np.random.seed(subject_id)
        base_face = create_synthetic_face(image_size)

        # 添加一些个性化变化
        noise = np.random.randn(*base_face.shape) * 0.05
        base_face = np.clip(base_face + noise, 0, 1)

        # 创建 mask
        mask = create_mask_from_face(base_face)

        # 保存 mask
        cv2.imwrite(str(subject_dir / "mask.png"), mask)

        # 创建 flat-lit（均匀光照）
        flat_lit = base_face.copy()

        # 保存 flat-lit
        # 为了简化，我们用 PNG 保存（实际应该是 EXR）
        flat_lit_8bit = (np.clip(flat_lit, 0, 1) * 255).astype(np.uint8)
        flat_lit_bgr = cv2.cvtColor(flat_lit_8bit, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(subject_dir / "flat_lit.png"), flat_lit_bgr)

        # 生成不同光照下的 relit 图像
        for env_id in range(num_envs):
            env_name = f"env_{env_id:03d}"
            env_map = env_maps[env_id]

            # 应用光照
            relit_img = apply_lighting(base_face, env_map, mask)

            # 保存 relit 图像
            save_hdr_as_exr_fallback(
                relit_dir / f"{env_name}.exr",
                relit_img
            )

    # 创建数据集统计信息
    stats = {
        'num_subjects': num_subjects,
        'num_envs': num_envs,
        'image_size': image_size,
        'total_samples': num_subjects * num_envs,
    }

    # 保存统计信息
    import json
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Dataset generated successfully!")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Location: {output_dir}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── subjects/")
    print(f"  │   ├── ID000/")
    print(f"  │   │   ├── flat_lit.png")
    print(f"  │   │   ├── mask.png")
    print(f"  │   │   └── relit/")
    print(f"  │   │       ├── env_000.exr")
    print(f"  │   │       └── ...")
    print(f"  │   └── ...")
    print(f"  └── env_maps/")
    print(f"      ├── env_000.exr")
    print(f"      ├── env_000_preview.png")
    print(f"      └── ...")


def main():
    parser = argparse.ArgumentParser(description='Generate toy dataset for quick testing')

    parser.add_argument('--output_dir', type=str, default='./toy_dataset',
                        help='Output directory')
    parser.add_argument('--num_subjects', type=int, default=3,
                        help='Number of subjects to generate')
    parser.add_argument('--num_envs', type=int, default=10,
                        help='Number of environment maps')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image resolution')

    args = parser.parse_args()

    generate_toy_dataset(
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        num_envs=args.num_envs,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main()
