"""
FaceOLAT Delit Dataset - 用于训练 delit 模型的数据集
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

try:
    import OpenEXR
    import Imath
    EXR_AVAILABLE = True
except ImportError:
    EXR_AVAILABLE = False
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")

from .hdr_codec import HDRCodec


def load_exr(filepath: str) -> np.ndarray:
    """
    加载 EXR 文件为线性 HDR 图像

    Args:
        filepath: EXR 文件路径

    Returns:
        img: [H, W, 3] 线性 HDR 图像，float32
    """
    if not EXR_AVAILABLE:
        raise ImportError("OpenEXR is required to load EXR files")

    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()

    # 获取图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Determine available channels
    header_channels = header['channels'].keys()
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    if 'R' in header_channels and 'G' in header_channels and 'B' in header_channels:
        channels_to_read = ['R', 'G', 'B']
        channel_data = [exr_file.channel(c, FLOAT) for c in channels_to_read]
        img = np.zeros((height, width, 3), dtype=np.float32)
        for i, data in enumerate(channel_data):
            img[:, :, i] = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    elif 'Y' in header_channels:
        # Grayscale EXR
        y_data = exr_file.channel('Y', FLOAT)
        y_img = np.frombuffer(y_data, dtype=np.float32).reshape(height, width)
        img = np.stack([y_img, y_img, y_img], axis=2) # Duplicate to RGB
    else:
        # Fallback usually leads to errors but return black or try to read whatever is there?
        raise ValueError(f"EXR file contains neither RGB nor Y channels: {filepath}, Found: {list(header_channels)}")

    return img


def load_hdr(filepath: str) -> np.ndarray:
    """
    加载 HDR 文件（支持 .hdr, .exr）

    Returns:
        img: [H, W, 3] 线性 HDR 图像
    """
    ext = Path(filepath).suffix.lower()

    if ext == '.exr':
        return load_exr(filepath)
    elif ext in ['.hdr', '.pic']:
        # OpenCV 可以读取 Radiance HDR
        img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load HDR file: {filepath}")
        # OpenCV 读取为 BGR，转为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported HDR format: {ext}")


def load_ldr_image(filepath: str, to_linear: bool = True) -> np.ndarray:
    """
    加载 LDR 图像（PNG, JPG 等）

    Args:
        filepath: 图像路径
        to_linear: 是否转换到线性空间

    Returns:
        img: [H, W, 3] 图像，范围 [0, 1]
    """
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {filepath}")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # sRGB to linear
    if to_linear:
        img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

    return img



def perspective_to_equirectangular_torch(
    img_tensor: torch.Tensor,
    mask_tensor: Optional[torch.Tensor] = None,
    output_size: Tuple[int, int] = (512, 1024),
    fov: float = 90.0,
    yaw: float = 0.0,
    pitch: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects a perspective image onto an equirectangular canvas.
    This creates a sparse equirectangular image where valid pixels are filled
    from the perspective image, and invalid pixels are zero.

    Args:
        img_tensor: [C, H, W] Perspective image
        mask_tensor: [1, H, W] Validity mask of the perspective image (optional)
        output_size: (Height, Width) of the output equirectangular map
        fov: Field of View in degrees
        yaw: Horizontal rotation in radians
        pitch: Vertical rotation in radians

    Returns:
        equir_img: [C, out_H, out_W] Projected image
        valid_mask: [1, out_H, out_W] Boolean mask indicating valid pixels
    """
    out_h, out_w = output_size
    device = img_tensor.device
    
    # 1. Create Equirectangular View Direction Vectors
    # Current pixel coordinates
    theta = torch.linspace(-np.pi, np.pi, out_w, device=device).unsqueeze(0).repeat(out_h, 1) # Longitude
    phi = torch.linspace(np.pi/2, -np.pi/2, out_h, device=device).unsqueeze(1).repeat(1, out_w) # Latitude
    
    # Spherical to Cartesian (Z is forward in our env map convention? 
    # Usually in graphics: 
    # X=right, Y=up, Z=backward (OpenGL) or Z=forward 
    # In `render_reference_envmap_relit.py`:
    # x = sin(phi)cos(theta), y = sin(phi)sin(theta), z = cos(phi) 
    # Let's align with standard latlong to cartesian:
    # x = cos(lat) * sin(lon)
    # y = sin(lat)
    # z = cos(lat) * cos(lon)
    # But usually env maps map X to center. Let's assume standard equirectangular.
    
    # Using the reverse of `equirect_to_perspective` logic from `render_reference_envmap_relit.py`
    # There: longitude = atan2(x, z). So z is forward (0), x is right (90 deg).
    # lat = asin(y) or similar.
    
    x_env = torch.cos(phi) * torch.sin(theta)
    y_env = torch.sin(phi)
    z_env = torch.cos(phi) * torch.cos(theta)
    
    # 2. Applying Inverse Rotation (Camera Rotation) to align with Camera Frame
    # To project World (Env) to Camera, we usually apply Rot^T (or inverse Rot).
    # Camera looking at (yaw, pitch).
    # Here we rotate the *rays* from World frame to Camera frame.
    
    # Yaw (rotation around Y axis)
    # In `render_reference_envmap_relit.py`: 
    # x_rot = x * cos_yaw + z * sin_yaw
    # This rotates vector by -yaw if we consider standard rotation. 
    # Let's just use the inverse logic carefully.
    
    # Inverse Yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # Rotating frame *back* by yaw means rotating vector by -yaw?
    # Actually, if camera is rotated by R, Ray_cam = R^T * Ray_world.
    # R_yaw(alpha) = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    # We want to align World Z with Camera Z.
    
    x_rot = x_env * cos_yaw - z_env * sin_yaw
    z_rot = x_env * sin_yaw + z_env * cos_yaw
    y_rot = y_env 
    
    # Inverse Pitch (rotation around X axis)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    
    # In `render_reference_envmap_relit.py`: y_rot = y * cos - z * sin
    # We invert this.
    y_cam = y_rot * cos_pitch + z_rot * sin_pitch
    z_cam = -y_rot * sin_pitch + z_rot * cos_pitch
    x_cam = x_rot
    
    # 3. Project to Perspective Plane
    # Camera looks down +Z axis.
    # Perspective projection: u = x/z * f, v = y/z * f
    # Valid pixels satisfy z > 0 (in front of camera)
    
    fov_rad = fov * np.pi / 180.0
    focal = 0.5 / np.tan(fov_rad / 2.0) # Assuming normalized image coord [-0.5, 0.5] or [-1, 1]?
    # Let's map to [-1, 1] range for grid_sample
    
    # Avoid division by zero and check for "behind camera"
    # z_cam > 0 check
    valid_z = z_cam > 1e-3
    
    # Normalized coordinates [-1, 1]
    # In Pytorch grid_sample: -1 is left/top, 1 is right/bottom.
    # x goes left->right, y goes top->bottom?
    # Standard: x right, y up in 3D. Image: y down.
    # So v = -y_cam / z_cam * focal * 2 (factor 2 because focal is for half-width 1 if calc above?)
    
    # Let's be consistent with `equirect_to_perspective`
    # focal = out_width / (2 * tan(fov/2)) -> pixel units
    # Here we work in normalized units.
    # u = x / z * (1 / tan(fov/2)) = x / z * cot(fov/2)
    
    cot_fov_2 = 1.0 / np.tan(fov_rad / 2.0)
    
    u_cam = (x_cam / (z_cam + 1e-8)) * cot_fov_2
    v_cam = (y_cam / (z_cam + 1e-8)) * cot_fov_2 
    
    # In image space y is down, but in 3D y is up. So invert v.
    v_cam = -v_cam
    
    # Check validity
    in_view = (u_cam >= -1.0) & (u_cam <= 1.0) & (v_cam >= -1.0) & (v_cam <= 1.0) & valid_z
    
    # 4. Sample from Perspective Image
    # We construct a grid for grid_sample
    # Shape: [1, Out_H, Out_W, 2]
    grid = torch.stack([u_cam, v_cam], dim=-1).unsqueeze(0)
    
    # Because we are "pulling" Perspective pixels onto the Equirectangular canvas,
    # we can just use grid_sample!
    # "Where does this Equirect pixel map to in the Perspective image?"
    # If it maps inside, we sample. If outside, we get padding (zeros).
    
    # Prepare input
    batch_img = img_tensor.unsqueeze(0) # [1, C, H, W]
    
    # Sample
    # We use zeroes for padding to represent invalid areas naturally
    equir_img = torch.nn.functional.grid_sample(
        batch_img, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False
    )
    
    # Create mask
    # If mask provided, sample it. Else create one from 'in_view' logic.
    if mask_tensor is not None:
        batch_mask = mask_tensor.unsqueeze(0)
        equir_mask = torch.nn.functional.grid_sample(
            batch_mask,
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
    else:
        # Create a full white mask to track sampling validity
        ones = torch.ones_like(batch_img[:, :1, :, :])
        equir_mask = torch.nn.functional.grid_sample(
            ones,
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        
    # Combine validity check (mathematical field of view) with sampling result
    # Although padding_mode='zeros' handles it, explicit mask is cleaner for binary validity
    valid_mask_tensor = in_view.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
    equir_mask = equir_mask * valid_mask_tensor
    
    return equir_img.squeeze(0), equir_mask.squeeze(0)


class FaceOLATDelitDataset(Dataset):
    """
    Dataset for FaceOLAT Delit Training (Wan2.2 LoRA version)
    
    Input:
        - Relit Image (Perspective, 512x512)
        - Person Mask (Perspective, 512x512)
    Target:
        - Full Environment Map (Equirectangular, 1024x512)
        
    Processing:
        - Projects Input Perspective Image -> Sparse Equirectangular Canvas
    """

    def __init__(
        self,
        data_root: str,
        env_resolution: Tuple[int, int] = (512, 1024), # H, W
        image_size: int = 512,
        log_scale: float = 10.0,
        use_mask: bool = True,
        fov: float = 90.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.env_resolution = env_resolution
        self.image_size = image_size
        self.log_scale = log_scale
        self.use_mask = use_mask
        self.fov = fov
        
        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} samples")

    def _collect_samples(self) -> List[dict]:
        samples = []
        
        # Strategy 1: Standard Structure
        subjects_dir = self.data_root / "subjects"
        if subjects_dir.exists():
            return self._collect_samples_standard(subjects_dir)
            
        # Strategy 2: User Custom Structure (testdata)
        # Look for folders ending with _relit
        relit_roots = list(self.data_root.glob("*_relit"))
        if not relit_roots:
            print(f"No subjects or *_relit folders found in {self.data_root}")
            return []
            
        # Check for env maps in 'hdri10' or 'env_maps'
        env_maps_dir = self.data_root / "hdri10"
        if not env_maps_dir.exists():
            env_maps_dir = self.data_root / "env_maps"
            
        if not env_maps_dir.exists():
            print(f"No environment maps (hdri10 or env_maps) found in {self.data_root}")
            return []
            
        print(f"Detected User Data Structure. Found {len(relit_roots)} subject folders.")
        
        for relit_root in relit_roots:
            # Subject name: C035_relit -> C035
            subject_name = relit_root.name.replace("_relit", "")
            
            # Mask
            mask_path = self.data_root / "mask" / f"{subject_name}.exr"
            if not mask_path.exists():
                 mask_path = self.data_root / "mask" / f"{subject_name}.png"
            
            # Iterate env folders inside C035_relit
            # Structure: C035_relit/env_name/000.exr
            for env_folder in sorted(relit_root.iterdir()):
                if not env_folder.is_dir(): continue
                
                env_name = env_folder.name
                relit_path = env_folder / "000.exr"
                
                if not relit_path.exists():
                    continue
                    
                # Find matching env map
                env_map_path = env_maps_dir / f"{env_name}.exr"
                if not env_map_path.exists():
                     env_map_path = env_maps_dir / f"{env_name}.hdr"
                
                if env_map_path.exists():
                     samples.append({
                        'relit_path': relit_path,
                        'mask_path': mask_path,
                        'env_map_path': env_map_path,
                        'subject_id': subject_name,
                        'env_name': env_name
                    })
                    
        return samples

    def _collect_samples_standard(self, subjects_dir: Path) -> List[dict]:
        samples = []
        env_maps_dir = self.data_root / "env_maps"
        
        if not env_maps_dir.exists():
            return []

        for subject_dir in sorted(subjects_dir.glob("*")):
            if not subject_dir.is_dir(): continue

            mask_path = subject_dir / "mask.png"
            # If not exists, will handle in __getitem__ (zeros) or skip here?
            # Let's be lenient.

            relit_dir = subject_dir / "relit"
            if not relit_dir.exists(): continue

            for relit_file in sorted(relit_dir.glob("*.exr")):
                env_name = relit_file.stem 
                if relit_file.parent.name != 'relit':
                    env_name = relit_file.parent.name
                
                env_map_path = None
                for ext in ['.hdr', '.exr']:
                    candidate = env_maps_dir / f"{env_name}{ext}"
                    if candidate.exists():
                        env_map_path = candidate
                        break
                
                if env_map_path:
                    samples.append({
                        'relit_path': relit_file,
                        'mask_path': mask_path,
                        'env_map_path': env_map_path,
                        'subject_id': subject_dir.name,
                        'env_name': env_name
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # 1. Load Inputs
        # Relit Image (Perspective)
        relit_persp = load_exr(str(sample['relit_path'])) # [H, W, 3]
        
        # Mask (Perspective)
        if sample['mask_path'].exists():
            if sample['mask_path'].suffix.lower() == '.exr':
                mask_img = load_exr(str(sample['mask_path'])) # [H, W, 3]
                mask_persp = mask_img[:, :, 0] # Assume consistency
            else:
                mask_persp = cv2.imread(str(sample['mask_path']), cv2.IMREAD_GRAYSCALE)
                mask_persp = mask_persp.astype(np.float32) / 255.0
        else:
            mask_persp = np.zeros(relit_persp.shape[:2], dtype=np.float32)

        # Env Map (Target Equirectangular)
        env_map = load_hdr(str(sample['env_map_path'])) # [H_env, W_env, 3]
        
        # 2. Resize Inputs
        relit_persp = cv2.resize(relit_persp, (self.image_size, self.image_size))
        mask_persp = cv2.resize(mask_persp, (self.image_size, self.image_size))
        
        # Resize Target Env Map
        target_h, target_w = self.env_resolution
        env_map = cv2.resize(env_map, (target_w, target_h)) # cv2 uses (W, H)
        
        # Clamp Extreme Values (Physical + Codec Constraint)
        # 20000.0 corresponds to log(20001) / 10 ~= 0.99, safe for log_scale=10
        # This prevents "blown out" relighting and codec clipping.
        env_map = np.clip(env_map, 0, 20000.0)
        
        # 3. Codec Encoding (Log Scale)
        # Encode Target
        env_encoded = HDRCodec.encode_to_4channel(env_map, self.log_scale) # [H, W, 4]
        # Encode Input Relit (It is also HDR)
        relit_encoded = HDRCodec.encode_to_4channel(relit_persp, self.log_scale) # [H, W, 4]
        
        # 4. To Tensor
        relit_tensor = torch.from_numpy(relit_encoded).permute(2, 0, 1).float() # [4, 512, 512]
        mask_tensor = torch.from_numpy(mask_persp).unsqueeze(0).float() # [1, 512, 512]
        env_tensor = torch.from_numpy(env_encoded).permute(2, 0, 1).float() # [4, 1024, 2048]
        
        # 5. Projection: Perspective -> Equirectangular
        # Assume camera looks at center of env map (yaw=0, pitch=0) for standard dataset
        # Or random rotation if augmentation is on? 
        # For now fixed 0.
        
        # Project Relit Image (Sparse)
        # Note: Input to projection is [4, H, W] (encoded HDR)
        relit_equir, valid_mask = perspective_to_equirectangular_torch(
            relit_tensor,
            mask_tensor=None, # We want full validity of the square frame
            output_size=self.env_resolution,
            fov=self.fov
        )
        
        # Project Person Mask separately (Sparse)
        # This mask tells us where the person is on the equirect map
        person_mask_equir, _ = perspective_to_equirectangular_torch(
            mask_tensor,
            mask_tensor=None,
            output_size=self.env_resolution,
            fov=self.fov
        )
        
        # Combined Mask for Input
        # We want the model to know: 
        # 1. Where we have valid pixels from the perspective projection (valid_mask)
        # 2. Where the person is (person_mask_equir)
        
        # Valid Mask: 1 inside POV, 0 outside (black)
        # Person Mask: 1 on person, 0 on background (inside POV)
        
        return {
            'relit_equir': relit_equir,             # [4, H, W] Sparse Input
            'person_mask_equir': person_mask_equir, # [1, H, W] Person location
            'valid_mask': valid_mask,               # [1, H, W] POV location
            'env_encoded': env_tensor,              # [4, H, W] Target (Dense)
            'subject_id': sample['subject_id'],
            'env_name': sample['env_name']
        }

def test_dataset():
    # Simple shape test with dummy data
    print("Testing dataset projection...")
    img = torch.randn(4, 512, 512)
    mask = torch.ones(1, 512, 512)
    
    equir, valid = perspective_to_equirectangular_torch(
        img, mask, output_size=(512, 1024), fov=90.0
    )
    
    print(f"Input: {img.shape}")
    print(f"Output Equir: {equir.shape}")
    print(f"Valid Mask: {valid.shape}")
    print("Passed.")

if __name__ == "__main__":
    test_dataset()

