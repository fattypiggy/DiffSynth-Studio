
import torch
import sys
from pathlib import Path
import os
import shutil
import cv2
import numpy as np

# Add project root (DiffSynth-Studio)
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from diffsynth.scripts.delit_dataset import FaceOLATDelitDataset, perspective_to_equirectangular_torch
from diffsynth.scripts.delit_model_lora import DelitWanLoRA

# Mocks
class MockVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.model = torch.nn.Module()
        self.model.encode = self.encode
        
    def encode(self, x, scale):
        # x: [B, 3, 1, H, W] or [B, 3, H, W]
        # In our code we unsqueeze T.
        # Output latent: [B, 16, 1, H/8, W/8]
        if x.dim() == 5:
            h, w = x.shape[3], x.shape[4]
        else:
            h, w = x.shape[2], x.shape[3]
        return torch.randn(x.shape[0], 16, 1, h//8, w//8)

class MockBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Simulate attention components that match target modules "q,k,v,o"
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)
        # Simulate gate and head if needed, though q,k,v,o is usually enough to pass
        self.gate = torch.nn.Linear(dim, dim)
        self.head = torch.nn.Linear(dim, dim)

class MockDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 36
        dim = 16
        # Use a list of blocks
        self.blocks = torch.nn.ModuleList([
            MockBlock(dim) for _ in range(2)
        ])
        
    def forward(self, x, t, context):
        # x: [B, 36, 1, h, w]
        # Return same shape as "noise prediction" usually? 
        # Wan uses v-prediction or noise. Output dim 16 (matches latent).
        return torch.randn(x.shape[0], 16, 1, x.shape[3], x.shape[4])

try:
    import OpenEXR
    import Imath
except ImportError:
    print("OpenEXR not found. Please install it in the 'relit' environment.")
    raise

def save_exr(path, image):
    """Save float32 image to EXR using OpenEXR"""
    # image: [H, W, 3]
    H, W, C = image.shape
    header = OpenEXR.Header(W, H)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}
    
    exr = OpenEXR.OutputFile(str(path), header)
    
    # Planar RGB
    r = image[:, :, 0].tobytes()
    g = image[:, :, 1].tobytes()
    b = image[:, :, 2].tobytes()
    
    exr.writePixels({'R': r, 'G': g, 'B': b})
    exr.close()

def create_dummy_data():
    root = Path("/tmp/delit_verify_data")
    if root.exists():
        shutil.rmtree(root)
    
    (root / "subjects" / "ID001" / "relit").mkdir(parents=True)
    (root / "env_maps").mkdir(parents=True)
    
    # Create dummy images
    # Relit Image (Perspective)
    relit_img = np.random.rand(512, 512, 3).astype(np.float32)
    save_exr(str(root / "subjects" / "ID001" / "relit" / "env_001.exr"), relit_img)
    
    # Mask (PNG is fine with cv2)
    cv2.imwrite(str(root / "subjects" / "ID001" / "mask.png"), (np.random.rand(512, 512) * 255).astype(np.uint8))
    
    # Env Map
    env_img = np.random.rand(512, 1024, 3).astype(np.float32)
    save_exr(str(root / "env_maps" / "env_001.exr"), env_img)
    
    return str(root)

def test_pipeline():
    print("Setting up verification...")
    data_root = create_dummy_data()
    
    # 1. Test Dataset
    print("\n[Test 1] Dataset")
    dataset = FaceOLATDelitDataset(
        data_root=data_root,
        env_resolution=(512, 1024),
        image_size=512,
        use_mask=True
    )
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Relit Equir Shape: {sample['relit_equir'].shape} (Expected [4, 512, 1024])")
    print(f"Person Mask Shape: {sample['person_mask_equir'].shape} (Expected [1, 512, 1024])")
    print(f"Env Encoded Shape: {sample['env_encoded'].shape} (Expected [4, 512, 1024])")
    
    assert sample['relit_equir'].shape == (4, 512, 1024)
    assert sample['person_mask_equir'].shape == (1, 512, 1024)
    assert sample['env_encoded'].shape == (4, 512, 1024)
    
    # 2. Test Model Logic
    print("\n[Test 2] Model Logic")
    vae = MockVAE()
    dit = MockDiT()
    model = DelitWanLoRA(vae, dit, lora_rank=4)
    
    # Prepare batch
    relit_equir = sample['relit_equir'].unsqueeze(0) # [1, 4, 512, 1024]
    person_mask = sample['person_mask_equir'].unsqueeze(0)
    valid_mask = sample['valid_mask'].unsqueeze(0)
    
    # Mock noisy latent (target size)
    # VAE downsample 8x -> 512/8=64, 1024/8=128
    noisy_latents = torch.randn(1, 16, 1, 64, 128)
    timestep = torch.tensor([500.0]) # Float for scheduler
    
    print("Running forward pass...")
    output = model(
        relit_equir=relit_equir,
        person_mask_equir=person_mask,
        valid_mask=valid_mask,
        noisy_latents=noisy_latents,
        timestep=timestep
    )
    
    print(f"Output shape: {output.shape} (Expected [1, 16, 1, 64, 128])")
    assert output.shape == (1, 16, 1, 64, 128)
    
    print("\nVerification Passed!")

if __name__ == "__main__":
    test_pipeline()
