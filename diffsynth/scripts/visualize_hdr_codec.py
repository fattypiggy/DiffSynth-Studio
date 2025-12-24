
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import shutil
import os

# Add project root
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from diffsynth.scripts.hdr_codec import HDRCodec
from diffsynth.scripts.delit_dataset import load_exr

try:
    import OpenEXR
    import Imath
except ImportError:
    print("OpenEXR not found.")
    raise

def save_exr(path, image):
    """Save float32 image to EXR using OpenEXR"""
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

def visualize_codec():
    # 1. Setup paths
    cwd = Path.cwd()
    possible_roots = [
        cwd / "testdata",
        cwd.parent / "testdata",
        cwd.parent.parent / "testdata"
    ]
    data_root = None
    for p in possible_roots:
        if p.exists():
            data_root = p
            break
            
    if data_root is None:
        print("Error: testdata not found")
        return

    hdri_dir = data_root / "hdri10"
    if not hdri_dir.exists():
        hdri_dir = data_root / "env_maps"
        
    if not hdri_dir.exists():
        print(f"Error: No environment maps found in {data_root}")
        return

    output_base = cwd / "output" / "codec_visualization"
    output_ldr_rgb = output_base / "ldr_rgb"
    output_ldr_log = output_base / "ldr_log"
    output_restored = output_base / "restored_hdr"
    
    for p in [output_ldr_rgb, output_ldr_log, output_restored]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
        
    print(f"I: Found {len(list(hdri_dir.glob('*.exr')))} EXR files.")
    print(f"O: Saving results to {output_base}")

    # 2. Process Files
    for exr_path in sorted(hdri_dir.glob("*.exr")):
        name = exr_path.stem
        print(f"Processing {name}...")
        
        # Load
        hdr = load_exr(str(exr_path))
        
        # Encode
        # Returns (H, W, 4)
        encoded = HDRCodec.encode_to_4channel(hdr, log_scale=10.0)
        
        # Separate Components
        norm_rgb = encoded[..., :3]     # [H, W, 3] in [0, 1]
        log_lum = encoded[..., 3]       # [H, W] in [0, 1]
        
        # Save LDR Visualizations (PNG)
        # RGB
        norm_rgb_uint8 = (norm_rgb * 255).clip(0, 255).astype(np.uint8)
        norm_rgb_bgr = cv2.cvtColor(norm_rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_ldr_rgb / f"{name}_rgb.png"), norm_rgb_bgr)
        
        # Log Lum
        log_lum_uint8 = (log_lum * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_ldr_log / f"{name}_log.png"), log_lum_uint8)
        
        # Decode/Restore
        restored = HDRCodec.decode_from_4channel(encoded, log_scale=10.0)
        
        # Save Restored HDR (EXR)
        save_exr(str(output_restored / f"{name}.exr"), restored)

    print("Done! Visualization results saved.")

if __name__ == "__main__":
    visualize_codec()
