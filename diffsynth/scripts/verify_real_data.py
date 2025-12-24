
import sys
from pathlib import Path
import torch
import shutil

# Add project root (DiffSynth-Studio)
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from diffsynth.scripts.delit_dataset import FaceOLATDelitDataset

def verify_real_data():
    print("Verifying Real Data Loading...")
    
    # Path to testdata
    # Assumes run from repo root or scripts dir
    # Try finding testdata
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
        print(f"Error: 'testdata' directory not found. Checked: {possible_roots}")
        return

    print(f"Data Root: {data_root}")
    
    # Initialize Dataset
    dataset = FaceOLATDelitDataset(
        data_root=str(data_root),
        env_resolution=(512, 1024),
        image_size=512,
        use_mask=True
    )
    
    print(f"Found {len(dataset)} samples.")
    if len(dataset) == 0:
        print("Error: No samples found.")
        return
        
    # Get first sample
    sample = dataset[0]
    print("\nSample 0:")
    print(f"  Subject: {sample['subject_id']}")
    print(f"  Env Name: {sample['env_name']}")
    print(f"  Relit Equir Shape: {sample['relit_equir'].shape}")
    print(f"  Person Mask Shape: {sample['person_mask_equir'].shape}")
    print(f"  Valid Mask Shape: {sample['valid_mask'].shape}")
    print(f"  Env Encoded Shape: {sample['env_encoded'].shape}")
    
    # Check values range
    print(f"  Relit Max: {sample['relit_equir'].max():.4f}")
    print(f"  Mask Mean: {sample['person_mask_equir'].mean():.4f}")
    
    # Check Env Map dynamic range (approximate from encoded)
    # Channel 3 is log_luminance
    log_lum = sample['env_encoded'][3, :, :]
    max_log = log_lum.max()
    estimated_hdr_max = torch.expm1(max_log * 10.0) # assuming default log_scale=10
    print(f"  Env Log Max: {max_log:.4f}")
    print(f"  Est. Env HDR Max: {estimated_hdr_max:.4f}")
    
    if estimated_hdr_max > 22000:
        print("  WARNING: Env Map intensity nears limit of log_scale=10!")
    
    print("\n✓ Real Data Verification Passed!")

if __name__ == "__main__":
    verify_real_data()
