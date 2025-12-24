
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# Add project root
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from diffsynth.scripts.hdr_codec import HDRCodec
from diffsynth.scripts.delit_dataset import load_exr

def analyze_codec():
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

    report_path = cwd / "output" / "hdr_codec_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(hdri_dir.glob("*.exr")))
    print(f"Found {len(files)} EXR files.")
    
    results = []
    
    for exr_path in files:
        name = exr_path.stem
        
        # Load
        hdr = load_exr(str(exr_path))
        
        # Apply Clamping (Match Dataset Logic)
        hdr = np.clip(hdr, 0, 20000.0)
        
        # Analyze Input
        hdr_max = hdr.max()
        hdr_min = hdr.min()
        
        # Encode
        # Returns (H, W, 4)
        encoded = HDRCodec.encode_to_4channel(hdr, log_scale=10.0)
        
        norm_rgb = encoded[..., :3]
        log_lum = encoded[..., 3]
        
        # Analyze Encoded Range
        rgb_min, rgb_max = norm_rgb.min(), norm_rgb.max()
        log_min, log_max = log_lum.min(), log_lum.max()
        
        # Decode/Restore
        restored = HDRCodec.decode_from_4channel(encoded, log_scale=10.0)
        
        # Compute Error
        abs_diff = np.abs(hdr - restored)
        mae = abs_diff.mean()
        max_error = abs_diff.max()
        
        results.append({
            "Name": name,
            "HDR Max": hdr_max,
            "RGB Range": f"[{rgb_min:.4f}, {rgb_max:.4f}]",
            "Log Range": f"[{log_min:.4f}, {log_max:.4f}]",
            "RGB Valid": (rgb_min >= 0 and rgb_max <= 1.0 + 1e-6),
            "Log Valid": (log_min >= 0 and log_max <= 1.0 + 1e-6),
            "MAE": mae,
            "Max Error": max_error
        })
        
    # Generate Markdown Report
    with open(report_path, "w") as f:
        f.write("# HDR Codec Analysis Report\n\n")
        f.write(f"**Date**: {pd.Timestamp.now()}\n")
        f.write(f"**Log Scale**: 10.0\n")
        f.write(f"**Data Path**: {hdri_dir}\n\n")
        
        f.write("## Summary\n")
        df = pd.DataFrame(results)
        
        all_rgb_valid = df["RGB Valid"].all()
        all_log_valid = df["Log Valid"].all()
        max_mae = df["MAE"].max()
        worst_max_error = df["Max Error"].max()
        
        f.write(f"- **All Encoded RGB in [0, 1]**: {'YES' if all_rgb_valid else 'NO'}\n")
        f.write(f"- **All Encoded Log in [0, 1]**: {'YES' if all_log_valid else 'NO'}\n")
        f.write(f"- **Worst Reconstruction MAE**: {max_mae:.6f}\n")
        f.write(f"- **Worst Absolute Max Error**: {worst_max_error:.6f}\n\n")
        
        f.write("## Detailed Statistics\n\n")
        f.write("| Name | HDR Max | RGB Range | Log Range | MAE | Max Error |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for r in results:
            f.write(f"| {r['Name']} | {r['HDR Max']:.2f} | {r['RGB Range']} | {r['Log Range']} | {r['MAE']:.6f} | {r['Max Error']:.6f} |\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    analyze_codec()
