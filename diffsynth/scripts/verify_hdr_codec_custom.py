
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root (DiffSynth-Studio)
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from diffsynth.scripts.hdr_codec import HDRCodec, HDRCodecTorch

def test_hdr_codec():
    print("Verifying HDR Codec...")
    
    # 1. Basic Range Test
    print("\n[Test 1] Basic Dynamic Range (0 - 100)")
    hdr_basic = np.random.rand(64, 64, 3) * 100.0
    encoded = HDRCodec.encode_to_4channel(hdr_basic)
    decoded = HDRCodec.decode_from_4channel(encoded)
    error = np.abs(hdr_basic - decoded).max()
    print(f"  Max Error: {error:.6f}")
    assert error < 1e-4, "Basic reconstruction failed"
    
    # 2. High Dynamic Range Test
    print("\n[Test 2] High Dynamic Range (0 - 50,000)")
    # Defaults: log_scale=10. Max value encodable = exp(10) - 1 ~= 22025.
    # If we pass 50,000, it should clip if log_scale is 10.
    
    hdr_high = np.array([[[50000.0, 100.0, 1.0]]], dtype=np.float32)
    encoded_def = HDRCodec.encode_to_4channel(hdr_high, log_scale=10.0)
    decoded_def = HDRCodec.decode_from_4channel(encoded_def, log_scale=10.0)
    
    print(f"  Input Max: {hdr_high.max()}")
    print(f"  Decoded Max (Scale=10): {decoded_def.max():.2f}")
    
    if np.abs(decoded_def.max() - 50000.0) > 1.0:
        print("  -> Clipping occurred as expected with log_scale=10")
    
    # Try with higher log_scale
    print("\n[Test 3] High Dynamic Range with Adjusted Scale")
    # log(50001) ~= 10.82. So scale=12 should work.
    encoded_high = HDRCodec.encode_to_4channel(hdr_high, log_scale=12.0)
    decoded_high = HDRCodec.decode_from_4channel(encoded_high, log_scale=12.0)
    
    print(f"  Decoded Max (Scale=12): {decoded_high.max():.2f}")
    error_high = np.abs(hdr_high - decoded_high).max()
    print(f"  Max Error: {error_high:.6f}")
    assert error_high < 1.0, "High dynamic range reconstruction failed even with adjusted scale"
    
    # 3. Torch Version Consistency
    print("\n[Test 4] Torch vs Numpy Consistency")
    hdr_torch = torch.from_numpy(hdr_basic).permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    encoded_torch = HDRCodecTorch.encode_to_4channel(hdr_torch)
    decoded_torch = HDRCodecTorch.decode_from_4channel(encoded_torch)
    
    encoded_np_from_torch = encoded_torch[0].permute(1, 2, 0).numpy()
    error_cross = np.abs(encoded - encoded_np_from_torch).max()
    print(f"  Numpy vs Torch Encoding Error: {error_cross:.6f}")
    assert error_cross < 1e-5
    
    print("\n✓ HDR Codec Verification Passed!")

if __name__ == "__main__":
    test_hdr_codec()
