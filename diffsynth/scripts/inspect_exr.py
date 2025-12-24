
import OpenEXR
import sys
from pathlib import Path

def inspect_exr(path):
    print(f"Inspecting: {path}")
    if not Path(path).exists():
        print("File not found")
        return
        
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    print("Channels:", header['channels'].keys())
    print("Data Window:", header['dataWindow'])
    
if __name__ == "__main__":
    path = "testdata/mask/C035.exr"
    # Find path relative to repo root if needed
    if not Path(path).exists():
        # Try two levels up (from diffsynth/scripts -> diffsynth -> root)
        path = "../../testdata/mask/C035.exr"
        
    inspect_exr(path)
