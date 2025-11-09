import os
import subprocess
from pathlib import Path

# ===== 修改這兩個路徑 =====
DICOM_ROOT = r"C:\Users\CPS\Desktop\BrLP-main\raw\ADNI"   # DICOM 主資料夾
OUTPUT_DIR = r"C:\Users\CPS\Desktop\BrLP-main\nii"        # NIfTI 輸出資料夾
# ==========================================

def check_dcm2niix():
    """Check if dcm2niix is installed and available."""
    try:
        subprocess.run(["dcm2niix", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def convert_dicom_folder(dicom_path, output_path):
    """Convert a single DICOM series folder to NIfTI."""
    cmd = [
        "dcm2niix",
        "-z", "y",         # compress to .nii.gz
        "-o", output_path, # output folder
        dicom_path         # input DICOM folder
    ]
    print(f"🔄 Converting: {dicom_path}")
    subprocess.run(cmd)

def batch_convert(dicom_root, output_dir):
    dicom_root = Path(dicom_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(dicom_root):
        # detect folders that contain .dcm files
        if any(f.lower().endswith(".dcm") for f in files):
            convert_dicom_folder(root, output_dir)

    print("\n✅ All DICOM folders processed!")
    print(f"✅ Output saved under: {output_dir}")

if __name__ == "__main__":
    if not check_dcm2niix():
        print("❌ dcm2niix not found! Please install it and add to PATH first.")
        print("Download: https://github.com/rordenlab/dcm2niix/releases")
    else:
        batch_convert(DICOM_ROOT, OUTPUT_DIR)
