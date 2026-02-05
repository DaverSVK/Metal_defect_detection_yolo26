#!/usr/bin/env python3
"""
verify_setup.py - Verify that all project files are in place
"""

import os
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {file_path}")
    return exists


def main():
    print("=" * 80)
    print("YOLO26 Metal Defect Detection Project - Setup Verification")
    print("=" * 80)
    
    all_ok = True
    
    # Core files
    print("\n1. Core Python Scripts:")
    all_ok &= check_file_exists("train.py", "Training script")
    all_ok &= check_file_exists("eval_report.py", "Evaluation script")
    all_ok &= check_file_exists("video_infer.py", "Video inference script")
    all_ok &= check_file_exists("export_trt.py", "TensorRT export script")
    all_ok &= check_file_exists("prepare_dataset.py", "Dataset preparation script")
    
    # Configuration files
    print("\n2. Configuration Files:")
    all_ok &= check_file_exists("data.yaml", "Dataset configuration")
    all_ok &= check_file_exists("requirements.txt", "Python dependencies")
    all_ok &= check_file_exists(".gitignore", "Git ignore rules")
    
    # Documentation
    print("\n3. Documentation:")
    all_ok &= check_file_exists("README.md", "Project README")
    
    # Check Python syntax
    print("\n4. Python Syntax Check:")
    scripts = ["train.py", "eval_report.py", "video_infer.py", "export_trt.py", "prepare_dataset.py"]
    
    for script in scripts:
        try:
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            print(f"  ✓ {script} - Syntax OK")
        except SyntaxError as e:
            print(f"  ✗ {script} - Syntax Error: {e}")
            all_ok = False
    
    # Check requirements.txt content
    print("\n5. Dependencies Check:")
    required_packages = [
        'ultralytics', 'torch', 'numpy', 'pandas', 'matplotlib',
        'opencv-python', 'scikit-learn', 'seaborn'
    ]
    
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read().lower()
        
        for package in required_packages:
            if package in requirements:
                print(f"  ✓ {package}")
            else:
                print(f"  ✗ {package} - Missing!")
                all_ok = False
    else:
        print("  ✗ requirements.txt not found!")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ Setup verification PASSED - All files present and valid!")
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Prepare dataset: python prepare_dataset.py --source <path> --output datasets/NEU-DET")
        print("  3. Start training: python train.py --data data.yaml --model yolov8n.pt")
    else:
        print("✗ Setup verification FAILED - Some files are missing or invalid!")
        print("\nPlease check the errors above and ensure all required files are present.")
    print("=" * 80)


if __name__ == '__main__':
    main()
