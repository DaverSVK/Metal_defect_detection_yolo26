#!/usr/bin/env python3
"""
show_tree.py - Display project file structure
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", max_depth=3, current_depth=0, exclude_dirs=None):
    """Print directory tree structure"""
    
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', 'venv', 'env', '.idea', '.vscode', 
                       'runs', 'datasets', 'eval_results', 'output_videos', 'weights'}
    
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return
    
    dirs = [e for e in entries if e.is_dir() and e.name not in exclude_dirs]
    files = [e for e in entries if e.is_file() and not e.name.startswith('.')]
    
    # Print files first
    for i, file in enumerate(files):
        is_last = (i == len(files) - 1) and len(dirs) == 0
        connector = "└── " if is_last else "├── "
        
        # Add file info
        size = file.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size/(1024*1024):.1f}MB"
        
        print(f"{prefix}{connector}{file.name} ({size_str})")
    
    # Then print directories
    for i, dir_path in enumerate(dirs):
        is_last = i == len(dirs) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        print(f"{prefix}{connector}{dir_path.name}/")
        print_tree(dir_path, prefix + extension, max_depth, current_depth + 1, exclude_dirs)


def main():
    print("=" * 80)
    print("YOLO26 Metal Defect Detection - Project Structure")
    print("=" * 80)
    print("\nMetal_defect_detection_yolo26/")
    
    print_tree(".", max_depth=4)
    
    print("\n" + "=" * 80)
    print("Notes:")
    print("  - Excluded: .git, __pycache__, venv, runs, datasets, eval_results")
    print("  - These directories will be created during training/evaluation")
    print("=" * 80)


if __name__ == '__main__':
    main()
