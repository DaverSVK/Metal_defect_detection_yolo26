#!/usr/bin/env python3
"""
prepare_dataset.py - Helper script to prepare NEU-DET dataset for YOLO training

This script helps convert NEU-DET dataset to YOLO format and split it into train/val/test sets.
"""

import os
import shutil
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare NEU-DET dataset for YOLO')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to original NEU-DET dataset directory')
    parser.add_argument('--output', type=str, default='datasets/NEU-DET',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation set ratio (rest goes to test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'pascal_voc', 'yolo'],
                        help='Source annotation format')
    
    return parser.parse_args()


class NEUDETConverter:
    """Convert NEU-DET dataset to YOLO format"""
    
    # NEU-DET class mapping
    CLASS_NAMES = {
        'crazing': 0,
        'inclusion': 1,
        'patches': 2,
        'pitted_surface': 3,
        'rolled-in_scale': 4,
        'scratches': 5,
    }
    
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
    def convert_pascal_voc_to_yolo(self, xml_file, img_width, img_height):
        """Convert Pascal VOC annotation to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower().replace(' ', '_')
            
            if class_name not in self.CLASS_NAMES:
                print(f"Warning: Unknown class '{class_name}' in {xml_file}")
                continue
            
            class_id = self.CLASS_NAMES[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = ((xmin + xmax) / 2) / img_width
            center_y = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Ensure values are in [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def prepare_dataset(self, train_split=0.7, val_split=0.2, seed=42):
        """Prepare dataset with train/val/test splits"""
        
        print("=" * 80)
        print("NEU-DET Dataset Preparation for YOLO")
        print("=" * 80)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(self.source_dir.glob(f'**/*{ext}')))
            all_images.extend(list(self.source_dir.glob(f'**/*{ext.upper()}')))
        
        all_images = list(set(all_images))  # Remove duplicates
        
        print(f"\nFound {len(all_images)} images")
        
        if len(all_images) == 0:
            print("Error: No images found!")
            return
        
        # Split dataset
        test_split = 1.0 - train_split - val_split
        
        train_images, temp_images = train_test_split(
            all_images, train_size=train_split, random_state=seed
        )
        
        val_size = val_split / (val_split + test_split)
        val_images, test_images = train_test_split(
            temp_images, train_size=val_size, random_state=seed
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_images)} images ({train_split*100:.1f}%)")
        print(f"  Val:   {len(val_images)} images ({val_split*100:.1f}%)")
        print(f"  Test:  {len(test_images)} images ({test_split*100:.1f}%)")
        
        # Process each split
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images,
        }
        
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, images in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            for img_path in tqdm(images, desc=f"Converting {split_name}"):
                # Copy image
                img_dest = self.output_dir / 'images' / split_name / img_path.name
                shutil.copy2(img_path, img_dest)
                
                # Look for annotation file
                xml_path = img_path.with_suffix('.xml')
                txt_path = img_path.with_suffix('.txt')
                
                label_dest = self.output_dir / 'labels' / split_name / f"{img_path.stem}.txt"
                
                if xml_path.exists():
                    # Convert Pascal VOC to YOLO
                    from PIL import Image
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    yolo_lines = self.convert_pascal_voc_to_yolo(xml_path, img_width, img_height)
                    
                    with open(label_dest, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    stats[split_name] += len(yolo_lines)
                    
                elif txt_path.exists():
                    # Already in YOLO format, just copy
                    shutil.copy2(txt_path, label_dest)
                    
                    with open(txt_path, 'r') as f:
                        stats[split_name] += len(f.readlines())
                else:
                    # No annotation, create empty file
                    label_dest.touch()
        
        print("\n" + "=" * 80)
        print("Dataset Preparation Complete!")
        print("=" * 80)
        print(f"\nAnnotation statistics:")
        print(f"  Train: {stats['train']} objects")
        print(f"  Val:   {stats['val']} objects")
        print(f"  Test:  {stats['test']} objects")
        print(f"\nOutput directory: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Verify the dataset structure")
        print("  2. Update data.yaml with the correct path")
        print("  3. Start training with: python train.py --data data.yaml")


def main():
    args = parse_args()
    
    # Check source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory not found: {args.source}")
        return
    
    # Check splits sum to 1.0
    total_split = args.train_split + args.val_split
    if total_split >= 1.0:
        print(f"Error: train_split ({args.train_split}) + val_split ({args.val_split}) must be < 1.0")
        return
    
    # Create converter
    converter = NEUDETConverter(args.source, args.output)
    
    # Prepare dataset
    converter.prepare_dataset(
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
