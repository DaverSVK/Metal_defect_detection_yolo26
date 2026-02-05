"""
train.py - Train YOLOv8 on NEU-DET dataset for metal surface defect detection
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO on NEU-DET dataset')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='neu-det',
                        help='Experiment name')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, 1, cpu, etc.)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (-1 to disable)')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images for faster training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['SGD', 'Adam', 'AdamW', 'auto'],
                        help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate factor')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='Warmup epochs')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Mixup augmentation probability')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation probability')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    print("=" * 80)
    print("YOLO Training on NEU-DET Dataset - Metal Surface Defect Detection")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.workers}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Initial LR: {args.lr0}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Mosaic: {args.mosaic}")
    print(f"  Mixup: {args.mixup}")
    print("=" * 80 + "\n")
    
    # Load model
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        resume=args.resume,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        augment=args.augment,
        mixup=args.mixup,
        mosaic=args.mosaic,
        verbose=True,
        # Additional training parameters
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last N epochs
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Dataset fraction to use
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_json=False,
        save_hybrid=False,
    )
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"\nBest model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    print(f"\nResults saved to: {model.trainer.save_dir}")
    
    # Print final metrics
    if results:
        print("\nFinal Metrics:")
        try:
            print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        except:
            print("  (Metrics will be available in results files)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
