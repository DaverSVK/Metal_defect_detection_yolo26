"""
YOLO26 Training Script for NEU-DET Metal Defect Detection
This script trains an Ultralytics YOLO model on the NEU-DET dataset.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO26 on NEU-DET Metal Defect Dataset'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to data.yaml configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model to train (yolov8n/s/m/l/x.pt or custom weights)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (cuda:0, cpu, or empty for auto)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for data loading'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='neu_det_yolo',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate factor'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='EarlyStopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='Save checkpoint every x epochs (disabled if -1)'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images for faster training'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Apply data augmentation'
    )
    parser.add_argument(
        '--hsv_h',
        type=float,
        default=0.015,
        help='HSV-Hue augmentation'
    )
    parser.add_argument(
        '--hsv_s',
        type=float,
        default=0.7,
        help='HSV-Saturation augmentation'
    )
    parser.add_argument(
        '--hsv_v',
        type=float,
        default=0.4,
        help='HSV-Value augmentation'
    )
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='Rotation augmentation (degrees)'
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='Translation augmentation'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale augmentation'
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='Horizontal flip probability'
    )
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='Vertical flip probability'
    )
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print system information
    print("=" * 60)
    print("YOLO26 Training - NEU-DET Metal Defect Detection")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    # Verify data config exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config file not found: {args.data}")
    
    # Training hyperparameters
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'patience': args.patience,
        'save_period': args.save_period,
        'cache': args.cache,
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': args.resume,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        # Augmentation parameters
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # Print training configuration
    print("\nTraining Configuration:")
    print("-" * 60)
    for key, value in train_args.items():
        if value is not None and value != '':
            print(f"  {key}: {value}")
    print("-" * 60)
    
    # Start training
    print("\nStarting training...")
    results = model.train(**train_args)
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Results saved to: {model.trainer.save_dir}")
    print(f"Best weights: {model.trainer.best}")
    print(f"Last weights: {model.trainer.last}")
    
    # Print final metrics
    if hasattr(model.trainer, 'metrics'):
        print("\nFinal Metrics:")
        print("-" * 60)
        metrics = model.trainer.metrics
        if hasattr(metrics, 'results_dict'):
            for key, value in metrics.results_dict.items():
                print(f"  {key}: {value:.4f}")
    
    print("\nTo evaluate the model, run:")
    print(f"  python eval_report.py --weights {model.trainer.best} --data {args.data}")
    print("=" * 60)


if __name__ == '__main__':
    main()
