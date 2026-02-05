# Quick Start Guide

## Installation (3 steps)

```bash
# 1. Clone repository
git clone https://github.com/DaverSVK/Metal_defect_detection_yolo26.git
cd Metal_defect_detection_yolo26

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset (organize NEU-DET into required structure)
mkdir -p datasets/NEU-DET/images/{train,val,test}
mkdir -p datasets/NEU-DET/labels/{train,val,test}
```

## Basic Usage (4 commands)

```bash
# 1. Train model (100 epochs, batch size 16)
python train.py --data data.yaml --epochs 100 --batch 16

# 2. Evaluate model (generates 7 visualizations + reports)
python eval_report.py --weights runs/train/neu_det_yolo/weights/best.pt --data data.yaml

# 3. Process video (real-time defect detection with OK/NOK status)
python video_infer.py --weights runs/train/neu_det_yolo/weights/best.pt --source video.mp4 --output result.mp4

# 4. Export to TensorRT (2-3x faster inference)
python export_trt.py --weights runs/train/neu_det_yolo/weights/best.pt --verify
```

## Dataset Structure

```
datasets/NEU-DET/
├── images/
│   ├── train/        # Training images
│   ├── val/          # Validation images
│   └── test/         # Test images (optional)
└── labels/
    ├── train/        # Training labels (YOLO format)
    ├── val/          # Validation labels
    └── test/         # Test labels (optional)
```

## YOLO Label Format

Each `.txt` file contains one line per object:
```
class x_center y_center width height
```
All values normalized to 0-1 range.

Example:
```
0 0.5 0.5 0.3 0.4
2 0.2 0.3 0.15 0.2
```

## NEU-DET Classes

0. crazing
1. inclusion
2. patches
3. pitted_surface
4. rolled_in_scale
5. scratches

## Evaluation Outputs

After running `eval_report.py`, you'll get:

1. **evaluation_report.json** - Machine-readable metrics
2. **evaluation_report.md** - Human-readable report
3. **confusion_matrix.png** - Class confusion visualization
4. **pr_curves.png** - Precision-Recall curves
5. **calibration_curve.png** - Model calibration
6. **fp_fn_analysis.png** - False Positive/Negative breakdown
7. **threshold_sweep.png** - Metrics vs confidence threshold

## Common Options

### Training
```bash
# Use larger model for better accuracy
python train.py --model yolov8s.pt --epochs 200 --batch 32

# Enable image caching for faster training
python train.py --cache --workers 8

# Custom learning rate
python train.py --lr0 0.001 --optimizer AdamW
```

### Evaluation
```bash
# Change confidence threshold
python eval_report.py --weights best.pt --conf 0.3

# Evaluate on test set
python eval_report.py --weights best.pt --split test

# Change IoU threshold
python eval_report.py --weights best.pt --iou 0.6
```

### Video Inference
```bash
# Use webcam
python video_infer.py --weights best.pt --source 0

# Adjust rolling window size
python video_infer.py --weights best.pt --source video.mp4 --window 60

# Process without display (batch mode)
python video_infer.py --weights best.pt --source video.mp4 --no-display --output result.mp4
```

### TensorRT Export
```bash
# Use FP32 precision (more accurate, slower)
python export_trt.py --weights best.pt --fp32

# Enable dynamic shapes (variable image sizes)
python export_trt.py --weights best.pt --dynamic

# Adjust workspace for limited GPU memory
python export_trt.py --weights best.pt --workspace 2
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch 8

# Reduce image size
python train.py --imgsz 416
```

### CUDA Not Available
```bash
# Use CPU (slower)
python train.py --device cpu
```

### TensorRT Not Found
```bash
# TensorRT is optional, use ONNX instead
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='onnx')"
```

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Cache images**: Use `--cache` for faster training
3. **Larger batch**: Increase `--batch` if GPU memory allows
4. **Use TensorRT**: 2-3x faster inference
5. **Optimize image size**: Balance between speed and accuracy

## Next Steps

1. Train model on your dataset
2. Evaluate and analyze results
3. Tune hyperparameters based on metrics
4. Export to TensorRT for deployment
5. Integrate video inference into your pipeline

## Support

- Read full README.md for detailed documentation
- Check FILE_TREE.md for project structure
- See IMPLEMENTATION_SUMMARY.md for features overview

## Example Workflow

```bash
# Complete workflow from training to deployment
# 1. Train
python train.py --data data.yaml --model yolov8s.pt --epochs 150 --batch 16 --cache

# 2. Evaluate
python eval_report.py --weights runs/train/neu_det_yolo/weights/best.pt --data data.yaml --output runs/eval/

# 3. Test on video
python video_infer.py --weights runs/train/neu_det_yolo/weights/best.pt --source test_video.mp4 --output result.mp4

# 4. Export for production
python export_trt.py --weights runs/train/neu_det_yolo/weights/best.pt --verify

# 5. Use exported model
python video_infer.py --weights runs/train/neu_det_yolo/weights/best.engine --source 0
```

---

**Ready to start?** Run the installation commands above and follow the basic usage!
