# Quick Start Guide - YOLO Metal Defect Detection

This guide will get you up and running in minutes.

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/DaverSVK/Metal_defect_detection_yolo26.git
cd Metal_defect_detection_yolo26

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

## 📊 Dataset Setup

### Option 1: Use Your Own NEU-DET Dataset

If you already have the NEU-DET dataset:

```bash
# Prepare dataset with automatic train/val/test split
python prepare_dataset.py \
    --source /path/to/NEU-DET \
    --output datasets/NEU-DET \
    --train-split 0.7 \
    --val-split 0.2
```

### Option 2: Download NEU-DET Dataset

```bash
# Download from official source or Kaggle
# Official: http://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm
# Kaggle: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

# Then run prepare_dataset.py as shown above
```

### Option 3: Manual Setup

Create the following structure:

```
datasets/NEU-DET/
├── images/
│   ├── train/   # Training images (.jpg, .png)
│   ├── val/     # Validation images
│   └── test/    # Test images (optional)
└── labels/
    ├── train/   # YOLO format labels (.txt)
    ├── val/     # One line per object: class_id center_x center_y width height
    └── test/    # All values normalized to [0, 1]
```

## 🚀 Training

### Basic Training

```bash
# Train with default settings (YOLOv8n, 100 epochs)
python train.py --data data.yaml --model yolov8n.pt --epochs 100
```

### Recommended Training

```bash
# Better accuracy with YOLOv8s and 200 epochs
python train.py \
    --data data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --patience 50 \
    --cache
```

Training results will be saved to `runs/train/neu-det/`.

## 📈 Evaluation

```bash
# Comprehensive evaluation with all metrics
python eval_report.py \
    --weights runs/train/neu-det/weights/best.pt \
    --data data.yaml \
    --output eval_results \
    --save-plots \
    --threshold-sweep
```

Outputs:
- `per_class_metrics.png` - P/R/F1/AP charts
- `confusion_matrix.png` - Confusion matrix
- `threshold_sweep.png` - Optimal threshold analysis
- `evaluation_report.txt` - Text summary

## 🎥 Video Inference

```bash
# Process video with defect detection
python video_infer.py \
    --weights runs/train/neu-det/weights/best.pt \
    --video your_video.mp4 \
    --output output_videos \
    --conf-thres 0.25 \
    --window-size 30 \
    --display \
    --save-video
```

Features:
- Real-time OK/NOK status
- Rolling statistics window
- Visual overlay with metrics
- Press 'q' to quit, 'p' to pause

## ⚡ TensorRT Export (Optional)

Requires: NVIDIA GPU, CUDA, TensorRT

```bash
# Export to TensorRT FP16 for 2x faster inference
python export_trt.py \
    --weights runs/train/neu-det/weights/best.pt \
    --imgsz 640 \
    --half \
    --device 0
```

## 📁 Expected File Structure

After setup and training:

```
Metal_defect_detection_yolo26/
├── train.py                      # Training script
├── eval_report.py               # Evaluation script
├── video_infer.py               # Video inference
├── export_trt.py                # TensorRT export
├── prepare_dataset.py           # Dataset preparation
├── verify_setup.py              # Setup verification
├── data.yaml                    # Dataset config
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── datasets/                    # Dataset directory
│   └── NEU-DET/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── runs/                        # Training outputs
│   └── train/
│       └── neu-det/
│           ├── weights/
│           │   ├── best.pt      # Best model
│           │   └── last.pt      # Last checkpoint
│           ├── results.png
│           ├── confusion_matrix.png
│           └── ...
├── eval_results/               # Evaluation outputs
│   ├── per_class_metrics.png
│   ├── confusion_matrix.png
│   ├── threshold_sweep.png
│   └── evaluation_report.txt
└── output_videos/              # Processed videos
    └── your_video_processed.mp4
```

## 🔧 Common Issues

### Issue: CUDA out of memory
```bash
# Reduce batch size or image size
python train.py --batch 8 --imgsz 512
```

### Issue: Slow training
```bash
# Use smaller model or fewer epochs
python train.py --model yolov8n.pt --epochs 50
```

### Issue: Low accuracy
```bash
# Train longer with larger model
python train.py --model yolov8m.pt --epochs 300 --patience 100
```

## 📊 Understanding Results

### Training Metrics
- **mAP@50**: Mean Average Precision at IoU=0.50 (higher is better)
- **mAP@50-95**: Mean AP averaged over IoU thresholds (COCO metric)
- Watch for overfitting: val metrics should stay close to train metrics

### Evaluation Metrics
- **Precision**: How many detections are correct
- **Recall**: How many actual defects are detected
- **F1**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which classes are confused

### Video Inference
- **OK**: No defects detected in frame
- **NOK**: Defects detected (count ≥ threshold)
- **Rolling Stats**: Average over last N frames

## 🎯 Next Steps

1. **Tune hyperparameters**: Adjust learning rate, augmentation, etc.
2. **Try different models**: yolov8s, yolov8m, yolov8l for better accuracy
3. **Optimize threshold**: Use threshold sweep to find best confidence
4. **Deploy model**: Export to TensorRT for production use
5. **Integrate**: Use the model in your quality control system

## 📚 Learn More

- Full documentation: See README.md
- YOLO docs: https://docs.ultralytics.com/
- NEU-DET paper: Search for "NEU surface defect database"

## 💡 Tips

- **Always use `--cache`** for faster training if you have enough RAM
- **Monitor with TensorBoard**: `tensorboard --logdir runs/train`
- **Save checkpoints**: Use `--save-period 10` to save every 10 epochs
- **Early stopping**: Use `--patience 50` to stop if no improvement
- **Data augmentation**: Enabled by default, helps prevent overfitting

---

**Ready to detect defects! 🔍🏭**
