# Metal Defect Detection with YOLO

A comprehensive YOLO-based solution for detecting surface defects on hot-rolled steel strips using the NEU-DET dataset. This project includes training, advanced evaluation, video inference with rolling OK/NOK detection, and TensorRT optimization.

## 🎯 Features

- **Training**: Full YOLO training pipeline with customizable hyperparameters
- **Advanced Evaluation**: Comprehensive metrics including:
  - Per-class Precision/Recall/F1 scores
  - mAP@50 and mAP@50-95
  - Confusion matrix visualization
  - Precision-Recall curves
  - ECE calibration analysis
  - False Positive/False Negative analysis
  - Confidence threshold sweep optimization
- **Video Inference**: Real-time defect detection with rolling OK/NOK statistics
- **TensorRT Export**: FP16 optimized model for high-performance inference

## 📋 NEU-DET Dataset

The NEU-DET dataset contains 6 classes of surface defects on hot-rolled steel strips:

1. **Crazing** (Cr) - Fine cracks on the surface
2. **Inclusion** (In) - Non-metallic inclusions
3. **Patches** (Pa) - Irregular patches
4. **Pitted Surface** (PS) - Small pits or holes
5. **Rolled-in Scale** (RS) - Oxide scale rolled into surface
6. **Scratches** (Sc) - Linear scratches

## 📁 Project Structure

```
Metal_defect_detection_yolo26/
├── train.py              # Training script
├── eval_report.py        # Comprehensive evaluation with advanced metrics
├── video_infer.py        # Video inference with rolling OK/NOK detection
├── export_trt.py         # TensorRT FP16 export script
├── data.yaml             # Dataset configuration
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── datasets/            # Dataset directory (you need to prepare this)
│   └── NEU-DET/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── runs/                # Training outputs (auto-generated)
│   └── train/
│       └── neu-det/
├── eval_results/        # Evaluation outputs (auto-generated)
└── output_videos/       # Processed videos (auto-generated)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/DaverSVK/Metal_defect_detection_yolo26.git
cd Metal_defect_detection_yolo26

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download and prepare the NEU-DET dataset:

```bash
# Create dataset directory
mkdir -p datasets/NEU-DET

# Download NEU-DET dataset
# Source: http://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm
# Or from: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

# Convert to YOLO format
# The dataset should have the following structure:
# datasets/NEU-DET/
#   images/
#     train/  (image files)
#     val/    (image files)
#     test/   (image files, optional)
#   labels/
#     train/  (YOLO format .txt files)
#     val/    (YOLO format .txt files)
#     test/   (YOLO format .txt files, optional)
```

**YOLO Label Format**: Each `.txt` file should contain one line per object:
```
class_id center_x center_y width height
```
All values are normalized to [0, 1].

### 3. Training

Train YOLO on NEU-DET dataset:

```bash
# Basic training with YOLOv8 nano
python train.py --data data.yaml --model yolov8n.pt --epochs 100

# Training with more options
python train.py \
    --data data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --patience 50 \
    --optimizer AdamW \
    --lr0 0.01 \
    --augment \
    --cache

# Available model sizes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

**Key Training Arguments**:
- `--model`: YOLO model variant (n/s/m/l/x)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Input image size
- `--device`: GPU device (0, 1, ...) or 'cpu'
- `--patience`: Early stopping patience
- `--cache`: Cache images in RAM for faster training

Training outputs will be saved to `runs/train/neu-det/`.

### 4. Evaluation

Generate comprehensive evaluation report:

```bash
# Evaluate trained model
python eval_report.py \
    --weights runs/train/neu-det/weights/best.pt \
    --data data.yaml \
    --split val \
    --output eval_results \
    --save-plots \
    --threshold-sweep

# With custom thresholds
python eval_report.py \
    --weights runs/train/neu-det/weights/best.pt \
    --data data.yaml \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --output eval_results \
    --min-conf 0.01 \
    --max-conf 0.95 \
    --conf-steps 50
```

**Evaluation Outputs**:
- `per_class_metrics.png`: Bar charts of P/R/F1/AP per class
- `confusion_matrix.png`: Confusion matrix (raw & normalized)
- `threshold_sweep.png`: Metrics vs confidence threshold
- `evaluation_report.txt`: Text summary of all metrics
- `evaluation_results.json`: JSON format results (with `--save-json`)

### 5. Video Inference

Process videos with real-time defect detection:

```bash
# Process video with rolling statistics
python video_infer.py \
    --weights runs/train/neu-det/weights/best.pt \
    --video test_video.mp4 \
    --output output_videos \
    --conf-thres 0.25 \
    --window-size 30 \
    --defect-threshold 1 \
    --display \
    --save-video

# Batch processing without display
python video_infer.py \
    --weights runs/train/neu-det/weights/best.pt \
    --video test_video.mp4 \
    --output output_videos \
    --save-video \
    --no-display
```

**Video Inference Features**:
- Real-time OK/NOK status
- Rolling window statistics (last N frames)
- Defect detection with bounding boxes
- Visual overlay with metrics
- Keyboard controls: 'q' to quit, 'p' to pause

### 6. TensorRT Export

Export model to TensorRT for optimized inference:

```bash
# Export to TensorRT FP16
python export_trt.py \
    --weights runs/train/neu-det/weights/best.pt \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --half

# Export with dynamic batching
python export_trt.py \
    --weights runs/train/neu-det/weights/best.pt \
    --imgsz 640 \
    --dynamic \
    --workspace 4 \
    --half

# Export with INT8 quantization (requires calibration)
python export_trt.py \
    --weights runs/train/neu-det/weights/best.pt \
    --imgsz 640 \
    --int8
```

**TensorRT Requirements**:
- NVIDIA GPU with CUDA
- TensorRT installed: `pip install tensorrt`
- Compatible CUDA version

## 📊 Evaluation Metrics Explained

### Per-Class Metrics
- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **Recall**: TP / (TP + FN) - Coverage of actual positives
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AP@50**: Average Precision at IoU=0.50
- **AP@50-95**: Average Precision averaged over IoU=0.50:0.05:0.95

### Overall Metrics
- **mAP@50**: Mean Average Precision at IoU=0.50 (across all classes)
- **mAP@50-95**: Mean Average Precision at IoU=0.50:0.95 (COCO metric)

### Confusion Matrix
- Shows predicted vs actual classes
- Diagonal = correct predictions
- Off-diagonal = misclassifications

### Threshold Sweep
- Analyzes performance across different confidence thresholds
- Helps find optimal threshold for your use case
- Trade-off between precision and recall

### Calibration (ECE)
- Expected Calibration Error
- Measures reliability of confidence scores
- Lower is better (well-calibrated model)

## 🔧 Configuration

Edit `data.yaml` to customize dataset paths and classes:

```yaml
path: ./datasets/NEU-DET
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

## 💡 Tips & Best Practices

### Training
- Start with a pretrained model (e.g., `yolov8n.pt`)
- Use `--cache` to speed up training (requires sufficient RAM)
- Monitor training on TensorBoard: `tensorboard --logdir runs/train`
- Use data augmentation for better generalization
- Adjust learning rate based on batch size

### Evaluation
- Run threshold sweep to find optimal confidence threshold
- Analyze confusion matrix to identify problematic classes
- Use per-class metrics to focus on weak classes
- Check calibration to ensure reliable confidence scores

### Video Inference
- Adjust `--window-size` based on your application
- Use `--defect-threshold` to control OK/NOK sensitivity
- Process videos in batches for efficiency
- Save processed videos for quality control

### TensorRT Optimization
- Use FP16 for 2x speedup with minimal accuracy loss
- Use INT8 for 4x speedup (requires calibration dataset)
- Match input size between training and export
- Benchmark on your target hardware

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or image size

```bash
python train.py --batch 8 --imgsz 512
```

### Issue: TensorRT export fails
**Solutions**:
1. Check CUDA and TensorRT versions are compatible
2. Try without `--dynamic` flag
3. Reduce `--workspace` value
4. Export to ONNX first, then convert manually

### Issue: Low mAP scores
**Solutions**:
1. Train for more epochs
2. Use larger model (yolov8s/m instead of yolov8n)
3. Increase image size
4. Check dataset quality and labels
5. Tune hyperparameters (lr, augmentation)

### Issue: Video inference is slow
**Solutions**:
1. Use TensorRT exported model
2. Reduce `--imgsz`
3. Use smaller YOLO model
4. Use GPU instead of CPU
5. Disable `--display` for faster processing

## 📚 References

- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **NEU-DET Dataset**: http://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm
- **TensorRT**: https://developer.nvidia.com/tensorrt

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Happy Defect Detection! 🔍🏭**