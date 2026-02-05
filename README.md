# Metal Defect Detection with YOLO26 on NEU-DET Dataset

A comprehensive Python project for training and evaluating Ultralytics YOLO models on the NEU-DET (Northeastern University Metal Surface Defect) dataset. This project includes advanced evaluation metrics, real-time video inference with rolling OK/NOK status, and TensorRT export for optimized deployment.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Video Inference](#video-inference)
  - [TensorRT Export](#tensorrt-export)
- [Advanced Metrics](#advanced-metrics)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

## 🎯 Features

### Core Capabilities
- **Training Script** (`train.py`): Full-featured YOLO training with extensive hyperparameter control
- **Advanced Evaluation** (`eval_report.py`): Comprehensive metrics and visualizations including:
  - Per-class Precision, Recall, F1-Score
  - mAP@0.5 and mAP@0.5:0.95
  - Confusion matrix visualization
  - Precision-Recall curves
  - Expected Calibration Error (ECE) analysis
  - False Positive/False Negative analysis
  - Threshold sweep analysis
- **Video Inference** (`video_infer.py`): Real-time defect detection with rolling OK/NOK status
- **TensorRT Export** (`export_trt.py`): FP16 TensorRT optimization for production deployment

### Dataset Support
- **NEU-DET Dataset**: 6 metal surface defect classes
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches

## 📁 Project Structure

```
Metal_defect_detection_yolo26/
│
├── train.py                 # Training script
├── eval_report.py          # Advanced evaluation and reporting
├── video_infer.py          # Video inference with rolling OK/NOK
├── export_trt.py           # TensorRT export (FP16)
├── data.yaml               # Dataset configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
│
├── datasets/              # Dataset directory (create this)
│   └── NEU-DET/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
└── runs/                  # Training and evaluation outputs
    ├── train/             # Training runs
    │   └── neu_det_yolo/
    │       ├── weights/
    │       │   ├── best.pt
    │       │   └── last.pt
    │       └── results.png
    └── eval/              # Evaluation results
        ├── evaluation_report.json
        ├── evaluation_report.md
        ├── confusion_matrix.png
        ├── pr_curves.png
        ├── calibration_curve.png
        ├── fp_fn_analysis.png
        └── threshold_sweep.png
```

## 📊 Dataset

### NEU-DET Dataset

The NEU-DET dataset is a surface defect database for metal materials. It contains 1,800 grayscale images of six kinds of typical surface defects:

1. **Crazing (Cr)**: Fine cracks on the surface
2. **Inclusion (In)**: Non-metallic inclusions
3. **Patches (Pa)**: Irregular patches
4. **Pitted Surface (PS)**: Small pits or holes
5. **Rolled-in Scale (RS)**: Oxide scale pressed into surface
6. **Scratches (Sc)**: Linear scratches

### Dataset Preparation

1. **Download the Dataset**:
   - Download NEU-DET dataset from [official source](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) or Kaggle
   - The dataset contains 300 images per class

2. **Organize the Dataset**:
   ```bash
   mkdir -p datasets/NEU-DET/images/{train,val,test}
   mkdir -p datasets/NEU-DET/labels/{train,val,test}
   ```

3. **Convert Annotations**:
   - If your dataset has annotations in a different format (e.g., XML, JSON), convert them to YOLO format
   - YOLO format: `class x_center y_center width height` (normalized 0-1)

4. **Split the Dataset**:
   - Recommended split: 70% train, 20% val, 10% test
   - Or use: 80% train, 20% val

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- TensorRT 8.0+ (optional, for TensorRT export)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/DaverSVK/Metal_defect_detection_yolo26.git
cd Metal_defect_detection_yolo26

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully')"
```

## 🚀 Quick Start

### 1. Prepare Your Dataset
Follow the [Dataset Preparation](#dataset-preparation) section above.

### 2. Train the Model
```bash
# Basic training (uses default settings)
python train.py --data data.yaml --epochs 100 --batch 16

# Advanced training with custom parameters
python train.py \
    --data data.yaml \
    --model yolov8n.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --optimizer AdamW \
    --lr0 0.001 \
    --patience 50
```

### 3. Evaluate the Model
```bash
# Run comprehensive evaluation
python eval_report.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --data data.yaml \
    --output runs/eval/neu_det_eval
```

### 4. Run Video Inference
```bash
# Process video with real-time display
python video_infer.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --source input_video.mp4 \
    --output output_video.mp4 \
    --conf 0.25 \
    --window 30

# Use webcam
python video_infer.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --source 0
```

### 5. Export to TensorRT
```bash
# Export with FP16 precision
python export_trt.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --imgsz 640 \
    --verify

# Export with FP32 precision
python export_trt.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --fp32 \
    --verify
```

## 📖 Usage Guide

### Training

The `train.py` script provides extensive control over the training process:

#### Basic Usage
```bash
python train.py --data data.yaml --epochs 100
```

#### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | Path to data.yaml | `data.yaml` |
| `--model` | Model architecture (yolov8n/s/m/l/x.pt) | `yolov8n.pt` |
| `--epochs` | Number of training epochs | `100` |
| `--batch` | Batch size | `16` |
| `--imgsz` | Image size | `640` |
| `--device` | Device (cuda:0, cpu, or auto) | `` (auto) |
| `--optimizer` | Optimizer (SGD, Adam, AdamW, auto) | `auto` |
| `--lr0` | Initial learning rate | `0.01` |
| `--patience` | Early stopping patience | `50` |
| `--cache` | Cache images for faster training | `False` |

#### Augmentation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hsv_h` | HSV-Hue augmentation | `0.015` |
| `--hsv_s` | HSV-Saturation augmentation | `0.7` |
| `--hsv_v` | HSV-Value augmentation | `0.4` |
| `--degrees` | Rotation augmentation (degrees) | `0.0` |
| `--translate` | Translation augmentation | `0.1` |
| `--scale` | Scale augmentation | `0.5` |
| `--fliplr` | Horizontal flip probability | `0.5` |
| `--mosaic` | Mosaic augmentation probability | `1.0` |

#### Example: Full Training Pipeline
```bash
# Train YOLOv8 small model with custom settings
python train.py \
    --data data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --optimizer AdamW \
    --lr0 0.001 \
    --lrf 0.01 \
    --patience 100 \
    --cache \
    --hsv_h 0.02 \
    --hsv_s 0.8 \
    --hsv_v 0.5 \
    --degrees 5.0 \
    --translate 0.2 \
    --scale 0.6 \
    --fliplr 0.5 \
    --mosaic 1.0 \
    --project runs/train \
    --name neu_det_yolov8s
```

### Evaluation

The `eval_report.py` script generates comprehensive evaluation metrics and visualizations.

#### Basic Usage
```bash
python eval_report.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --data data.yaml
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to model weights | Required |
| `--data` | Path to data.yaml | `data.yaml` |
| `--split` | Dataset split (train/val/test) | `val` |
| `--conf` | Confidence threshold | `0.25` |
| `--iou` | IoU threshold for matching | `0.5` |
| `--imgsz` | Image size | `640` |
| `--output` | Output directory | `runs/eval` |

#### Generated Outputs

1. **evaluation_report.json**: Machine-readable metrics
2. **evaluation_report.md**: Human-readable markdown report
3. **confusion_matrix.png**: Confusion matrix heatmap
4. **pr_curves.png**: Precision-Recall curves per class
5. **calibration_curve.png**: Model calibration analysis
6. **fp_fn_analysis.png**: False Positive/Negative breakdown
7. **threshold_sweep.png**: Metrics vs confidence threshold

### Video Inference

The `video_infer.py` script processes videos with real-time defect detection and rolling OK/NOK status.

#### Basic Usage
```bash
# Process video file
python video_infer.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --source input_video.mp4 \
    --output output_video.mp4

# Use webcam (camera index 0)
python video_infer.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --source 0
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to model weights | Required |
| `--source` | Video file or camera index | Required |
| `--output` | Output video path | `None` |
| `--conf` | Confidence threshold | `0.25` |
| `--window` | Rolling window size (frames) | `30` |
| `--no-display` | Disable real-time display | `False` |
| `--no-stats` | Disable saving statistics | `False` |

#### Interactive Controls

- Press `q` to quit
- Press `p` to pause/resume

#### Rolling OK/NOK Logic

- **OK**: Less than 20% of recent frames contain defects
- **NOK**: More than 20% of recent frames contain defects
- Window size determines how many frames are considered (default: 30)

### TensorRT Export

The `export_trt.py` script exports models to optimized TensorRT format.

#### Basic Usage
```bash
# Export with FP16 (recommended)
python export_trt.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --verify

# Export with FP32
python export_trt.py \
    --weights runs/train/neu_det_yolo/weights/best.pt \
    --fp32 \
    --verify
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to model weights | Required |
| `--output` | Output engine path | Auto-generated |
| `--imgsz` | Input image size | `640` |
| `--fp32` | Use FP32 instead of FP16 | `False` |
| `--dynamic` | Enable dynamic shapes | `False` |
| `--workspace` | Max workspace size (GB) | `4` |
| `--device` | CUDA device | `0` |
| `--verify` | Verify after export | `False` |
| `--test-image` | Test image for verification | `None` |

#### Using Exported Model

```python
from ultralytics import YOLO

# Load TensorRT model
model = YOLO('best.engine')

# Run inference
results = model('image.jpg')
```

## 📈 Advanced Metrics

### Evaluation Metrics Explained

#### 1. Per-Class Precision, Recall, F1-Score
- **Precision**: What proportion of positive predictions are correct?
- **Recall**: What proportion of actual positives are detected?
- **F1-Score**: Harmonic mean of precision and recall

#### 2. mAP (mean Average Precision)
- **mAP@0.5**: Mean AP at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean AP averaged over IoU thresholds 0.5 to 0.95

#### 3. Confusion Matrix
- Shows the relationship between predicted and actual classes
- Diagonal elements represent correct predictions
- Off-diagonal elements show misclassifications

#### 4. Precision-Recall Curves
- Visualizes the trade-off between precision and recall
- Area under curve (AUC) represents Average Precision
- Separate curve for each defect class

#### 5. Expected Calibration Error (ECE)
- Measures how well predicted confidences match actual accuracy
- Lower ECE indicates better calibration
- Important for decision-making based on confidence scores

#### 6. False Positive/False Negative Analysis
- **False Positives**: Detected defects that don't exist
- **False Negatives**: Missed defects that do exist
- Helps identify model weaknesses per class

#### 7. Threshold Sweep
- Shows how metrics change with different confidence thresholds
- Helps select optimal threshold for deployment
- Visualizes precision/recall trade-off

## ⚙️ Configuration

### data.yaml Structure

```yaml
# Dataset root directory
path: ./datasets/NEU-DET

# Dataset splits (relative to 'path')
train: images/train
val: images/val
test: images/test  # optional

# Classes
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled_in_scale
  5: scratches

# Number of classes
nc: 6
```

### Customization Tips

1. **Adjust for Your Dataset**:
   - Modify class names in `data.yaml`
   - Update paths to match your directory structure

2. **Hyperparameter Tuning**:
   - Start with default settings
   - Increase epochs if loss is still decreasing
   - Adjust learning rate if training is unstable
   - Tune augmentation based on your data characteristics

3. **Model Selection**:
   - YOLOv8n: Fastest, smallest (3.2M params)
   - YOLOv8s: Balanced (11.2M params)
   - YOLOv8m: Higher accuracy (25.9M params)
   - YOLOv8l: Very high accuracy (43.7M params)
   - YOLOv8x: Maximum accuracy (68.2M params)

## 📊 Results

After training and evaluation, you can expect outputs like:

### Training Results
- Best and last model weights
- Training curves (loss, mAP, precision, recall)
- Validation results

### Evaluation Results
- Comprehensive metrics report (JSON and Markdown)
- Confusion matrix showing class relationships
- PR curves for each defect type
- Calibration analysis
- FP/FN breakdown
- Threshold sweep analysis

### Video Inference
- Annotated video with bounding boxes
- Real-time OK/NOK status indicator
- Statistics JSON file with detection summary

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- NEU-DET Dataset: Northeastern University
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- NVIDIA TensorRT: https://developer.nvidia.com/tensorrt

## 📞 Support

For questions and issues:
- Create an issue on GitHub
- Check the Ultralytics documentation: https://docs.ultralytics.com/

---

**Note**: Make sure you have proper permissions to use the NEU-DET dataset. This project is for educational and research purposes.