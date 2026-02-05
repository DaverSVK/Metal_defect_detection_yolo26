# Project File Tree and Overview

## Complete File Structure

```
Metal_defect_detection_yolo26/
│
├── .gitignore                 # Git ignore rules (458 bytes)
├── README.md                  # Comprehensive documentation (524 lines, 15KB)
├── requirements.txt           # Python dependencies (22 lines, 388 bytes)
├── data.yaml                  # Dataset configuration (21 lines, 642 bytes)
│
├── train.py                   # Training script (296 lines, 7.5KB)
├── eval_report.py            # Advanced evaluation (811 lines, 31KB)
├── video_infer.py            # Video inference with OK/NOK (444 lines, 15KB)
└── export_trt.py             # TensorRT export (300 lines, 8.6KB)
```

## File Descriptions

### Core Scripts

#### 1. train.py (296 lines)
- **Purpose**: Train YOLO model on NEU-DET dataset
- **Features**:
  - Extensive hyperparameter control
  - Support for all YOLO architectures (n/s/m/l/x)
  - Configurable augmentation
  - Auto-resume capability
  - Mixed precision training
  - Multiple optimizer support (SGD, Adam, AdamW)
- **Key Arguments**:
  - `--data`: Dataset configuration file
  - `--model`: Model architecture
  - `--epochs`: Training epochs
  - `--batch`: Batch size
  - `--optimizer`: Optimizer choice
  - And 30+ more parameters for fine-tuning

#### 2. eval_report.py (811 lines)
- **Purpose**: Comprehensive model evaluation and reporting
- **Features**:
  - Per-class metrics (Precision, Recall, F1-Score)
  - mAP@0.5 and mAP@0.5:0.95
  - Confusion matrix visualization
  - Precision-Recall curves
  - Expected Calibration Error (ECE)
  - False Positive/False Negative analysis
  - Threshold sweep analysis
- **Outputs**:
  - evaluation_report.json (machine-readable)
  - evaluation_report.md (human-readable)
  - confusion_matrix.png
  - pr_curves.png
  - calibration_curve.png
  - fp_fn_analysis.png
  - threshold_sweep.png

#### 3. video_infer.py (444 lines)
- **Purpose**: Real-time video inference with rolling OK/NOK status
- **Features**:
  - Real-time defect detection
  - Rolling window status calculation
  - Visual annotations with bounding boxes
  - Status panel overlay
  - FPS counter
  - Statistics export
  - Webcam support
  - Interactive controls (pause/resume)
- **Rolling OK/NOK Logic**:
  - Analyzes last N frames (default: 30)
  - OK if < 20% frames have defects
  - NOK if >= 20% frames have defects

#### 4. export_trt.py (300 lines)
- **Purpose**: Export model to TensorRT for optimized inference
- **Features**:
  - FP16 precision (default)
  - FP32 precision (optional)
  - Dynamic shape support
  - Configurable workspace
  - Verification with benchmarking
  - Performance metrics
- **Benefits**:
  - 2-3x faster inference
  - Lower memory usage
  - Optimized for NVIDIA GPUs

### Configuration Files

#### data.yaml (21 lines)
- Dataset root path
- Train/val/test split paths
- 6 NEU-DET class names:
  1. crazing
  2. inclusion
  3. patches
  4. pitted_surface
  5. rolled_in_scale
  6. scratches

#### requirements.txt (22 lines)
Core dependencies:
- ultralytics>=8.0.0 (YOLO framework)
- torch>=2.0.0 (Deep learning)
- opencv-python>=4.8.0 (Computer vision)
- matplotlib>=3.7.0 (Visualization)
- scikit-learn>=1.3.0 (Metrics)
- pandas>=2.0.0 (Data handling)
- seaborn>=0.12.0 (Statistical plots)
- numpy, scipy, pillow, pyyaml, tqdm

### Documentation

#### README.md (524 lines)
Comprehensive documentation including:
- Feature overview
- Project structure
- Dataset information
- Installation guide
- Quick start tutorial
- Detailed usage for all scripts
- Advanced metrics explanation
- Configuration tips
- Troubleshooting

#### .gitignore (458 bytes)
Excludes:
- Python cache (__pycache__, *.pyc)
- Virtual environments (venv/, env/)
- Data and models (data/, runs/, *.pt, *.engine)
- IDE files (.vscode/, .idea/)
- OS files (.DS_Store, Thumbs.db)

## Usage Examples

### 1. Training
```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 100 --batch 16
```

### 2. Evaluation
```bash
python eval_report.py --weights runs/train/neu_det_yolo/weights/best.pt --data data.yaml
```

### 3. Video Inference
```bash
python video_infer.py --weights best.pt --source video.mp4 --output result.mp4
```

### 4. TensorRT Export
```bash
python export_trt.py --weights best.pt --verify
```

## Statistics

- **Total Lines of Code**: 2,418 lines
- **Total Size**: ~78KB
- **Number of Scripts**: 4 main scripts
- **Number of Config Files**: 2 (data.yaml, requirements.txt)
- **Documentation**: 524 lines in README

## Key Features Summary

1. ✅ Complete training pipeline with YOLO
2. ✅ Advanced evaluation with 7+ metric types
3. ✅ Real-time video inference with rolling status
4. ✅ TensorRT export for production deployment
5. ✅ Comprehensive documentation
6. ✅ Professional project structure
7. ✅ Ready for NEU-DET dataset (6 classes)
8. ✅ All requested features implemented
