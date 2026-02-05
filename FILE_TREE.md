# Project File Tree

## Complete Project Structure

```
Metal_defect_detection_yolo26/
│
├── README.md                      # Complete documentation and usage guide
├── QUICKSTART.md                  # Quick start guide for rapid setup
├── API_DOCUMENTATION.md           # Programmatic API usage examples
│
├── requirements.txt               # Python dependencies
├── data.yaml                      # NEU-DET dataset configuration (6 classes)
├── .gitignore                     # Git ignore rules
│
├── train.py                       # YOLO training script
├── eval_report.py                 # Comprehensive evaluation with advanced stats
│                                  #   - Per-class P/R/F1, mAP50, mAP50-95
│                                  #   - Confusion matrix visualization
│                                  #   - PR curves
│                                  #   - ECE calibration analysis
│                                  #   - FP/FN analysis
│                                  #   - Threshold sweep optimization
├── video_infer.py                 # Video inference with rolling OK/NOK detection
├── export_trt.py                  # FP16 TensorRT export script
│
├── prepare_dataset.py             # Dataset preparation and conversion helper
├── verify_setup.py                # Installation and setup verification
├── show_tree.py                   # Project structure visualization
├── end_to_end_example.sh          # Complete pipeline example script
│
├── datasets/                      # Dataset directory (created by user)
│   └── NEU-DET/
│       ├── images/
│       │   ├── train/             # Training images
│       │   ├── val/               # Validation images
│       │   └── test/              # Test images (optional)
│       └── labels/
│           ├── train/             # Training labels (YOLO format)
│           ├── val/               # Validation labels
│           └── test/              # Test labels (optional)
│
├── runs/                          # Training outputs (auto-generated)
│   └── train/
│       └── neu-det/
│           ├── weights/
│           │   ├── best.pt        # Best model checkpoint
│           │   └── last.pt        # Last epoch checkpoint
│           ├── results.png        # Training curves
│           ├── results.csv        # Training metrics
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           ├── F1_curve.png
│           ├── P_curve.png
│           ├── PR_curve.png
│           ├── R_curve.png
│           └── args.yaml          # Training arguments
│
├── eval_results/                  # Evaluation outputs (auto-generated)
│   ├── per_class_metrics.png     # Bar charts of P/R/F1/AP by class
│   ├── confusion_matrix.png      # Confusion matrix (raw & normalized)
│   ├── threshold_sweep.png       # Metrics vs confidence threshold
│   ├── evaluation_report.txt     # Text summary of all metrics
│   └── evaluation_results.json   # JSON format results (optional)
│
├── output_videos/                 # Processed videos (auto-generated)
│   └── {video_name}_processed.mp4
│
└── weights/                       # Exported models (auto-generated)
    ├── best.onnx                  # ONNX format
    ├── best.engine                # TensorRT engine (FP16)
    └── best.torchscript           # TorchScript format
```

## File Descriptions

### Core Scripts

| File | Description | Key Features |
|------|-------------|--------------|
| `train.py` | Training script | YOLO training on NEU-DET with full hyperparameter control |
| `eval_report.py` | Evaluation script | Advanced metrics, confusion matrix, PR curves, ECE, FP/FN analysis, threshold sweep |
| `video_infer.py` | Video inference | Real-time defect detection with rolling OK/NOK statistics |
| `export_trt.py` | Model export | TensorRT FP16 optimization for production deployment |

### Helper Scripts

| File | Description | Purpose |
|------|-------------|---------|
| `prepare_dataset.py` | Dataset preparation | Convert NEU-DET to YOLO format with automatic train/val/test split |
| `verify_setup.py` | Setup verification | Validate installation and project structure |
| `show_tree.py` | Tree visualization | Display project structure |
| `end_to_end_example.sh` | Pipeline example | Complete workflow from training to deployment |

### Configuration Files

| File | Description | Content |
|------|-------------|---------|
| `data.yaml` | Dataset config | Paths and class definitions for NEU-DET (6 classes) |
| `requirements.txt` | Dependencies | All required Python packages |
| `.gitignore` | Git ignore | Exclude generated files, datasets, and outputs |

### Documentation

| File | Description | Audience |
|------|-------------|----------|
| `README.md` | Complete guide | All users - setup, usage, troubleshooting |
| `QUICKSTART.md` | Quick start | Users who want to get started immediately |
| `API_DOCUMENTATION.md` | API reference | Developers integrating the system programmatically |
| `FILE_TREE.md` | This file | Understanding project structure |

## NEU-DET Dataset Classes

The project is configured for 6 classes of surface defects:

1. **Crazing (Cr)** - Fine cracks on the surface
2. **Inclusion (In)** - Non-metallic inclusions
3. **Patches (Pa)** - Irregular patches on surface
4. **Pitted Surface (PS)** - Small pits or holes
5. **Rolled-in Scale (RS)** - Oxide scale rolled into surface
6. **Scratches (Sc)** - Linear scratches

## Output Files

### Training Outputs (`runs/train/neu-det/`)

- **weights/best.pt** - Best model based on validation mAP
- **weights/last.pt** - Last epoch checkpoint
- **results.png** - Training curves (loss, metrics over epochs)
- **results.csv** - Numerical training metrics
- **confusion_matrix.png** - Confusion matrix from validation
- **PR_curve.png** - Precision-Recall curve
- **F1_curve.png**, **P_curve.png**, **R_curve.png** - Metric curves

### Evaluation Outputs (`eval_results/`)

- **per_class_metrics.png** - 4-panel chart showing:
  - Precision by class (bar chart)
  - Recall by class (bar chart)
  - F1 score by class (bar chart)
  - AP@50 by class (bar chart)
  
- **confusion_matrix.png** - 2-panel visualization:
  - Raw counts
  - Normalized (percentages)
  
- **threshold_sweep.png** - 4-panel analysis:
  - Precision & Recall vs threshold
  - F1 score vs threshold (with optimal point marked)
  - mAP@50 vs threshold
  - mAP@50-95 vs threshold
  
- **evaluation_report.txt** - Text summary including:
  - Overall metrics (P/R/F1/mAP)
  - Per-class metrics table
  - Optimal threshold recommendation
  
- **evaluation_results.json** - Machine-readable results (optional)

### Video Outputs (`output_videos/`)

Processed videos with:
- Bounding boxes on detected defects
- Class labels and confidence scores
- Status indicator (OK/NOK)
- Rolling statistics panel showing:
  - Frame number and FPS
  - Current status
  - Detections count
  - Rolling window OK/NOK rates
  - Average defects per frame

### Exported Models (`weights/`)

- **best.onnx** - ONNX format (cross-platform)
- **best.engine** - TensorRT engine (NVIDIA GPU, FP16)
- **best.torchscript** - TorchScript (PyTorch ecosystem)
- Other formats as configured

## Size Estimates

Typical file sizes (approximate):

| File/Directory | Size | Notes |
|----------------|------|-------|
| Scripts (all .py) | ~100 KB | Python code |
| Documentation | ~50 KB | Markdown files |
| Dataset (NEU-DET) | ~200 MB | 1800 images + annotations |
| Trained model (.pt) | 3-50 MB | Depends on variant (n/s/m/l/x) |
| Training outputs | ~50 MB | Plots, logs, checkpoints |
| Evaluation outputs | ~5 MB | Plots and reports |
| TensorRT engine | 5-100 MB | Optimized model |

Total project size (with dataset and trained models): **~500 MB - 1 GB**

## Dependencies Summary

From `requirements.txt`:

**Core ML Libraries:**
- ultralytics (YOLO implementation)
- torch, torchvision (PyTorch)
- opencv-python (Computer vision)

**Data Science:**
- numpy, pandas (Data manipulation)
- matplotlib, seaborn (Visualization)
- scikit-learn, scipy (ML utilities)

**Utilities:**
- PyYAML (Configuration)
- tqdm (Progress bars)
- Pillow (Image processing)

**Optional:**
- netcal (Calibration metrics)
- tensorrt (GPU optimization)

## Workflow Summary

```
1. Setup
   ├── Install: pip install -r requirements.txt
   ├── Verify: python verify_setup.py
   └── Prepare: python prepare_dataset.py

2. Training
   ├── Train: python train.py --data data.yaml
   └── Output: runs/train/neu-det/weights/best.pt

3. Evaluation
   ├── Evaluate: python eval_report.py --weights best.pt
   └── Output: eval_results/ (metrics, plots, reports)

4. Deployment
   ├── Export: python export_trt.py --weights best.pt
   ├── Test: python video_infer.py --weights best.pt --video test.mp4
   └── Integrate: Use API from API_DOCUMENTATION.md
```

## Notes

- All auto-generated directories are excluded from git (see `.gitignore`)
- Dataset must be prepared by user (not included in repository)
- Training outputs vary based on selected model variant and hyperparameters
- TensorRT export requires NVIDIA GPU with CUDA and TensorRT installed
