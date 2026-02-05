# API Documentation - Programmatic Usage

This document shows how to use the YOLO metal defect detection system programmatically in Python.

## Installation

```python
# Install required packages
!pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn scipy pillow pyyaml tqdm netcal
```

## Basic Usage Examples

### 1. Training

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on NEU-DET dataset
results = model.train(
    data='data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    patience=50,
    cache=True,
    augment=True,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,
)

print(f"Training completed! Best model: {model.trainer.best}")
```

### 2. Inference on Images

```python
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/train/neu-det/weights/best.pt')

# Predict on single image
results = model('path/to/image.jpg', conf=0.25, iou=0.45)

# Process results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs (if available)
    probs = result.probs  # Class probabilities for classification outputs
    
    # Get bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        class_name = result.names[cls]
        
        print(f"Detected {class_name} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    # Save annotated image
    result.save('output_image.jpg')
```

### 3. Batch Inference

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/train/neu-det/weights/best.pt')

# Predict on directory of images
image_dir = Path('path/to/images')
results = model.predict(
    source=image_dir,
    conf=0.25,
    iou=0.45,
    save=True,
    save_txt=True,  # Save results in YOLO format
    save_conf=True,
)

# Process all results
for i, result in enumerate(results):
    print(f"Image {i}: {len(result.boxes)} detections")
```

### 4. Video Inference with Real-time Processing

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/train/neu-det/weights/best.pt')

# Open video
cap = cv2.VideoCapture('test_video.mp4')

# Process frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame, conf=0.25, verbose=False)
    
    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('Defect Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5. Evaluation

```python
from ultralytics import YOLO

model = YOLO('runs/train/neu-det/weights/best.pt')

# Run validation
results = model.val(
    data='data.yaml',
    split='val',
    conf=0.25,
    iou=0.45,
    plots=True,
    save_json=True,
)

# Access metrics
print(f"mAP@50: {results.box.map50:.4f}")
print(f"mAP@50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")

# Per-class metrics
for i, class_name in enumerate(model.names.values()):
    print(f"{class_name}: AP50={results.box.ap50[i]:.4f}")
```

### 6. Export to Different Formats

```python
from ultralytics import YOLO

model = YOLO('runs/train/neu-det/weights/best.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640, dynamic=True)

# Export to TensorRT (requires CUDA and TensorRT)
model.export(format='engine', imgsz=640, half=True, workspace=4)

# Export to TorchScript
model.export(format='torchscript', imgsz=640)

# Export to CoreML (macOS)
model.export(format='coreml', imgsz=640)

# Export to TFLite (mobile)
model.export(format='tflite', imgsz=640, int8=True)
```

### 7. Custom Defect Detection Pipeline

```python
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class DefectDetector:
    """Custom defect detection pipeline with quality assessment"""
    
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = list(self.model.names.values())
        
    def detect(self, image):
        """Detect defects in image"""
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        return results[0]
    
    def is_ok(self, image, min_defects=1):
        """Check if image is OK (no defects) or NOK (has defects)"""
        result = self.detect(image)
        num_defects = len(result.boxes) if result.boxes else 0
        return num_defects < min_defects, num_defects
    
    def get_defect_stats(self, image):
        """Get detailed defect statistics"""
        result = self.detect(image)
        
        stats = {
            'total_defects': 0,
            'defect_classes': {},
            'defect_locations': [],
            'confidence_scores': [],
        }
        
        if result.boxes:
            stats['total_defects'] = len(result.boxes)
            
            for box in result.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names[cls_id]
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Count by class
                stats['defect_classes'][cls_name] = stats['defect_classes'].get(cls_name, 0) + 1
                
                # Store locations
                stats['defect_locations'].append({
                    'class': cls_name,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                })
                
                stats['confidence_scores'].append(conf)
        
        if stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['min_confidence'] = np.min(stats['confidence_scores'])
            stats['max_confidence'] = np.max(stats['confidence_scores'])
        
        return stats

# Usage example
detector = DefectDetector('runs/train/neu-det/weights/best.pt', conf_threshold=0.25)

# Check single image
image = cv2.imread('test_image.jpg')
is_ok, num_defects = detector.is_ok(image)
print(f"Quality: {'OK' if is_ok else 'NOK'} ({num_defects} defects)")

# Get detailed stats
stats = detector.get_defect_stats(image)
print(f"Defect Statistics: {stats}")
```

### 8. Rolling Statistics for Video

```python
from collections import deque
import numpy as np

class RollingQualityMonitor:
    """Monitor quality with rolling statistics"""
    
    def __init__(self, detector, window_size=30):
        self.detector = detector
        self.window_size = window_size
        self.ok_window = deque(maxlen=window_size)
        self.defect_window = deque(maxlen=window_size)
        
    def update(self, image):
        """Update with new frame"""
        is_ok, num_defects = self.detector.is_ok(image)
        
        self.ok_window.append(1 if is_ok else 0)
        self.defect_window.append(num_defects)
        
        return is_ok, num_defects
    
    def get_stats(self):
        """Get rolling statistics"""
        if not self.ok_window:
            return None
        
        total = len(self.ok_window)
        ok_count = sum(self.ok_window)
        
        return {
            'ok_rate': ok_count / total,
            'nok_rate': (total - ok_count) / total,
            'avg_defects': np.mean(self.defect_window),
            'max_defects': max(self.defect_window),
            'total_frames': total,
        }

# Usage
detector = DefectDetector('runs/train/neu-det/weights/best.pt')
monitor = RollingQualityMonitor(detector, window_size=30)

cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    is_ok, num_defects = monitor.update(frame)
    stats = monitor.get_stats()
    
    print(f"Frame OK: {is_ok}, Rolling OK Rate: {stats['ok_rate']*100:.1f}%")

cap.release()
```

### 9. Multi-threshold Analysis

```python
import numpy as np
from ultralytics import YOLO

def analyze_thresholds(model_path, data_yaml, thresholds=None):
    """Analyze model performance at different confidence thresholds"""
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.95, 20)
    
    model = YOLO(model_path)
    results = []
    
    for conf in thresholds:
        # Run validation at this threshold
        val_results = model.val(
            data=data_yaml,
            conf=conf,
            iou=0.45,
            verbose=False,
            plots=False,
        )
        
        # Extract metrics
        metrics = {
            'threshold': conf,
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
        }
        
        # Calculate F1
        p, r = metrics['precision'], metrics['recall']
        metrics['f1'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        results.append(metrics)
    
    # Find optimal threshold (max F1)
    best = max(results, key=lambda x: x['f1'])
    
    print(f"Optimal threshold: {best['threshold']:.3f}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    
    return results, best

# Usage
results, best = analyze_thresholds(
    'runs/train/neu-det/weights/best.pt',
    'data.yaml',
    thresholds=np.linspace(0.05, 0.95, 30)
)
```

## Advanced Features

### Custom Callbacks

```python
from ultralytics import YOLO

def on_train_epoch_end(trainer):
    """Custom callback for end of training epoch"""
    print(f"Epoch {trainer.epoch} completed!")
    print(f"  Train loss: {trainer.loss:.4f}")

# Add callback
model = YOLO('yolov8n.pt')
model.add_callback('on_train_epoch_end', on_train_epoch_end)

# Train
model.train(data='data.yaml', epochs=100)
```

### Hyperparameter Tuning

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Run hyperparameter tuning
results = model.tune(
    data='data.yaml',
    epochs=30,
    iterations=100,
    optimizer='AdamW',
    plots=True,
    save=True,
)
```

## Tips and Best Practices

1. **Use batch inference** for better performance
2. **Adjust confidence threshold** based on your use case
3. **Monitor GPU memory** when processing large batches
4. **Use TensorRT** for production deployment
5. **Implement error handling** for production systems
6. **Cache models** to avoid repeated loading
7. **Use multiprocessing** for parallel video processing

## References

- Ultralytics Documentation: https://docs.ultralytics.com/
- YOLO Tutorial: https://docs.ultralytics.com/modes/
- Model Export Guide: https://docs.ultralytics.com/modes/export/
