"""
Advanced Evaluation and Reporting for YOLO26 Metal Defect Detection
This script provides comprehensive evaluation metrics including:
- Per-class Precision, Recall, F1-Score
- mAP50 and mAP50-95
- Confusion Matrix
- Precision-Recall Curves
- Expected Calibration Error (ECE)
- False Positive/False Negative Analysis
- Threshold Sweep Analysis
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
import yaml


class YOLOEvaluator:
    """Advanced evaluation class for YOLO models."""
    
    def __init__(self, model_path, data_path, device=''):
        """Initialize evaluator with model and data paths."""
        self.model = YOLO(model_path)
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load data configuration
        with open(data_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = list(self.data_config['names'].values())
        self.num_classes = self.data_config['nc']
        
        # Storage for predictions and ground truth
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_confidences = []
        
    def run_inference(self, split='val', imgsz=640, conf_thresh=0.001):
        """Run inference on dataset and collect predictions."""
        print(f"\nRunning inference on {split} set...")
        
        # Get dataset path
        dataset_path = Path(self.data_config['path']) / self.data_config[split]
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(dataset_path.glob(ext)))
        
        if not image_files:
            raise ValueError(f"No images found in {dataset_path}")
        
        print(f"Found {len(image_files)} images")
        
        # Reset storage
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_confidences = []
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            # Run inference
            results = self.model(img_path, imgsz=imgsz, conf=conf_thresh, verbose=False)
            
            # Get predictions
            pred_boxes = []
            pred_classes = []
            pred_confs = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    pred_boxes = boxes.xyxy.cpu().numpy()
                    pred_classes = boxes.cls.cpu().numpy().astype(int)
                    pred_confs = boxes.conf.cpu().numpy()
            
            # Load ground truth labels
            label_path = img_path.parent.parent / 'labels' / split / f"{img_path.stem}.txt"
            gt_boxes = []
            gt_classes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert from YOLO format to xyxy
                            img = cv2.imread(str(img_path))
                            h, w = img.shape[:2]
                            x1 = (x_center - width / 2) * w
                            y1 = (y_center - height / 2) * h
                            x2 = (x_center + width / 2) * w
                            y2 = (y_center + height / 2) * h
                            
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(cls)
            
            # Store for evaluation
            self.all_predictions.append({
                'boxes': np.array(pred_boxes),
                'classes': np.array(pred_classes),
                'confidences': np.array(pred_confs),
                'image_path': str(img_path)
            })
            
            self.all_ground_truth.append({
                'boxes': np.array(gt_boxes),
                'classes': np.array(gt_classes),
                'image_path': str(img_path)
            })
        
        print(f"Inference completed on {len(image_files)} images")
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def compute_metrics_at_threshold(self, conf_thresh=0.25, iou_thresh=0.5):
        """Compute metrics at specific confidence and IoU thresholds."""
        tp_per_class = np.zeros(self.num_classes)
        fp_per_class = np.zeros(self.num_classes)
        fn_per_class = np.zeros(self.num_classes)
        gt_count_per_class = np.zeros(self.num_classes)
        
        for pred, gt in zip(self.all_predictions, self.all_ground_truth):
            # Filter by confidence threshold
            valid_idx = pred['confidences'] >= conf_thresh
            pred_boxes = pred['boxes'][valid_idx]
            pred_classes = pred['classes'][valid_idx]
            
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            # Count ground truth per class
            for cls in gt_classes:
                gt_count_per_class[cls] += 1
            
            # Track matched GT boxes
            matched_gt = set()
            
            # Match predictions to ground truth
            for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
                best_iou = 0
                best_gt_idx = -1
                
                for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if j in matched_gt:
                        continue
                    if pred_cls != gt_cls:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    tp_per_class[pred_cls] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp_per_class[pred_cls] += 1
            
            # Count false negatives
            for j, gt_cls in enumerate(gt_classes):
                if j not in matched_gt:
                    fn_per_class[gt_cls] += 1
        
        # Calculate metrics per class
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1 = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            if tp_per_class[i] + fp_per_class[i] > 0:
                precision[i] = tp_per_class[i] / (tp_per_class[i] + fp_per_class[i])
            if tp_per_class[i] + fn_per_class[i] > 0:
                recall[i] = tp_per_class[i] / (tp_per_class[i] + fn_per_class[i])
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_per_class,
            'fp': fp_per_class,
            'fn': fn_per_class,
            'gt_count': gt_count_per_class
        }
    
    def generate_confusion_matrix(self, output_dir, conf_thresh=0.25, iou_thresh=0.5):
        """Generate and save confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        y_true = []
        y_pred = []
        
        for pred, gt in zip(self.all_predictions, self.all_ground_truth):
            valid_idx = pred['confidences'] >= conf_thresh
            pred_boxes = pred['boxes'][valid_idx]
            pred_classes = pred['classes'][valid_idx]
            
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            matched_gt = set()
            matched_pred = set()
            
            # Match predictions to GT
            for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
                best_iou = 0
                best_gt_idx = -1
                best_gt_cls = -1
                
                for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                        best_gt_cls = gt_cls
                
                if best_iou >= iou_thresh:
                    y_true.append(best_gt_cls)
                    y_pred.append(pred_cls)
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix (conf={conf_thresh}, IoU={iou_thresh})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
        
        return cm
    
    def generate_pr_curves(self, output_dir, iou_thresh=0.5):
        """Generate precision-recall curves for each class."""
        print("\nGenerating PR curves...")
        
        plt.figure(figsize=(15, 10))
        
        for cls_idx in range(self.num_classes):
            all_confs = []
            all_labels = []
            
            for pred, gt in zip(self.all_predictions, self.all_ground_truth):
                pred_boxes = pred['boxes']
                pred_classes = pred['classes']
                pred_confs = pred['confidences']
                
                gt_boxes = gt['boxes']
                gt_classes = gt['classes']
                
                # Get predictions for this class
                cls_mask = pred_classes == cls_idx
                cls_pred_boxes = pred_boxes[cls_mask]
                cls_pred_confs = pred_confs[cls_mask]
                
                # Get GT for this class
                cls_gt_boxes = gt_boxes[gt_classes == cls_idx]
                
                # Match predictions to GT
                for pred_box, conf in zip(cls_pred_boxes, cls_pred_confs):
                    is_tp = False
                    for gt_box in cls_gt_boxes:
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou >= iou_thresh:
                            is_tp = True
                            break
                    
                    all_confs.append(conf)
                    all_labels.append(1 if is_tp else 0)
            
            if len(all_confs) > 0:
                precision, recall, _ = precision_recall_curve(all_labels, all_confs)
                ap = auc(recall, precision)
                plt.plot(recall, precision, label=f'{self.class_names[cls_idx]} (AP={ap:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PR curves saved to {output_dir / 'pr_curves.png'}")
    
    def calculate_ece(self, output_dir, n_bins=10, conf_thresh=0.25, iou_thresh=0.5):
        """Calculate Expected Calibration Error."""
        print("\nCalculating Expected Calibration Error...")
        
        confidences = []
        accuracies = []
        
        for pred, gt in zip(self.all_predictions, self.all_ground_truth):
            valid_idx = pred['confidences'] >= conf_thresh
            pred_boxes = pred['boxes'][valid_idx]
            pred_classes = pred['classes'][valid_idx]
            pred_confs = pred['confidences'][valid_idx]
            
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            for pred_box, pred_cls, conf in zip(pred_boxes, pred_classes, pred_confs):
                is_correct = False
                
                for gt_box, gt_cls in zip(gt_boxes, gt_classes):
                    if pred_cls == gt_cls:
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou >= iou_thresh:
                            is_correct = True
                            break
                
                confidences.append(conf)
                accuracies.append(1 if is_correct else 0)
        
        if len(confidences) == 0:
            print("No predictions to calculate ECE")
            return 0.0
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidences.append(confidences[mask].mean())
                bin_accuracies.append(accuracies[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_confidences.append(0)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_confidences = np.array(bin_confidences)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)
        
        # Calculate ECE
        ece = np.sum(bin_counts * np.abs(bin_confidences - bin_accuracies)) / len(confidences)
        
        # Plot calibration curve
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        valid_bins = bin_counts > 0
        plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                'o-', label=f'Model (ECE={ece:.4f})', linewidth=2, markersize=8)
        
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            if count > 0:
                plt.text(conf, acc, f'{count}', fontsize=8, ha='center', va='bottom')
        
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Calibration Curve', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ECE: {ece:.4f}")
        print(f"Calibration curve saved to {output_dir / 'calibration_curve.png'}")
        
        return ece
    
    def fp_fn_analysis(self, output_dir, conf_thresh=0.25, iou_thresh=0.5):
        """Analyze false positives and false negatives."""
        print("\nAnalyzing false positives and false negatives...")
        
        fp_by_class = {cls: [] for cls in range(self.num_classes)}
        fn_by_class = {cls: [] for cls in range(self.num_classes)}
        
        for pred, gt in zip(self.all_predictions, self.all_ground_truth):
            valid_idx = pred['confidences'] >= conf_thresh
            pred_boxes = pred['boxes'][valid_idx]
            pred_classes = pred['classes'][valid_idx]
            pred_confs = pred['confidences'][valid_idx]
            
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            matched_gt = set()
            
            # Identify FPs
            for pred_box, pred_cls, conf in zip(pred_boxes, pred_classes, pred_confs):
                best_iou = 0
                best_gt_idx = -1
                
                for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if j in matched_gt or pred_cls != gt_cls:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    matched_gt.add(best_gt_idx)
                else:
                    fp_by_class[pred_cls].append({
                        'confidence': conf,
                        'image': pred['image_path']
                    })
            
            # Identify FNs
            for j, gt_cls in enumerate(gt_classes):
                if j not in matched_gt:
                    fn_by_class[gt_cls].append({
                        'image': gt['image_path']
                    })
        
        # Create summary
        fp_counts = [len(fp_by_class[i]) for i in range(self.num_classes)]
        fn_counts = [len(fn_by_class[i]) for i in range(self.num_classes)]
        
        # Plot FP/FN counts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # False Positives
        ax1.bar(range(self.num_classes), fp_counts, color='red', alpha=0.7)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('False Positives by Class', fontsize=14)
        ax1.set_xticks(range(self.num_classes))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # False Negatives
        ax2.bar(range(self.num_classes), fn_counts, color='orange', alpha=0.7)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('False Negatives by Class', fontsize=14)
        ax2.set_xticks(range(self.num_classes))
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fp_fn_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"FP/FN analysis saved to {output_dir / 'fp_fn_analysis.png'}")
        
        # Save detailed analysis
        analysis = {
            'false_positives': {self.class_names[i]: fp_counts[i] for i in range(self.num_classes)},
            'false_negatives': {self.class_names[i]: fn_counts[i] for i in range(self.num_classes)}
        }
        
        with open(output_dir / 'fp_fn_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return fp_by_class, fn_by_class
    
    def threshold_sweep(self, output_dir, iou_thresh=0.5):
        """Perform threshold sweep analysis."""
        print("\nPerforming threshold sweep...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = {
            'thresholds': thresholds.tolist(),
            'precision': {cls: [] for cls in self.class_names},
            'recall': {cls: [] for cls in self.class_names},
            'f1': {cls: [] for cls in self.class_names}
        }
        
        for thresh in tqdm(thresholds, desc="Sweeping thresholds"):
            metrics = self.compute_metrics_at_threshold(conf_thresh=thresh, iou_thresh=iou_thresh)
            
            for i, cls_name in enumerate(self.class_names):
                results['precision'][cls_name].append(metrics['precision'][i])
                results['recall'][cls_name].append(metrics['recall'][i])
                results['f1'][cls_name].append(metrics['f1'][i])
        
        # Plot threshold sweep
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision vs threshold
        ax = axes[0, 0]
        for cls_name in self.class_names:
            ax.plot(thresholds, results['precision'][cls_name], label=cls_name, linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision vs Confidence Threshold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Recall vs threshold
        ax = axes[0, 1]
        for cls_name in self.class_names:
            ax.plot(thresholds, results['recall'][cls_name], label=cls_name, linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall vs Confidence Threshold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # F1 vs threshold
        ax = axes[1, 0]
        for cls_name in self.class_names:
            ax.plot(thresholds, results['f1'][cls_name], label=cls_name, linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score vs Confidence Threshold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Mean metrics vs threshold
        ax = axes[1, 1]
        mean_precision = [np.mean([results['precision'][cls][i] for cls in self.class_names]) 
                         for i in range(len(thresholds))]
        mean_recall = [np.mean([results['recall'][cls][i] for cls in self.class_names]) 
                      for i in range(len(thresholds))]
        mean_f1 = [np.mean([results['f1'][cls][i] for cls in self.class_names]) 
                  for i in range(len(thresholds))]
        
        ax.plot(thresholds, mean_precision, label='Mean Precision', linewidth=2)
        ax.plot(thresholds, mean_recall, label='Mean Recall', linewidth=2)
        ax.plot(thresholds, mean_f1, label='Mean F1', linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Mean Metrics vs Confidence Threshold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_sweep.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Threshold sweep saved to {output_dir / 'threshold_sweep.png'}")
        
        # Save results
        with open(output_dir / 'threshold_sweep.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_full_report(self, output_dir, conf_thresh=0.25, iou_thresh=0.5):
        """Generate complete evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("YOLO26 Advanced Evaluation Report")
        print("=" * 60)
        
        # Run standard YOLO validation
        print("\nRunning YOLO validation...")
        metrics = self.model.val(data=self.data_config, imgsz=640, conf=conf_thresh, iou=iou_thresh)
        
        # Calculate metrics at threshold
        print(f"\nCalculating metrics at conf={conf_thresh}, IoU={iou_thresh}...")
        metrics_at_thresh = self.compute_metrics_at_threshold(conf_thresh, iou_thresh)
        
        # Print per-class metrics
        print("\n" + "-" * 60)
        print("Per-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        
        for i, cls_name in enumerate(self.class_names):
            print(f"{cls_name:<20} {metrics_at_thresh['precision'][i]:<12.4f} "
                  f"{metrics_at_thresh['recall'][i]:<12.4f} {metrics_at_thresh['f1'][i]:<12.4f}")
        
        print("-" * 60)
        print(f"{'Mean':<20} {metrics_at_thresh['precision'].mean():<12.4f} "
              f"{metrics_at_thresh['recall'].mean():<12.4f} {metrics_at_thresh['f1'].mean():<12.4f}")
        print("-" * 60)
        
        # Print mAP scores
        print("\nmAP Scores:")
        print("-" * 60)
        if hasattr(metrics, 'box'):
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
        print("-" * 60)
        
        # Generate visualizations
        self.generate_confusion_matrix(output_dir, conf_thresh, iou_thresh)
        self.generate_pr_curves(output_dir, iou_thresh)
        ece = self.calculate_ece(output_dir, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        self.fp_fn_analysis(output_dir, conf_thresh, iou_thresh)
        self.threshold_sweep(output_dir, iou_thresh)
        
        # Save summary report
        report = {
            'evaluation_parameters': {
                'confidence_threshold': conf_thresh,
                'iou_threshold': iou_thresh,
            },
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(metrics_at_thresh['precision'][i]),
                    'recall': float(metrics_at_thresh['recall'][i]),
                    'f1_score': float(metrics_at_thresh['f1'][i]),
                    'true_positives': int(metrics_at_thresh['tp'][i]),
                    'false_positives': int(metrics_at_thresh['fp'][i]),
                    'false_negatives': int(metrics_at_thresh['fn'][i]),
                    'ground_truth_count': int(metrics_at_thresh['gt_count'][i])
                }
                for i in range(self.num_classes)
            },
            'mean_metrics': {
                'precision': float(metrics_at_thresh['precision'].mean()),
                'recall': float(metrics_at_thresh['recall'].mean()),
                'f1_score': float(metrics_at_thresh['f1'].mean())
            },
            'map_scores': {
                'mAP50': float(metrics.box.map50) if hasattr(metrics, 'box') else None,
                'mAP50_95': float(metrics.box.map) if hasattr(metrics, 'box') else None
            },
            'calibration': {
                'ECE': float(ece)
            }
        }
        
        with open(output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_report = f"""# YOLO26 Evaluation Report - NEU-DET Metal Defect Detection

## Evaluation Parameters
- Confidence Threshold: {conf_thresh}
- IoU Threshold: {iou_thresh}
- Number of Classes: {self.num_classes}

## mAP Scores
- **mAP@0.5**: {metrics.box.map50:.4f if hasattr(metrics, 'box') else 'N/A'}
- **mAP@0.5:0.95**: {metrics.box.map:.4f if hasattr(metrics, 'box') else 'N/A'}

## Overall Metrics
- **Mean Precision**: {metrics_at_thresh['precision'].mean():.4f}
- **Mean Recall**: {metrics_at_thresh['recall'].mean():.4f}
- **Mean F1-Score**: {metrics_at_thresh['f1'].mean():.4f}

## Calibration
- **Expected Calibration Error (ECE)**: {ece:.4f}

## Per-Class Performance

| Class | Precision | Recall | F1-Score | TP | FP | FN | GT Count |
|-------|-----------|--------|----------|----|----|----|----|
"""
        for i, cls_name in enumerate(self.class_names):
            md_report += f"| {cls_name} | {metrics_at_thresh['precision'][i]:.4f} | "
            md_report += f"{metrics_at_thresh['recall'][i]:.4f} | {metrics_at_thresh['f1'][i]:.4f} | "
            md_report += f"{int(metrics_at_thresh['tp'][i])} | {int(metrics_at_thresh['fp'][i])} | "
            md_report += f"{int(metrics_at_thresh['fn'][i])} | {int(metrics_at_thresh['gt_count'][i])} |\n"
        
        md_report += f"\n## Visualizations\n\n"
        md_report += f"- [Confusion Matrix](confusion_matrix.png)\n"
        md_report += f"- [Precision-Recall Curves](pr_curves.png)\n"
        md_report += f"- [Calibration Curve](calibration_curve.png)\n"
        md_report += f"- [FP/FN Analysis](fp_fn_analysis.png)\n"
        md_report += f"- [Threshold Sweep](threshold_sweep.png)\n"
        
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(md_report)
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
        print(f"  - evaluation_report.json")
        print(f"  - evaluation_report.md")
        print(f"  - confusion_matrix.png")
        print(f"  - pr_curves.png")
        print(f"  - calibration_curve.png")
        print(f"  - fp_fn_analysis.png")
        print(f"  - threshold_sweep.png")
        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Evaluation for YOLO26 Metal Defect Detection'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to data.yaml configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for evaluation'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for matching predictions to ground truth'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (cuda:0, cpu, or empty for auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/eval',
        help='Output directory for evaluation results'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 60)
    print("YOLO26 Advanced Evaluation")
    print("=" * 60)
    print(f"Model: {args.weights}")
    print(f"Data config: {args.data}")
    print(f"Split: {args.split}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(args.weights, args.data, args.device)
    
    # Run inference
    evaluator.run_inference(args.split, args.imgsz, conf_thresh=0.001)
    
    # Generate full report
    evaluator.generate_full_report(args.output, args.conf, args.iou)


if __name__ == '__main__':
    main()
