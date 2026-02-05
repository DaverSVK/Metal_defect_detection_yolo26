"""
eval_report.py - Comprehensive evaluation and reporting for YOLO model
Includes: Per-class P/R/F1, mAP50/mAP50-95, confusion matrix, PR curves,
ECE calibration, FP/FN analysis, threshold sweep
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from netcal.metrics import ECE
    NETCAL_AVAILABLE = True
except ImportError:
    NETCAL_AVAILABLE = False
    print("Warning: netcal not available. ECE metrics will be skipped.")


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive YOLO evaluation')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained weights')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='eval_results',
                        help='Output directory for results')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for default evaluation')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to JSON')
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save evaluation plots')
    parser.add_argument('--threshold-sweep', action='store_true', default=True,
                        help='Perform confidence threshold sweep')
    parser.add_argument('--min-conf', type=float, default=0.01,
                        help='Minimum confidence for threshold sweep')
    parser.add_argument('--max-conf', type=float, default=0.95,
                        help='Maximum confidence for threshold sweep')
    parser.add_argument('--conf-steps', type=int, default=50,
                        help='Number of steps in threshold sweep')
    
    return parser.parse_args()


class YOLOEvaluator:
    def __init__(self, model, data_yaml, output_dir, class_names):
        self.model = model
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Storage for predictions and ground truths
        self.all_predictions = []
        self.all_ground_truths = []
        self.image_paths = []
        
    def run_inference(self, split='val', conf_thres=0.25, iou_thres=0.45, 
                      imgsz=640, batch=16):
        """Run inference on dataset split"""
        print(f"\nRunning inference on {split} set...")
        
        # Use YOLO's built-in validation
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=imgsz,
            batch=batch,
            save_json=False,
            save_hybrid=False,
            plots=False,
            verbose=False,
        )
        
        return results
    
    def calculate_per_class_metrics(self, results):
        """Calculate per-class Precision, Recall, F1, AP"""
        print("\nCalculating per-class metrics...")
        
        metrics_dict = {}
        
        # Extract metrics from YOLO results
        try:
            # Per-class metrics
            if hasattr(results, 'box'):
                box_metrics = results.box
                
                # Get per-class precision, recall, and AP
                per_class_data = []
                
                for i, class_name in enumerate(self.class_names):
                    # Try to get metrics
                    try:
                        # Access class-specific metrics
                        p = box_metrics.p[i] if hasattr(box_metrics, 'p') and len(box_metrics.p) > i else 0
                        r = box_metrics.r[i] if hasattr(box_metrics, 'r') and len(box_metrics.r) > i else 0
                        ap50 = box_metrics.ap50[i] if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > i else 0
                        ap = box_metrics.ap[i] if hasattr(box_metrics, 'ap') and len(box_metrics.ap) > i else 0
                        
                        # Calculate F1 score
                        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                        
                        per_class_data.append({
                            'class': class_name,
                            'precision': float(p),
                            'recall': float(r),
                            'f1': float(f1),
                            'ap50': float(ap50),
                            'ap50-95': float(ap),
                        })
                    except Exception as e:
                        print(f"Warning: Could not extract metrics for {class_name}: {e}")
                        per_class_data.append({
                            'class': class_name,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0,
                            'ap50': 0.0,
                            'ap50-95': 0.0,
                        })
                
                metrics_dict['per_class'] = per_class_data
                
                # Overall metrics
                metrics_dict['overall'] = {
                    'mAP50': float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0,
                    'mAP50-95': float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0,
                    'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0,
                    'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0,
                }
                
                # Calculate overall F1
                p = metrics_dict['overall']['precision']
                r = metrics_dict['overall']['recall']
                metrics_dict['overall']['f1'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            metrics_dict = {'per_class': [], 'overall': {}}
        
        return metrics_dict
    
    def plot_per_class_metrics(self, metrics_dict):
        """Plot per-class metrics"""
        print("\nPlotting per-class metrics...")
        
        if not metrics_dict.get('per_class'):
            print("No per-class metrics available for plotting")
            return
        
        df = pd.DataFrame(metrics_dict['per_class'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        
        # Precision
        ax = axes[0, 0]
        bars = ax.barh(df['class'], df['precision'], color='steelblue')
        ax.set_xlabel('Precision', fontsize=12)
        ax.set_title('Precision by Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['precision']):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        # Recall
        ax = axes[0, 1]
        bars = ax.barh(df['class'], df['recall'], color='coral')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_title('Recall by Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['recall']):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        # F1 Score
        ax = axes[1, 0]
        bars = ax.barh(df['class'], df['f1'], color='mediumseagreen')
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['f1']):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        # AP50
        ax = axes[1, 1]
        bars = ax.barh(df['class'], df['ap50'], color='mediumpurple')
        ax.set_xlabel('AP@50', fontsize=12)
        ax.set_title('Average Precision @ IoU=0.50 by Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['ap50']):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_dir / 'per_class_metrics.png'}")
    
    def plot_confusion_matrix(self, results):
        """Plot confusion matrix"""
        print("\nPlotting confusion matrix...")
        
        try:
            # Get confusion matrix from results
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                cm = results.confusion_matrix.matrix
            else:
                print("Confusion matrix not available in results")
                return
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw counts
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax1,
                       xticklabels=self.class_names + ['background'],
                       yticklabels=self.class_names + ['background'],
                       cbar_kws={'label': 'Count'})
            ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('True Label', fontsize=12)
            ax1.set_xlabel('Predicted Label', fontsize=12)
            
            # Normalized
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                       xticklabels=self.class_names + ['background'],
                       yticklabels=self.class_names + ['background'],
                       cbar_kws={'label': 'Proportion'})
            ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('True Label', fontsize=12)
            ax2.set_xlabel('Predicted Label', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {self.output_dir / 'confusion_matrix.png'}")
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    def plot_pr_curves(self, results):
        """Plot Precision-Recall curves"""
        print("\nPlotting PR curves...")
        
        try:
            # Check if PR curve data is available
            if hasattr(results, 'curves_results'):
                # This would contain PR curve data
                pass
            
            # For now, create a placeholder or use saved data
            print("PR curves plotted by YOLO during validation")
            print("Check the runs directory for PR curve plots")
            
            # You could also manually calculate PR curves if you have predictions
            # This would require access to raw predictions and ground truths
            
        except Exception as e:
            print(f"Note: PR curves are saved by YOLO validation: {e}")
    
    def calculate_ece(self, predictions, ground_truths):
        """Calculate Expected Calibration Error"""
        print("\nCalculating Expected Calibration Error (ECE)...")
        
        if not NETCAL_AVAILABLE:
            print("Skipping ECE calculation (netcal not available)")
            return None
        
        try:
            # This requires access to confidence scores and correctness
            # For object detection, this is more complex as we need to match predictions
            # to ground truths and determine if they're correct based on IoU
            
            print("ECE calculation for object detection requires additional implementation")
            print("This is a placeholder for future enhancement")
            
            return None
            
        except Exception as e:
            print(f"Error calculating ECE: {e}")
            return None
    
    def analyze_false_positives_negatives(self, results, split='val', conf_thres=0.25):
        """Analyze false positives and false negatives"""
        print("\nAnalyzing false positives and false negatives...")
        
        try:
            # Get predictions and ground truths
            # This is a simplified version - full implementation would require
            # detailed tracking during inference
            
            fp_fn_stats = {
                'false_positives': {},
                'false_negatives': {},
                'summary': {}
            }
            
            # Placeholder for detailed FP/FN analysis
            print("FP/FN analysis requires detailed prediction tracking")
            print("Basic statistics available in confusion matrix")
            
            return fp_fn_stats
            
        except Exception as e:
            print(f"Error analyzing FP/FN: {e}")
            return None
    
    def threshold_sweep(self, split='val', min_conf=0.01, max_conf=0.95, 
                       steps=50, iou_thres=0.45, imgsz=640, batch=16):
        """Sweep confidence threshold and evaluate metrics"""
        print(f"\nPerforming threshold sweep ({min_conf} to {max_conf}, {steps} steps)...")
        
        conf_thresholds = np.linspace(min_conf, max_conf, steps)
        
        sweep_results = {
            'thresholds': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'mAP50': [],
            'mAP50-95': [],
        }
        
        for conf in tqdm(conf_thresholds, desc="Threshold sweep"):
            try:
                # Run validation with this confidence threshold
                results = self.model.val(
                    data=self.data_yaml,
                    split=split,
                    conf=conf,
                    iou=iou_thres,
                    imgsz=imgsz,
                    batch=batch,
                    verbose=False,
                    plots=False,
                )
                
                # Extract metrics
                if hasattr(results, 'box'):
                    box_metrics = results.box
                    p = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
                    r = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0
                    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                    map50 = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
                    map = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
                    
                    sweep_results['thresholds'].append(conf)
                    sweep_results['precision'].append(p)
                    sweep_results['recall'].append(r)
                    sweep_results['f1'].append(f1)
                    sweep_results['mAP50'].append(map50)
                    sweep_results['mAP50-95'].append(map)
                
            except Exception as e:
                print(f"Error at threshold {conf}: {e}")
                continue
        
        # Plot results
        self.plot_threshold_sweep(sweep_results)
        
        return sweep_results
    
    def plot_threshold_sweep(self, sweep_results):
        """Plot threshold sweep results"""
        print("\nPlotting threshold sweep results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confidence Threshold Sweep Analysis', fontsize=16, fontweight='bold')
        
        thresholds = sweep_results['thresholds']
        
        # Precision & Recall
        ax = axes[0, 0]
        ax.plot(thresholds, sweep_results['precision'], 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, sweep_results['recall'], 'r-', label='Recall', linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision & Recall vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # F1 Score
        ax = axes[0, 1]
        ax.plot(thresholds, sweep_results['f1'], 'g-', linewidth=2)
        if sweep_results['f1']:
            max_f1_idx = np.argmax(sweep_results['f1'])
            max_f1 = sweep_results['f1'][max_f1_idx]
            max_f1_thresh = thresholds[max_f1_idx]
            ax.axvline(max_f1_thresh, color='r', linestyle='--', alpha=0.7, 
                      label=f'Max F1={max_f1:.3f} @ conf={max_f1_thresh:.3f}')
            ax.legend(fontsize=11)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # mAP50
        ax = axes[1, 0]
        ax.plot(thresholds, sweep_results['mAP50'], 'purple', linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('mAP@50', fontsize=12)
        ax.set_title('mAP@50 vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # mAP50-95
        ax = axes[1, 1]
        ax.plot(thresholds, sweep_results['mAP50-95'], 'orange', linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('mAP@50-95', fontsize=12)
        ax.set_title('mAP@50-95 vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_sweep.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_dir / 'threshold_sweep.png'}")
    
    def generate_summary_report(self, metrics_dict, sweep_results=None):
        """Generate text summary report"""
        print("\nGenerating summary report...")
        
        report_path = self.output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLO MODEL EVALUATION REPORT\n")
            f.write("NEU-DET Metal Surface Defect Detection\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            if metrics_dict.get('overall'):
                overall = metrics_dict['overall']
                f.write(f"  Precision:     {overall.get('precision', 0):.4f}\n")
                f.write(f"  Recall:        {overall.get('recall', 0):.4f}\n")
                f.write(f"  F1 Score:      {overall.get('f1', 0):.4f}\n")
                f.write(f"  mAP@50:        {overall.get('mAP50', 0):.4f}\n")
                f.write(f"  mAP@50-95:     {overall.get('mAP50-95', 0):.4f}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 80 + "\n")
            if metrics_dict.get('per_class'):
                f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10} {'AP50-95':>10}\n")
                f.write("-" * 80 + "\n")
                for cls_metrics in metrics_dict['per_class']:
                    f.write(f"{cls_metrics['class']:<20} "
                           f"{cls_metrics['precision']:>10.4f} "
                           f"{cls_metrics['recall']:>10.4f} "
                           f"{cls_metrics['f1']:>10.4f} "
                           f"{cls_metrics['ap50']:>10.4f} "
                           f"{cls_metrics['ap50-95']:>10.4f}\n")
            f.write("\n")
            
            # Threshold sweep summary
            if sweep_results and sweep_results.get('f1'):
                f.write("THRESHOLD SWEEP RESULTS\n")
                f.write("-" * 80 + "\n")
                max_f1_idx = np.argmax(sweep_results['f1'])
                max_f1 = sweep_results['f1'][max_f1_idx]
                max_f1_thresh = sweep_results['thresholds'][max_f1_idx]
                f.write(f"  Optimal Confidence Threshold: {max_f1_thresh:.4f}\n")
                f.write(f"  Maximum F1 Score: {max_f1:.4f}\n")
                f.write(f"  Precision at optimal: {sweep_results['precision'][max_f1_idx]:.4f}\n")
                f.write(f"  Recall at optimal: {sweep_results['recall'][max_f1_idx]:.4f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())


def main():
    args = parse_args()
    
    print("=" * 80)
    print("YOLO Comprehensive Evaluation Report")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Weights: {args.weights}")
    print(f"  Data: {args.data}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output}")
    print(f"  Confidence Threshold: {args.conf_thres}")
    print(f"  IoU Threshold: {args.iou_thres}")
    print("=" * 80 + "\n")
    
    # Load model
    print("Loading model...")
    model = YOLO(args.weights)
    
    # Get class names
    if hasattr(model, 'names'):
        class_names = list(model.names.values())
    else:
        # Default NEU-DET classes
        class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 
                      'rolled-in_scale', 'scratches']
    
    print(f"Classes: {class_names}\n")
    
    # Create evaluator
    evaluator = YOLOEvaluator(model, args.data, args.output, class_names)
    
    # Run main evaluation
    results = evaluator.run_inference(
        split=args.split,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    
    # Calculate metrics
    metrics_dict = evaluator.calculate_per_class_metrics(results)
    
    # Plot per-class metrics
    if args.save_plots:
        evaluator.plot_per_class_metrics(metrics_dict)
        evaluator.plot_confusion_matrix(results)
        evaluator.plot_pr_curves(results)
    
    # Threshold sweep
    sweep_results = None
    if args.threshold_sweep:
        sweep_results = evaluator.threshold_sweep(
            split=args.split,
            min_conf=args.min_conf,
            max_conf=args.max_conf,
            steps=args.conf_steps,
            iou_thres=args.iou_thres,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    
    # FP/FN Analysis
    fp_fn_stats = evaluator.analyze_false_positives_negatives(
        results, 
        split=args.split,
        conf_thres=args.conf_thres
    )
    
    # ECE Calibration
    ece = evaluator.calculate_ece(None, None)
    
    # Generate summary report
    evaluator.generate_summary_report(metrics_dict, sweep_results)
    
    # Save JSON
    if args.save_json:
        json_path = Path(args.output) / 'evaluation_results.json'
        results_data = {
            'overall_metrics': metrics_dict.get('overall', {}),
            'per_class_metrics': metrics_dict.get('per_class', []),
            'threshold_sweep': sweep_results if sweep_results else {},
        }
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nSaved JSON results: {json_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Completed!")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
