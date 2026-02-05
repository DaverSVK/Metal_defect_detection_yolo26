"""
video_infer.py - Video inference with rolling OK/NOK detection
Processes video files and displays real-time defect detection with rolling statistics
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Video inference with rolling OK/NOK stats')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained YOLO weights')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_videos',
                        help='Output directory for processed video')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, cpu, etc.)')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Rolling window size (frames)')
    parser.add_argument('--defect-threshold', type=int, default=1,
                        help='Minimum detections to mark frame as NOK')
    parser.add_argument('--display', action='store_true', default=True,
                        help='Display video while processing')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save processed video')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output video FPS (default: same as input)')
    parser.add_argument('--show-boxes', action='store_true', default=True,
                        help='Draw bounding boxes on detections')
    parser.add_argument('--show-labels', action='store_true', default=True,
                        help='Show class labels on detections')
    
    return parser.parse_args()


class RollingStats:
    """Track rolling statistics over a window of frames"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.detections_window = deque(maxlen=window_size)
        self.ok_nok_window = deque(maxlen=window_size)
        self.class_counts = {}
        
    def update(self, num_detections, is_ok, class_detections=None):
        """Update rolling statistics"""
        self.detections_window.append(num_detections)
        self.ok_nok_window.append(1 if is_ok else 0)
        
        if class_detections:
            for cls, count in class_detections.items():
                if cls not in self.class_counts:
                    self.class_counts[cls] = deque(maxlen=self.window_size)
                self.class_counts[cls].append(count)
    
    def get_stats(self):
        """Get current rolling statistics"""
        if not self.detections_window:
            return {
                'avg_detections': 0,
                'ok_rate': 0,
                'nok_rate': 0,
                'total_frames': 0,
            }
        
        total_frames = len(self.ok_nok_window)
        ok_frames = sum(self.ok_nok_window)
        
        return {
            'avg_detections': np.mean(self.detections_window),
            'ok_rate': ok_frames / total_frames if total_frames > 0 else 0,
            'nok_rate': (total_frames - ok_frames) / total_frames if total_frames > 0 else 0,
            'total_frames': total_frames,
            'ok_frames': ok_frames,
            'nok_frames': total_frames - ok_frames,
        }


def draw_stats_overlay(frame, stats, current_status, frame_num, total_frames, 
                       detections_count, processing_fps):
    """Draw statistics overlay on frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Semi-transparent panel
    panel_height = 220
    cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "DEFECT DETECTION STATUS", (20, 35),
                cv2.FONT_HERSHEY_BOLD, 0.6, (255, 255, 255), 2)
    
    # Frame info
    y_offset = 60
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    cv2.putText(frame, f"FPS: {processing_fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Current status
    y_offset += 30
    status_text = "OK" if current_status else "NOK - DEFECT DETECTED"
    status_color = (0, 255, 0) if current_status else (0, 0, 255)
    cv2.putText(frame, f"Status: {status_text}", (20, y_offset),
                cv2.FONT_HERSHEY_BOLD, 0.6, status_color, 2)
    
    y_offset += 25
    cv2.putText(frame, f"Detections: {detections_count}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Rolling window stats
    y_offset += 30
    cv2.putText(frame, f"Rolling Stats (last {stats['total_frames']} frames):", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    y_offset += 25
    cv2.putText(frame, f"  OK: {stats['ok_frames']} ({stats['ok_rate']*100:.1f}%)", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    y_offset += 20
    cv2.putText(frame, f"  NOK: {stats['nok_frames']} ({stats['nok_rate']*100:.1f}%)", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    y_offset += 20
    cv2.putText(frame, f"  Avg Defects/Frame: {stats['avg_detections']:.2f}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Status indicator (large)
    indicator_size = 40
    indicator_x = w - indicator_size - 20
    indicator_y = 20
    cv2.circle(frame, (indicator_x, indicator_y), indicator_size//2, status_color, -1)
    cv2.circle(frame, (indicator_x, indicator_y), indicator_size//2, (255, 255, 255), 2)
    
    return frame


def draw_detections(frame, results, show_boxes=True, show_labels=True):
    """Draw detection boxes and labels on frame"""
    if results is None or len(results) == 0:
        return frame
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return frame
    
    boxes = result.boxes
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        
        # Get class name
        class_name = result.names[cls] if hasattr(result, 'names') else f"Class {cls}"
        
        # Draw box
        if show_boxes:
            color = (0, 0, 255)  # Red for defects
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        if show_labels:
            label = f"{class_name} {conf:.2f}"
            
            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(int(y1) - 10, label_size[1] + 10)
            
            cv2.rectangle(frame, 
                         (int(x1), label_y - label_size[1] - 5),
                         (int(x1) + label_size[0] + 5, label_y + 5),
                         (0, 0, 255), -1)
            
            cv2.putText(frame, label, (int(x1) + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def process_video(model, video_path, output_dir, args):
    """Process video with YOLO detection and rolling stats"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_fps = args.fps if args.fps else fps
    
    print(f"\nVideo Info:")
    print(f"  Total Frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Output FPS: {output_fps:.2f}")
    
    # Initialize video writer
    video_writer = None
    if args.save_video:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = output_dir / f"{video_name}_processed.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, 
                                       (width, height))
        print(f"  Output: {output_path}")
    
    # Initialize rolling stats
    rolling_stats = RollingStats(window_size=args.window_size)
    
    # Processing loop
    frame_num = 0
    total_detections = 0
    total_ok = 0
    total_nok = 0
    
    print("\nProcessing video...")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    start_time = time.time()
    processing_times = deque(maxlen=30)
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            frame_start = time.time()
            
            # Run inference
            results = model(frame, 
                          conf=args.conf_thres,
                          iou=args.iou_thres,
                          imgsz=args.imgsz,
                          device=args.device,
                          verbose=False)
            
            # Count detections
            num_detections = 0
            class_detections = {}
            
            if results and len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                
                # Count per class
                for box in results[0].boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = results[0].names[cls] if hasattr(results[0], 'names') else str(cls)
                    class_detections[class_name] = class_detections.get(class_name, 0) + 1
            
            # Determine OK/NOK status
            is_ok = num_detections < args.defect_threshold
            
            # Update statistics
            total_detections += num_detections
            if is_ok:
                total_ok += 1
            else:
                total_nok += 1
            
            rolling_stats.update(num_detections, is_ok, class_detections)
            current_stats = rolling_stats.get_stats()
            
            # Draw detections
            if args.show_boxes or args.show_labels:
                frame = draw_detections(frame, results, args.show_boxes, args.show_labels)
            
            # Calculate processing FPS
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            processing_fps = 1.0 / np.mean(processing_times) if processing_times else 0
            
            # Draw stats overlay
            frame = draw_stats_overlay(frame, current_stats, is_ok, frame_num, 
                                      total_frames, num_detections, processing_fps)
            
            # Write frame
            if video_writer:
                video_writer.write(frame)
            
            # Display
            if args.display:
                cv2.imshow('Defect Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Paused. Press 'p' to resume.")
            else:
                print("Resumed.")
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed: {frame_num}")
    print(f"Total Processing Time: {elapsed_time:.2f}s")
    print(f"Average FPS: {frame_num / elapsed_time:.2f}")
    print(f"\nDetection Statistics:")
    print(f"  Total Detections: {total_detections}")
    print(f"  Average Detections/Frame: {total_detections/frame_num:.2f}")
    print(f"\nQuality Assessment:")
    print(f"  OK Frames: {total_ok} ({total_ok/frame_num*100:.1f}%)")
    print(f"  NOK Frames (Defects): {total_nok} ({total_nok/frame_num*100:.1f}%)")
    
    if args.save_video:
        print(f"\nProcessed video saved to: {output_path}")
    
    print("=" * 80)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("YOLO Video Inference with Rolling OK/NOK Detection")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Weights: {args.weights}")
    print(f"  Video: {args.video}")
    print(f"  Output: {args.output}")
    print(f"  Confidence Threshold: {args.conf_thres}")
    print(f"  IoU Threshold: {args.iou_thres}")
    print(f"  Rolling Window: {args.window_size} frames")
    print(f"  Defect Threshold: {args.defect_threshold}")
    print("=" * 80)
    
    # Check video exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Load model
    print("\nLoading model...")
    model = YOLO(args.weights)
    print(f"Model loaded: {args.weights}")
    
    # Process video
    process_video(model, args.video, args.output, args)
    
    print("\nProcessing completed!")


if __name__ == '__main__':
    main()
