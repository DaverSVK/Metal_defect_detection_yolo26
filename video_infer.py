"""
Video Inference with Rolling OK/NOK Status for Metal Defect Detection
This script processes video frames and displays real-time defect detection
with a rolling OK/NOK status based on recent detections.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time


class VideoDefectDetector:
    """Real-time video defect detector with rolling OK/NOK status."""
    
    def __init__(self, model_path, conf_thresh=0.25, window_size=30):
        """
        Initialize video detector.
        
        Args:
            model_path: Path to trained YOLO model
            conf_thresh: Confidence threshold for detections
            window_size: Number of frames for rolling status (default: 30 frames)
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.window_size = window_size
        self.detection_history = deque(maxlen=window_size)
        
        # Color scheme
        self.colors = {
            0: (0, 255, 255),    # crazing - yellow
            1: (255, 0, 255),    # inclusion - magenta
            2: (0, 165, 255),    # patches - orange
            3: (0, 0, 255),      # pitted_surface - red
            4: (255, 0, 0),      # rolled_in_scale - blue
            5: (0, 255, 0)       # scratches - green
        }
        
        self.class_names = [
            'crazing', 'inclusion', 'patches',
            'pitted_surface', 'rolled_in_scale', 'scratches'
        ]
    
    def get_rolling_status(self):
        """
        Calculate rolling OK/NOK status based on recent detections.
        
        Returns:
            tuple: (status, defect_rate, total_detections)
        """
        if len(self.detection_history) == 0:
            return "OK", 0.0, 0
        
        total_detections = sum(self.detection_history)
        defect_rate = total_detections / len(self.detection_history)
        
        # Status logic: NOK if more than 20% of recent frames have defects
        status = "NOK" if defect_rate > 0.2 else "OK"
        
        return status, defect_rate, total_detections
    
    def draw_detections(self, frame, results):
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        has_detections = False
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0])
                
                if conf >= self.conf_thresh:
                    has_detections = True
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    
                    # Draw bounding box
                    color = self.colors.get(cls, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"{self.class_names[cls]}: {conf:.2f}"
                    label_size, baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
        
        return annotated_frame, has_detections
    
    def draw_status_panel(self, frame, status, defect_rate, total_detections, fps):
        """
        Draw status panel on frame.
        
        Args:
            frame: Input frame
            status: Current OK/NOK status
            defect_rate: Defect detection rate
            total_detections: Total detections in window
            fps: Current FPS
            
        Returns:
            Frame with status panel
        """
        h, w = frame.shape[:2]
        panel_height = 120
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Determine status color
        status_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        
        # Draw status
        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_BOLD,
            1.2,
            status_color,
            3
        )
        
        # Draw metrics
        metrics_y = 70
        cv2.putText(
            frame,
            f"Defect Rate: {defect_rate*100:.1f}% ({total_detections}/{self.window_size} frames)",
            (20, metrics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, metrics_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw rolling window indicator
        bar_width = w - 40
        bar_x = 20
        bar_y = panel_height - 20
        bar_height = 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Fill bar based on defect rate
        fill_width = int(bar_width * defect_rate)
        if fill_width > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                status_color,
                -1
            )
        
        # Draw threshold line
        threshold_x = bar_x + int(bar_width * 0.2)
        cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 0), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, display=True, save_stats=True):
        """
        Process video and apply defect detection with rolling OK/NOK status.
        
        Args:
            video_path: Path to input video file or camera index (0 for webcam)
            output_path: Path to save output video (optional)
            display: Whether to display video in real-time
            save_stats: Whether to save detection statistics
        """
        # Open video
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            cap = cv2.VideoCapture(int(video_path))
            video_name = f"camera_{video_path}"
        else:
            cap = cv2.VideoCapture(str(video_path))
            video_name = Path(video_path).stem
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Window size: {self.window_size} frames")
        print("-" * 60)
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Statistics
        frame_count = 0
        total_detections = 0
        status_history = []
        
        # FPS calculation
        fps_history = deque(maxlen=30)
        
        print("Press 'q' to quit, 'p' to pause/resume")
        print("-" * 60)
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    start_time = time.time()
                    
                    # Run detection
                    results = self.model(frame, conf=self.conf_thresh, verbose=False)
                    
                    # Draw detections
                    annotated_frame, has_detections = self.draw_detections(frame, results)
                    
                    # Update detection history
                    self.detection_history.append(1 if has_detections else 0)
                    if has_detections:
                        total_detections += 1
                    
                    # Get rolling status
                    status, defect_rate, window_detections = self.get_rolling_status()
                    status_history.append(status)
                    
                    # Calculate FPS
                    process_time = time.time() - start_time
                    current_fps = 1.0 / process_time if process_time > 0 else 0
                    fps_history.append(current_fps)
                    avg_fps = np.mean(fps_history)
                    
                    # Draw status panel
                    annotated_frame = self.draw_status_panel(
                        annotated_frame, status, defect_rate, window_detections, avg_fps
                    )
                    
                    # Save frame
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Display
                    if display:
                        cv2.imshow('Metal Defect Detection - Rolling OK/NOK', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"Frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Detection rate: {total_detections/frame_count*100:.2f}%")
        
        ok_count = status_history.count("OK")
        nok_count = status_history.count("NOK")
        print(f"OK frames: {ok_count} ({ok_count/len(status_history)*100:.2f}%)")
        print(f"NOK frames: {nok_count} ({nok_count/len(status_history)*100:.2f}%)")
        
        if output_path:
            print(f"\nOutput saved to: {output_path}")
        
        # Save statistics
        if save_stats and output_path:
            stats_path = Path(output_path).parent / f"{Path(output_path).stem}_stats.json"
            import json
            
            stats = {
                'video_name': video_name,
                'total_frames': frame_count,
                'total_detections': total_detections,
                'detection_rate': total_detections / frame_count if frame_count > 0 else 0,
                'ok_frames': ok_count,
                'nok_frames': nok_count,
                'ok_percentage': ok_count / len(status_history) * 100 if status_history else 0,
                'nok_percentage': nok_count / len(status_history) * 100 if status_history else 0,
                'window_size': self.window_size,
                'confidence_threshold': self.conf_thresh
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Statistics saved to: {stats_path}")
        
        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video Inference with Rolling OK/NOK Status for Metal Defect Detection'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Video file path or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=30,
        help='Rolling window size in frames for OK/NOK status'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display (useful for batch processing)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable saving statistics file'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv8 Video Inference - Rolling OK/NOK Status")
    print("=" * 60)
    print(f"Model: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"Window size: {args.window} frames")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 60)
    
    # Initialize detector
    detector = VideoDefectDetector(
        model_path=args.weights,
        conf_thresh=args.conf,
        window_size=args.window
    )
    
    # Process video
    detector.process_video(
        video_path=args.source,
        output_path=args.output,
        display=not args.no_display,
        save_stats=not args.no_stats
    )


if __name__ == '__main__':
    main()
