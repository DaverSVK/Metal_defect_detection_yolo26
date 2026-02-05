"""
TensorRT Export Script for YOLOv8 Metal Defect Detection
Export trained YOLOv8 model to TensorRT format with FP16 precision for optimized inference.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import os


def export_to_tensorrt(
    model_path,
    output_path=None,
    imgsz=640,
    half=True,
    dynamic=False,
    workspace=4,
    device='0'
):
    """
    Export YOLO model to TensorRT format.
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        output_path: Path to save exported model (optional)
        imgsz: Input image size (default: 640)
        half: Use FP16 precision (default: True)
        dynamic: Enable dynamic input shapes (default: False)
        workspace: Maximum workspace size in GB (default: 4)
        device: CUDA device to use (default: '0')
    
    Returns:
        Path to exported model
    """
    print("=" * 60)
    print("YOLOv8 TensorRT Export")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}")
    print(f"Precision: {'FP16' if half else 'FP32'}")
    print(f"Dynamic shapes: {dynamic}")
    print(f"Workspace: {workspace}GB")
    print(f"Device: cuda:{device}")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT export requires CUDA.")
    
    print(f"\nCUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = YOLO(model_path)
    
    # Verify model
    if not hasattr(model, 'export'):
        raise ValueError("Invalid model file. Please provide a valid YOLO model.")
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    
    # Export parameters
    export_args = {
        'format': 'engine',  # TensorRT format
        'imgsz': imgsz,
        'half': half,
        'dynamic': dynamic,
        'workspace': workspace,
        'device': device,
        'verbose': True,
        'simplify': True,
        'int8': False,  # Use FP16 instead of INT8
    }
    
    print("\nExport Configuration:")
    print("-" * 60)
    for key, value in export_args.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    # Export model
    print("\nExporting to TensorRT...")
    print("This may take several minutes...")
    
    try:
        exported_path = model.export(**export_args)
        
        print("\n" + "=" * 60)
        print("Export Successful!")
        print("=" * 60)
        print(f"TensorRT engine saved to: {exported_path}")
        
        # Move to custom output path if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            exported_path = Path(exported_path)
            if exported_path != output_path:
                import shutil
                shutil.move(str(exported_path), str(output_path))
                print(f"Moved to: {output_path}")
                exported_path = output_path
        
        # Print file size
        file_size = Path(exported_path).stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        
        # Print usage instructions
        print("\n" + "=" * 60)
        print("Usage:")
        print("=" * 60)
        print("To use the exported TensorRT model for inference:")
        print(f"\n  from ultralytics import YOLO")
        print(f"  model = YOLO('{exported_path}')")
        print(f"  results = model('image.jpg')")
        print("\nFor video inference:")
        print(f"  python video_infer.py --weights {exported_path} --source video.mp4")
        print("=" * 60)
        
        return exported_path
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("Export Failed!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure TensorRT is properly installed")
        print("2. Check CUDA and cuDNN versions are compatible")
        print("3. Verify you have sufficient GPU memory")
        print("4. Try with --workspace parameter to reduce memory usage")
        print("=" * 60)
        raise


def verify_tensorrt_model(engine_path, test_image=None, imgsz=640):
    """
    Verify exported TensorRT model.
    
    Args:
        engine_path: Path to TensorRT engine file
        test_image: Optional test image path
        imgsz: Image size for inference
    """
    print("\n" + "=" * 60)
    print("Verifying TensorRT Model")
    print("=" * 60)
    
    try:
        # Load TensorRT model
        print(f"Loading TensorRT model from {engine_path}...")
        model = YOLO(engine_path)
        
        print("Model loaded successfully!")
        
        # Run test inference if image provided
        if test_image:
            print(f"\nRunning test inference on {test_image}...")
            import time
            
            # Warmup
            print("Warming up...")
            for _ in range(5):
                model(test_image, imgsz=imgsz, verbose=False)
            
            # Benchmark
            print("Benchmarking...")
            times = []
            for _ in range(20):
                start = time.time()
                results = model(test_image, imgsz=imgsz, verbose=False)
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            print(f"\nInference Performance:")
            print(f"  Average time: {avg_time*1000:.2f}ms")
            print(f"  FPS: {fps:.2f}")
            
            # Print detections
            if len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                print(f"  Detections: {num_detections}")
        
        print("\n" + "=" * 60)
        print("Verification Successful!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("Verification Failed!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export YOLOv8 Model to TensorRT with FP16 Precision'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save exported TensorRT engine (optional)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size for the model (default: 640)'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 precision instead of FP16'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic input shapes (allows variable image sizes)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device to use (default: 0)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify exported model after export'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        default=None,
        help='Test image for verification (used with --verify)'
    )
    
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    
    # Check if model file exists
    model_path = Path(args.weights)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.weights}")
    
    # Export to TensorRT
    try:
        exported_path = export_to_tensorrt(
            model_path=args.weights,
            output_path=args.output,
            imgsz=args.imgsz,
            half=not args.fp32,
            dynamic=args.dynamic,
            workspace=args.workspace,
            device=args.device
        )
        
        # Verify if requested
        if args.verify:
            verify_tensorrt_model(
                engine_path=exported_path,
                test_image=args.test_image,
                imgsz=args.imgsz
            )
    
    except Exception as e:
        print(f"\nError during export: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
