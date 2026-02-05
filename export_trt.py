"""
export_trt.py - Export YOLO model to TensorRT with FP16 precision
Optimizes the model for NVIDIA GPU inference
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Export YOLO model to TensorRT FP16')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained YOLO weights (.pt file)')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640],
                        help='Image size(s) for export (height width or single size)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size for export (static batching)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device to use for export')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic axes (batch and image size)')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify ONNX model before TensorRT conversion')
    parser.add_argument('--output-dir', type=str, default='weights',
                        help='Output directory for exported model')
    parser.add_argument('--half', action='store_true', default=True,
                        help='Export with FP16 precision (half precision)')
    parser.add_argument('--int8', action='store_true',
                        help='Export with INT8 precision (requires calibration)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output during export')
    
    return parser.parse_args()


def check_requirements():
    """Check if TensorRT is available"""
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("WARNING: TensorRT not found!")
        print("TensorRT export requires:")
        print("  1. NVIDIA GPU with CUDA")
        print("  2. TensorRT installed (pip install tensorrt)")
        print("  3. Matching CUDA version")
        print("\nWill attempt export anyway (may use ONNX fallback)")
        return False


def export_to_tensorrt(model, args):
    """Export model to TensorRT format"""
    
    print("\n" + "=" * 80)
    print("YOLO MODEL EXPORT TO TENSORRT")
    print("=" * 80)
    
    # Parse image size
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    else:
        imgsz = tuple(args.imgsz)
    
    print(f"\nExport Configuration:")
    print(f"  Weights: {args.weights}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Device: {args.device}")
    print(f"  Precision: {'FP16' if args.half else 'FP32'}")
    print(f"  Workspace: {args.workspace} GB")
    print(f"  Dynamic: {args.dynamic}")
    print(f"  Simplify: {args.simplify}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. TensorRT export requires GPU.")
        print("Falling back to ONNX export...")
        export_format = 'onnx'
    else:
        print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
        export_format = 'engine'  # TensorRT engine
    
    print("\nStarting export...")
    print("-" * 80)
    
    try:
        # Export model
        exported_model = model.export(
            format=export_format,
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            int8=args.int8,
            dynamic=args.dynamic,
            simplify=args.simplify,
            workspace=args.workspace,
            verbose=args.verbose,
        )
        
        print("-" * 80)
        print("\n✓ Export successful!")
        print(f"Exported model: {exported_model}")
        
        # Get file size
        if os.path.exists(exported_model):
            file_size = os.path.getsize(exported_model) / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")
        
        return exported_model
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Ensure TensorRT is installed: pip install tensorrt")
        print("  2. Check CUDA version compatibility")
        print("  3. Verify GPU has sufficient memory")
        print("  4. Try without --dynamic flag")
        print("  5. Try with smaller --workspace value")
        
        # Try fallback to ONNX
        if export_format == 'engine':
            print("\nAttempting fallback to ONNX export...")
            try:
                exported_model = model.export(
                    format='onnx',
                    imgsz=imgsz,
                    batch=args.batch,
                    dynamic=args.dynamic,
                    simplify=args.simplify,
                )
                print(f"\n✓ ONNX export successful: {exported_model}")
                print("Note: You can convert ONNX to TensorRT using trtexec:")
                print(f"  trtexec --onnx={exported_model} --saveEngine=model.engine --fp16")
                return exported_model
            except Exception as e2:
                print(f"\n✗ ONNX export also failed: {e2}")
        
        raise


def benchmark_model(model_path, args):
    """Benchmark the exported model"""
    print("\n" + "=" * 80)
    print("BENCHMARKING EXPORTED MODEL")
    print("=" * 80)
    
    try:
        # Load exported model
        model = YOLO(model_path)
        
        # Run benchmark
        print("\nRunning speed benchmark...")
        
        # Parse image size
        if len(args.imgsz) == 1:
            imgsz = args.imgsz[0]
        else:
            imgsz = args.imgsz[0]  # Use first dimension
        
        results = model.val(
            data='coco128.yaml',  # Use small dataset for benchmark
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            verbose=False,
        )
        
        print("\nBenchmark completed!")
        print(f"Speed: {results.speed}")
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        print("This is normal if the dataset is not available.")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("TensorRT Model Export - FP16 Optimization")
    print("Metal Surface Defect Detection")
    print("=" * 80)
    
    # Check requirements
    check_requirements()
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Export
    exported_model = export_to_tensorrt(model, args)
    
    # Optional benchmark
    if exported_model and os.path.exists(exported_model):
        try:
            benchmark_model(exported_model, args)
        except:
            pass
    
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"Original model: {args.weights}")
    print(f"Exported model: {exported_model}")
    print(f"\nUsage example:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{exported_model}')")
    print(f"  results = model('image.jpg')")
    print("\nFor best performance:")
    print("  - Use batch inference when possible")
    print("  - Keep image size consistent with export size")
    print("  - Use CUDA streams for async inference")
    print("=" * 80)


if __name__ == '__main__':
    main()
