#!/bin/bash
# end_to_end_example.sh - Complete workflow example
# This script demonstrates the full pipeline from training to deployment

set -e  # Exit on error

echo "================================================================================"
echo "YOLO Metal Defect Detection - Complete Pipeline Example"
echo "================================================================================"

# Configuration
DATA_YAML="data.yaml"
MODEL="yolov8n.pt"
EPOCHS=100
BATCH=16
DEVICE="0"

echo -e "\n📋 Step 1: Verify Setup"
echo "--------------------------------------------------------------------------------"
python verify_setup.py

echo -e "\n📊 Step 2: Prepare Dataset (if needed)"
echo "--------------------------------------------------------------------------------"
echo "Skipping dataset preparation - assuming dataset is already prepared"
echo "To prepare dataset, run:"
echo "  python prepare_dataset.py --source /path/to/NEU-DET --output datasets/NEU-DET"

echo -e "\n🚀 Step 3: Training"
echo "--------------------------------------------------------------------------------"
echo "Training YOLOv8 on NEU-DET dataset..."
python train.py \
    --data "$DATA_YAML" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --patience 50 \
    --cache

# Get the latest training run directory
TRAIN_DIR=$(ls -td runs/train/neu-det* | head -1)
WEIGHTS="${TRAIN_DIR}/weights/best.pt"

echo -e "\n✓ Training completed! Best model: $WEIGHTS"

echo -e "\n📈 Step 4: Comprehensive Evaluation"
echo "--------------------------------------------------------------------------------"
echo "Running evaluation with advanced metrics..."
python eval_report.py \
    --weights "$WEIGHTS" \
    --data "$DATA_YAML" \
    --output eval_results \
    --save-plots \
    --threshold-sweep \
    --save-json

echo -e "\n✓ Evaluation completed! Results in eval_results/"

echo -e "\n🎥 Step 5: Video Inference (if video available)"
echo "--------------------------------------------------------------------------------"
if [ -f "test_video.mp4" ]; then
    echo "Processing test video..."
    python video_infer.py \
        --weights "$WEIGHTS" \
        --video test_video.mp4 \
        --output output_videos \
        --save-video
    echo -e "\n✓ Video processing completed!"
else
    echo "No test video found (test_video.mp4)"
    echo "To process a video, run:"
    echo "  python video_infer.py --weights $WEIGHTS --video your_video.mp4"
fi

echo -e "\n⚡ Step 6: TensorRT Export (optional)"
echo "--------------------------------------------------------------------------------"
echo "Exporting to TensorRT FP16..."
if command -v nvidia-smi &> /dev/null; then
    python export_trt.py \
        --weights "$WEIGHTS" \
        --imgsz 640 \
        --half \
        --device "$DEVICE"
    echo -e "\n✓ TensorRT export completed!"
else
    echo "NVIDIA GPU not detected. Skipping TensorRT export."
    echo "To export manually, run:"
    echo "  python export_trt.py --weights $WEIGHTS --half"
fi

echo -e "\n================================================================================"
echo "🎉 Pipeline Complete!"
echo "================================================================================"
echo -e "\nGenerated files:"
echo "  Training:   $TRAIN_DIR"
echo "  Best model: $WEIGHTS"
echo "  Evaluation: eval_results/"
echo "  Videos:     output_videos/ (if processed)"
echo ""
echo "Next steps:"
echo "  1. Review evaluation metrics in eval_results/evaluation_report.txt"
echo "  2. Check confusion matrix: eval_results/confusion_matrix.png"
echo "  3. Analyze threshold sweep: eval_results/threshold_sweep.png"
echo "  4. Deploy model for production use"
echo "================================================================================"
