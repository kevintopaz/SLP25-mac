#!/bin/bash

# SeedVR2 Batch Video Upscaler
# Process all video files in input folder and save to output folder
#
# Usage: ./run_batch.sh <input_folder> <output_folder> [resolution]
#
# Examples:
#   ./run_batch.sh ./input_videos ./output_videos
#   ./run_batch.sh ./input_videos ./output_videos 2160

# set -e

# # Check arguments
# if [ $# -lt 2 ]; then
#     echo "Usage: $0 <input_folder> <output_folder> [resolution]"
#     echo ""
#     echo "Arguments:"
#     echo "  input_folder   - Folder containing input video files"
#     echo "  output_folder  - Folder to save processed videos"
#     echo "  resolution     - Target resolution (default: 1080)"
#     exit 1
# fi

INPUT_FOLDER="./magnific_batch_test"
OUTPUT_FOLDER="./magnific_batch_test_output"
RESOLUTION="2160"

# Validate input folder
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Change to script directory
cd "$(dirname "$0")"

# Model settings
MODEL_DIR="models"
DIT_MODEL="ema_3b_fp16.safetensors"
VAE_MODEL="ema_vae_fp16.safetensors"

# Video extensions to process
VIDEO_EXTENSIONS="mp4 MP4 mov MOV avi AVI mkv MKV webm WEBM"

# Count total videos
total_count=0
for ext in $VIDEO_EXTENSIONS; do
    count=$(find "$INPUT_FOLDER" -maxdepth 1 -type f -name "*.$ext" 2>/dev/null | wc -l)
    total_count=$((total_count + count))
done

if [ $total_count -eq 0 ]; then
    echo "No video files found in: $INPUT_FOLDER"
    exit 0
fi

echo "========================================"
echo "Batch Video Upscaler"
echo "========================================"
echo "Input folder:  $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Resolution:    $RESOLUTION"
echo "Videos found:  $total_count"
echo "========================================"
echo ""

# Process counter
current=0

# Process each video file
for ext in $VIDEO_EXTENSIONS; do
    for INPUT_FILE in "$INPUT_FOLDER"/*.$ext; do
        # Skip if no matches (glob didn't expand)
        [ -e "$INPUT_FILE" ] || continue
        
        current=$((current + 1))
        
        # Get filename without path
        FILENAME=$(basename "$INPUT_FILE")
        
        # Build output path (same filename in output folder)
        OUTPUT_FILE="$OUTPUT_FOLDER/$FILENAME"
        
        echo ""
        echo "========================================"
        echo "[$current/$total_count] Processing: $FILENAME"
        echo "========================================"
        echo "Input:  $INPUT_FILE"
        echo "Output: $OUTPUT_FILE"
        echo ""
        
        # Skip if output already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping: Output already exists"
            continue
        fi
        
        # Process video
        python inference_cli_latent_chunked.py "$INPUT_FILE" \
            --output "$OUTPUT_FILE" \
            --model_dir "$MODEL_DIR" \
            --dit_model "$DIT_MODEL" \
            --vae_model "$VAE_MODEL" \
            --resolution $RESOLUTION \
            --latent_chunk_size 15 \
            --latent_overlap 7 \
            --color_correction "lab_reinhard" \
            --vae_encode_tiled \
            --vae_decode_tiled \
            --vae_encode_tile_size 1024 \
            --vae_decode_tile_size 768 \
            --vae_encode_tile_overlap 128 \
            --vae_decode_tile_overlap 128 \
            --video_backend "ffmpeg" \
            --usm_weight 0.5 \
            --usm_radius 10 \
            --usm_threshold 10
        
        echo ""
        echo "Completed: $FILENAME"
    done
done

echo ""
echo "========================================"
echo "Batch processing complete!"
echo "Processed $current videos"
echo "========================================"
