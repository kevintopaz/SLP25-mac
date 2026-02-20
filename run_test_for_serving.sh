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

INPUT_FOLDER="./landscape_test"
OUTPUT_FOLDER="./landscape_test_4k"
OUTPUT_WIDTH="3840"
OUTPUT_HEIGHT="2160"

# Validate input folder
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Change to script directory
cd "$(dirname "$0")"


# Count total files
total_count=$(find "$INPUT_FOLDER" -maxdepth 1 -type f 2>/dev/null | wc -l)

if [ $total_count -eq 0 ]; then
    echo "No files found in: $INPUT_FOLDER"
    exit 0
fi

echo "========================================"
echo "Batch Video Upscaler"
echo "========================================"
echo "Input folder:  $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Target:        ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Videos found:  $total_count"
echo "========================================"
echo ""

# Process counter
current=0

# Process each file
for INPUT_FILE in "$INPUT_FOLDER"/*; do
    # Skip if not a regular file
    [ -f "$INPUT_FILE" ] || continue
    
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
    python inference_cli_latent_chunked_for_serving.py "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --output_width $OUTPUT_WIDTH \
        --output_height $OUTPUT_HEIGHT \
        --debug
    
    echo ""
    echo "Completed: $FILENAME"
done

echo ""
echo "========================================"
echo "Batch processing complete!"
echo "Processed $current videos"
echo "========================================"