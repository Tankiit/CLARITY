#!/bin/bash
# Script to run the CEBaB concept-to-rationale mapper

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in your PATH"
    exit 1
fi

# Function to show help
show_help() {
    echo "Usage: $0 [options] TEXT"
    echo "Map CEBaB concepts to rationales in text explanations"
    echo
    echo "Options:"
    echo "  -f FILE         Process text from a file (one per line)"
    echo "  -o FILE         Output file (default: prints to console)"
    echo "  -h              Generate HTML visualization"
    echo "  -d              Use concepts directly from the CEBaB dataset"
    echo "  -n NUMBER       Number of CEBaB samples to use (default: 100)"
    echo "  -c FILE         Path to concept word map JSON file"
    echo "  --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0 \"The food was amazing but the service was slow.\""
    echo "  $0 -f examples.txt -o cebab_explanations.txt -h"
    echo "  $0 -d -n 500 \"The restaurant had a wonderful ambiance.\""
}

# Default values
INPUT_FILE=""
OUTPUT_FILE=""
HTML_FLAG=""
USE_DATASET_FLAG=""
CEBAB_SAMPLES="100"
CONCEPT_MAP=""
TEXT=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h)
            HTML_FLAG="--html"
            shift
            ;;
        -d)
            USE_DATASET_FLAG="--use_dataset_concepts"
            shift
            ;;
        -n)
            CEBAB_SAMPLES="$2"
            shift 2
            ;;
        -c)
            CONCEPT_MAP="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            # If no flags are left, treat as text input
            TEXT="$*"
            break
            ;;
    esac
done

# Check that we have at least one input method
if [ -z "$TEXT" ] && [ -z "$INPUT_FILE" ]; then
    echo "Error: Please provide either text or an input file (-f)"
    show_help
    exit 1
fi

# Build the command
CMD="python cebab_mapper.py --pretty"

# Add input options
if [ ! -z "$INPUT_FILE" ]; then
    # Check if file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: File '$INPUT_FILE' not found"
        exit 1
    fi
    CMD="$CMD --input_file \"$INPUT_FILE\""
else
    CMD="$CMD --text \"$TEXT\""
fi

# Add output options
if [ ! -z "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi

# Add HTML flag if specified
if [ ! -z "$HTML_FLAG" ]; then
    CMD="$CMD $HTML_FLAG"
fi

# Add dataset options
if [ ! -z "$USE_DATASET_FLAG" ]; then
    CMD="$CMD $USE_DATASET_FLAG --cebab_samples $CEBAB_SAMPLES"
fi

# Add concept map if specified
if [ ! -z "$CONCEPT_MAP" ]; then
    CMD="$CMD --concept_map \"$CONCEPT_MAP\""
fi

# Print the command being run
echo "Running: $CMD"

# Execute the command
eval $CMD 