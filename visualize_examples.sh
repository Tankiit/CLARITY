#!/bin/bash
# Script to visualize examples with their predicted sentiment

# Check if dataset is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset> [num_examples] [output_dir]"
    echo "Example: $0 sst2"
    echo "Example with custom number of examples: $0 sst2 10"
    echo "Example with custom output directory: $0 sst2 4 my_visualizations"
    echo "Supported datasets: sst2, yelp_polarity, agnews, dbpedia"
    exit 1
fi

DATASET=$1
NUM_EXAMPLES=${2:-4}
OUTPUT_DIR=${3:-"visualizations"}

echo "Visualizing examples for $DATASET dataset"
echo "Number of examples: $NUM_EXAMPLES"
echo "Output directory: $OUTPUT_DIR"

# Run visualization
python visualize_examples.py \
    --dataset "$DATASET" \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo "Visualization complete!"
