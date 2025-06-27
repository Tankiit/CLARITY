#!/bin/bash
# Script to visualize the SST-2 model

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [dataset] [num_examples]"
    echo "Example: $0 results/distilbert-base-uncased_sst2_1"
    echo "Example with dataset override: $0 results/distilbert-base-uncased_sst2_1 agnews"
    echo "Example with custom number of examples: $0 results/distilbert-base-uncased_sst2_1 sst2 10"
    echo "Supported datasets: sst2, yelp_polarity, agnews, dbpedia"
    exit 1
fi

MODEL_DIR=$1

# Determine dataset from directory name
if [[ $MODEL_DIR == *"yelp"* ]]; then
    DATASET="yelp_polarity"
elif [[ $MODEL_DIR == *"dbpedia"* ]]; then
    DATASET="dbpedia"
elif [[ $MODEL_DIR == *"sst2"* ]]; then
    DATASET="sst2"
elif [[ $MODEL_DIR == *"agnews"* || $MODEL_DIR == *"ag_news"* ]]; then
    DATASET="agnews"
else
    DATASET="sst2"  # Default
fi

# Allow dataset override from command line
if [ ! -z "$2" ]; then
    DATASET="$2"
    echo "Dataset overridden to: $DATASET"
fi

# Number of examples
NUM_EXAMPLES=${3:-4}

echo "Visualizing SST-2 model from $MODEL_DIR"
echo "Dataset: $DATASET"
echo "Number of examples: $NUM_EXAMPLES"

# Create output directory
OUTPUT_DIR="${MODEL_DIR}/visualizations_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_sst2_model.py \
    --model_dir "$MODEL_DIR" \
    --dataset "$DATASET" \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo "Visualization complete! Results saved to $OUTPUT_DIR"
