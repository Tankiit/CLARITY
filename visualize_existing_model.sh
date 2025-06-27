#!/bin/bash
# Script to visualize an existing model with a custom file structure

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [model_filename] [config_filename] [dataset]"
    echo "Example: $0 results/distilbert-base-uncased_sst2_1"
    echo "Example with custom filenames: $0 results/distilbert-base-uncased_sst2_1 sst2_best_model.pt sst2_config.json"
    echo "Example with dataset override: $0 results/distilbert-base-uncased_sst2_1 sst2_best_model.pt sst2_config.json agnews"
    echo "Supported datasets: yelp_polarity, dbpedia, sst2, agnews"
    exit 1
fi

MODEL_DIR=$1
MODEL_FILENAME=${2:-"sst2_best_model.pt"}  # Default model filename
CONFIG_FILENAME=${3:-"sst2_config.json"}   # Default config filename

MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"
CONFIG_PATH="${MODEL_DIR}/${CONFIG_FILENAME}"

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

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
    DATASET="yelp_polarity"  # Default
fi

# Allow dataset override from command line
if [ ! -z "$4" ]; then
    DATASET="$4"
    echo "Dataset overridden to: $DATASET"
fi

echo "Visualizing model from $MODEL_DIR"
echo "Model file: $MODEL_FILENAME"
echo "Config file: $CONFIG_FILENAME"
echo "Dataset: $DATASET"

# Create output directory
OUTPUT_DIR="${MODEL_DIR}/visualizations_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_model.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR"

echo "Visualization complete! Results saved to $OUTPUT_DIR"
