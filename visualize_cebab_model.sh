#!/bin/bash
# Script to visualize CEBaB model predictions

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [attribute] [model_filename] [config_filename]"
    echo "Example: $0 results/distilbert-base-uncased_cebab_1"
    echo "Example with attribute: $0 results/distilbert-base-uncased_cebab_1 food"
    echo "Example with custom filenames: $0 results/distilbert-base-uncased_cebab_1 food cebab_best_model.pt cebab_config.json"
    echo "Supported attributes: food, ambiance, service, noise, price"
    exit 1
fi

MODEL_DIR=$1
ATTRIBUTE=${2:-"food"}  # Default attribute is food
MODEL_FILENAME=${3:-"checkpoints/best_model.pt"}  # Default model filename
CONFIG_FILENAME=${4:-"config.json"}   # Default config filename

# Check if model file exists
MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Will continue with randomly initialized weights for demonstration"
fi

# Check if config file exists
CONFIG_PATH="${MODEL_DIR}/${CONFIG_FILENAME}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file not found at $CONFIG_PATH"
    echo "Will use default configuration"
fi

echo "Visualizing CEBaB model from $MODEL_DIR"
echo "Attribute: $ATTRIBUTE"
echo "Model file: $MODEL_FILENAME"
echo "Config file: $CONFIG_FILENAME"

# Create output directory
OUTPUT_DIR="${MODEL_DIR}/visualizations_$(date +%Y%m%d-%H%M%S)_${ATTRIBUTE}"
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_cebab_model.py \
    --model_dir "$MODEL_DIR" \
    --model_filename "$MODEL_FILENAME" \
    --config_filename "$CONFIG_FILENAME" \
    --attribute "$ATTRIBUTE" \
    --output_dir "$OUTPUT_DIR"

echo "Visualization complete! Results saved to $OUTPUT_DIR"
