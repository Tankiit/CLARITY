#!/bin/bash
# Script to analyze the relationship between rationales and concepts in the CEBaB model

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [attribute] [num_examples]"
    echo "Example: $0 cebab_models/20250519-151741_distilbert-base-uncased"
    echo "Example with attribute: $0 cebab_models/20250519-151741_distilbert-base-uncased food"
    echo "Example with custom number of examples: $0 cebab_models/20250519-151741_distilbert-base-uncased food 50"
    echo "Supported attributes: food, ambiance, service, noise, price"
    exit 1
fi

MODEL_DIR=$1
ATTRIBUTE=${2:-"food"}  # Default attribute is food
NUM_EXAMPLES=${3:-50}   # Default number of examples is 50

# Check if model file exists
MODEL_PATH="${MODEL_DIR}/checkpoints/best_model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Will continue with randomly initialized weights for demonstration"
fi

# Check if config file exists
CONFIG_PATH="${MODEL_DIR}/config.json"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file not found at $CONFIG_PATH"
    echo "Will use default configuration"
fi

echo "Analyzing rationale-concept relationship for $ATTRIBUTE attribute"
echo "Model directory: $MODEL_DIR"
echo "Number of examples: $NUM_EXAMPLES"

# Create output directory
OUTPUT_DIR="${MODEL_DIR}/concept_analysis_$(date +%Y%m%d-%H%M%S)_${ATTRIBUTE}"
mkdir -p "$OUTPUT_DIR"

# Run analysis
python analyze_rationale_concept_relationship.py \
    --model_dir "$MODEL_DIR" \
    --attribute "$ATTRIBUTE" \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis complete! Results saved to $OUTPUT_DIR"
