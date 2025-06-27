#!/bin/bash
# Script to analyze concept activations across different attributes in the CEBaB model

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [num_examples] [num_concepts]"
    echo "Example: $0 cebab_models/20250519-151741_distilbert-base-uncased"
    echo "Example with custom number of examples: $0 cebab_models/20250519-151741_distilbert-base-uncased 50"
    echo "Example with custom number of concepts: $0 cebab_models/20250519-151741_distilbert-base-uncased 50 10"
    exit 1
fi

MODEL_DIR=$1
NUM_EXAMPLES=${2:-50}   # Default number of examples is 50
NUM_CONCEPTS=${3:-10}   # Default number of concepts is 10

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

echo "Analyzing concept activations across attributes"
echo "Model directory: $MODEL_DIR"
echo "Number of examples per attribute: $NUM_EXAMPLES"
echo "Number of concepts to analyze: $NUM_CONCEPTS"

# Create output directory
OUTPUT_DIR="${MODEL_DIR}/concept_attributes_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run analysis
python analyze_concept_attributes.py \
    --model_dir "$MODEL_DIR" \
    --num_examples "$NUM_EXAMPLES" \
    --num_concepts "$NUM_CONCEPTS" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis complete! Results saved to $OUTPUT_DIR"
