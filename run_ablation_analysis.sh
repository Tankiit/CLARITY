#!/bin/bash
# Script to run ablation analysis on a trained model

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_directory> [ablation_type] [dataset]"
    echo "Example: $0 yelp_models/20250520-084354_distilbert-base-uncased rationale_threshold"
    echo "Example with dataset override: $0 yelp_models/20250520-084354_distilbert-base-uncased all sst2"
    echo "Ablation types: rationale_threshold, concept_count, interactions, skip_connection, all (default)"
    echo "Supported datasets: yelp_polarity, dbpedia, sst2, agnews"
    exit 1
fi

MODEL_DIR=$1
MODEL_PATH="${MODEL_DIR}/checkpoints/best_model.pt"
CONFIG_PATH="${MODEL_DIR}/config.json"

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
elif [[ $MODEL_DIR == *"agnews"* ]]; then
    DATASET="agnews"
else
    DATASET="yelp_polarity"  # Default
fi

# Get ablation type
ABLATION_TYPE=${2:-"all"}

# Allow dataset override from command line
if [ ! -z "$3" ]; then
    DATASET="$3"
    echo "Dataset overridden to: $DATASET"
fi

echo "Running ablation analysis on model from $MODEL_DIR"
echo "Dataset: $DATASET"
echo "Ablation type: $ABLATION_TYPE"

# Run ablation analysis
python ablation_analysis.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --ablation_type "$ABLATION_TYPE" \
    --max_test_samples 500 \
    --output_dir "${MODEL_DIR}/ablation_results"

echo "Ablation analysis complete!"
