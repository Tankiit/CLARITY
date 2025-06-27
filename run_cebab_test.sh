#!/bin/bash
# Run CEBaB test examples through the concept-rationale mapper

# Set up directories
RESULTS_DIR="cebab_results"
mkdir -p "$RESULTS_DIR"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in your PATH"
    exit 1
fi

# Step 1: Extract test examples from CEBaB
echo "Extracting test examples from CEBaB dataset..."
python cebab_test_examples.py

# Check if extraction succeeded
if [ ! -f "test_examples.txt" ]; then
    echo "Error: Failed to extract test examples. Please check for errors above."
    exit 1
fi

# Step 2: Run the concept mapper on the examples
echo "Running concept-rationale mapper on all examples..."
./cebab_map.sh -f test_examples.txt -o "$RESULTS_DIR/all_examples.txt" -h

# Step 3: Run specific aspect examples
run_aspect_examples() {
    ASPECT=$1
    echo "Processing $ASPECT examples..."
    
    # Check if positive examples exist
    if [ -f "test_examples/${ASPECT}_positive.txt" ]; then
        echo "  - Running positive $ASPECT examples..."
        ./cebab_map.sh -f "test_examples/${ASPECT}_positive.txt" -o "$RESULTS_DIR/${ASPECT}_positive.txt" -h
    fi
    
    # Check if negative examples exist
    if [ -f "test_examples/${ASPECT}_negative.txt" ]; then
        echo "  - Running negative $ASPECT examples..."
        ./cebab_map.sh -f "test_examples/${ASPECT}_negative.txt" -o "$RESULTS_DIR/${ASPECT}_negative.txt" -h
    fi
}

# Run for each aspect
echo "Processing aspect-specific examples..."
run_aspect_examples "food"
run_aspect_examples "service" 
run_aspect_examples "ambiance"
run_aspect_examples "noise"

# Step 4: Compare with dynamically discovered concepts
echo "Running with concepts discovered directly from the dataset..."
./cebab_map.sh -f test_examples.txt -o "$RESULTS_DIR/discovered_concepts.txt" -h -d -n 200

echo "All done! Results saved to $RESULTS_DIR directory."
echo "Check the HTML files in $RESULTS_DIR for visualizations of the concept-rationale mappings." 