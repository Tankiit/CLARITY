#!/usr/bin/env python
"""
Extract test examples from CEBaB dataset to demonstrate concept-rationale mapping.
This script creates a test.json file with diverse examples that we can use with our mapper.
"""
import json
import random
from collections import defaultdict
import os

try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False
    print("Error: 'datasets' library not installed. Run: pip install datasets")
    exit(1)

def load_cebab_test_set():
    """Load the CEBaB test set"""
    print("Loading CEBaB test set...")
    try:
        # Load just the test split
        ds = load_dataset("CEBaB/CEBaB", split="test")
        print(f"Successfully loaded {len(ds)} test samples")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_diverse_examples(dataset, num_examples=20):
    """
    Extract a diverse set of examples from the dataset.
    We want to include examples with different aspects and sentiments.
    """
    # Track examples by aspect and sentiment
    aspect_sentiment_examples = defaultdict(list)
    
    # Define all possible combinations
    aspects = ["food", "service", "ambiance", "noise"]
    sentiments = ["Positive", "Negative"]
    
    # Initialize with empty lists
    for aspect in aspects:
        for sentiment in sentiments:
            aspect_sentiment_examples[f"{aspect}_{sentiment}"] = []
    
    # Categorize examples
    for sample in dataset:
        # For each sample, check which aspects are mentioned
        for aspect in aspects:
            sentiment = sample[f"{aspect}_aspect_majority"]
            
            # If there's a clear sentiment, categorize it
            if sentiment in sentiments:
                # Create a simpler example dict with just what we need
                example = {
                    "id": sample["id"],
                    "text": sample["description"],
                    "aspects": {},
                    "rating": sample["review_majority"]
                }
                
                # Add all aspect sentiments
                for a in aspects:
                    example["aspects"][a] = sample[f"{a}_aspect_majority"]
                
                # Add to the appropriate category
                aspect_sentiment_examples[f"{aspect}_{sentiment}"].append(example)
    
    # Select a diverse set of examples
    diverse_examples = []
    
    # Try to get examples from each category
    for key, examples in aspect_sentiment_examples.items():
        # If we have examples in this category, select some
        if examples:
            # Take up to 3 examples from each category, but ensure diversity
            samples = random.sample(examples, min(3, len(examples)))
            diverse_examples.extend(samples)
    
    # If we have too many, randomly sample to get the desired number
    if len(diverse_examples) > num_examples:
        diverse_examples = random.sample(diverse_examples, num_examples)
    
    print(f"Selected {len(diverse_examples)} diverse examples from the test set")
    
    # Print statistics
    print("\nExample distribution:")
    for aspect in aspects:
        pos_count = sum(1 for ex in diverse_examples if ex["aspects"][aspect] == "Positive")
        neg_count = sum(1 for ex in diverse_examples if ex["aspects"][aspect] == "Negative")
        print(f"  {aspect}: {pos_count} positive, {neg_count} negative")
    
    return diverse_examples

def save_examples(examples, output_file="test_examples.json"):
    """Save examples to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved {len(examples)} examples to {output_file}")
    
    # Also create a simple text file for easy processing
    with open("test_examples.txt", 'w') as f:
        for example in examples:
            f.write(f"{example['text']}\n")
    print(f"Also saved plain text to test_examples.txt")

def create_aspect_specific_files(examples):
    """Create separate files for examples with specific aspects"""
    aspects = ["food", "service", "ambiance", "noise"]
    
    # Create directories
    os.makedirs("test_examples", exist_ok=True)
    
    # Create files for each aspect
    for aspect in aspects:
        # Extract positive examples
        positive_examples = [ex for ex in examples if ex["aspects"][aspect] == "Positive"]
        # Extract negative examples
        negative_examples = [ex for ex in examples if ex["aspects"][aspect] == "Negative"]
        
        # Save positive examples
        if positive_examples:
            with open(f"test_examples/{aspect}_positive.txt", 'w') as f:
                for example in positive_examples:
                    f.write(f"{example['text']}\n")
            print(f"Saved {len(positive_examples)} positive {aspect} examples")
        
        # Save negative examples
        if negative_examples:
            with open(f"test_examples/{aspect}_negative.txt", 'w') as f:
                for example in negative_examples:
                    f.write(f"{example['text']}\n")
            print(f"Saved {len(negative_examples)} negative {aspect} examples")
    
    print("\nAspect-specific example files saved to the test_examples directory")

def print_example_previews(examples, num_to_show=5):
    """Print previews of some examples"""
    print("\nExample previews:")
    for i, example in enumerate(examples[:num_to_show]):
        print(f"\nExample {i+1}:")
        print(f"Text: {example['text'][:100]}..." if len(example['text']) > 100 else example['text'])
        print(f"Aspects: ", end="")
        for aspect, sentiment in example["aspects"].items():
            if sentiment in ["Positive", "Negative"]:
                print(f"{aspect}={sentiment}, ", end="")
        print(f"\nOverall rating: {example['rating']}")

def main():
    """Main function"""
    # Load test set
    dataset = load_cebab_test_set()
    if not dataset:
        return
    
    # Extract diverse examples
    examples = extract_diverse_examples(dataset, num_examples=20)
    
    # Print some example previews
    print_example_previews(examples)
    
    # Save examples to files
    save_examples(examples)
    
    # Create aspect-specific files
    create_aspect_specific_files(examples)
    
    print("\nNow you can use these examples with your concept-rationale mapper:")
    print("  ./cebab_map.sh -f test_examples.txt -o mapped_examples.txt -h")
    print("  ./cebab_map.sh -f test_examples/food_positive.txt -o food_positive_concepts.txt -h")

if __name__ == "__main__":
    main() 