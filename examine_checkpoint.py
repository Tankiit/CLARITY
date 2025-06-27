#!/usr/bin/env python
"""
Examine Checkpoint Structure

This script loads a checkpoint and prints its structure to help understand
how to properly use it for generating explanations.
"""

import argparse
import torch
import json
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def examine_checkpoint(checkpoint_path):
    """Load and examine a checkpoint's structure"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print basic info
    print(f"Checkpoint contains {len(checkpoint)} keys")
    
    # Group keys by prefix
    key_groups = defaultdict(list)
    for key in checkpoint.keys():
        # Get the prefix (e.g., 'encoder', 'rationale_extractor')
        prefix = key.split('.')[0]
        key_groups[prefix].append(key)
    
    # Print summary of key groups
    print("\nKey groups:")
    for prefix, keys in key_groups.items():
        print(f"  {prefix}: {len(keys)} keys")
    
    # Print detailed information for each group
    for prefix, keys in key_groups.items():
        print(f"\nDetails for {prefix}:")
        
        # For each group, sample a few keys and print their shapes
        sample_size = min(5, len(keys))
        for key in sorted(keys)[:sample_size]:
            tensor = checkpoint[key]
            print(f"  {key}: {tensor.shape}, {tensor.dtype}")
        
        if len(keys) > sample_size:
            print(f"  ... and {len(keys) - sample_size} more keys")
    
    # If there are classifier keys, examine them more closely
    if 'classifiers' in key_groups:
        print("\nClassifier details:")
        classifier_keys = key_groups['classifiers']
        
        # Group by dataset
        dataset_groups = defaultdict(list)
        for key in classifier_keys:
            parts = key.split('.')
            if len(parts) > 1:
                dataset = parts[1]
                dataset_groups[dataset].append(key)
        
        # Print dataset groups
        for dataset, keys in dataset_groups.items():
            print(f"  Dataset: {dataset}, {len(keys)} keys")
            for key in sorted(keys):
                tensor = checkpoint[key]
                print(f"    {key}: {tensor.shape}, {tensor.dtype}")
    
    # If there are concept mapper keys, examine them
    if 'concept_mapper' in key_groups:
        print("\nConcept mapper details:")
        concept_keys = key_groups['concept_mapper']
        
        # Group by dataset
        dataset_groups = defaultdict(list)
        for key in concept_keys:
            parts = key.split('.')
            if len(parts) > 2 and parts[1] == 'encoders':
                dataset = parts[2]
                dataset_groups[dataset].append(key)
            elif len(parts) > 2 and parts[1] == 'interactions':
                dataset = parts[2]
                dataset_groups[dataset].append(key)
        
        # Print dataset groups
        for dataset, keys in dataset_groups.items():
            print(f"  Dataset: {dataset}, {len(keys)} keys")
            for key in sorted(keys):
                tensor = checkpoint[key]
                print(f"    {key}: {tensor.shape}, {tensor.dtype}")

def main():
    parser = argparse.ArgumentParser(description="Examine checkpoint structure")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    examine_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main()
