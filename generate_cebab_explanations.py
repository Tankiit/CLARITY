#!/usr/bin/env python
"""
Generate Explanations from CEBAB-trained Model

This script loads a model trained on CEBAB and generates explanations
for text inputs, with higher confidence scores and better rationales.
"""

import os
import argparse
import json
import torch
import numpy as np
import logging
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
logger.info(f"Using device: {device}")

def load_model_from_checkpoint(checkpoint_path, config_path):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration
        
    Returns:
        model: Loaded model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    # Load configuration
    config = ModelConfig.load(config_path)
    
    # Initialize model with config
    model = RationaleConceptBottleneckModel(config)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    
    return model, tokenizer, config

def generate_explanation(model, tokenizer, text, min_concept_prob=0.5):
    """
    Generate an explanation for a text input
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for preprocessing
        text: Input text
        min_concept_prob: Minimum probability for a concept to be considered active
        
    Returns:
        explanation: Dictionary with explanation details
    """
    explanation = model.explain_prediction(tokenizer, text, min_concept_prob)
    return explanation

def generate_concept_interventions(model, tokenizer, text, num_concepts=5):
    """
    Generate explanations by intervening on different concepts
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for preprocessing
        text: Input text
        num_concepts: Number of top concepts to intervene on
        
    Returns:
        interventions: List of intervention results
    """
    # First get the basic explanation to identify top concepts
    basic_explanation = model.explain_prediction(tokenizer, text)
    
    # Get the top concept indices
    concept_indices = [int(concept[0].split('_')[1]) for concept in basic_explanation['top_concepts']]
    
    # If we don't have enough top concepts, add some random ones
    if len(concept_indices) < num_concepts:
        all_indices = set(range(model.num_concepts))
        remaining = list(all_indices - set(concept_indices))
        concept_indices.extend(np.random.choice(remaining, 
                                               num_concepts - len(concept_indices), 
                                               replace=False))
    
    # Generate interventions for each concept
    interventions = []
    
    # Try different values for each concept
    for concept_idx in concept_indices[:num_concepts]:
        # Try setting the concept to 0 (off)
        zero_intervention = model.intervene_on_concepts(tokenizer, text, concept_idx, 0.0)
        interventions.append(zero_intervention)
        
        # Try setting the concept to 1 (fully on)
        one_intervention = model.intervene_on_concepts(tokenizer, text, concept_idx, 1.0)
        interventions.append(one_intervention)
    
    return interventions

def print_explanation(explanation, dataset_name=None):
    """Pretty print an explanation"""
    # Define class names for CEBAB (1-5 stars)
    class_names = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    
    # Get class name
    prediction_class = explanation['prediction']
    class_name = class_names[prediction_class] if prediction_class < len(class_names) else str(prediction_class)
    
    print("\n" + "="*80)
    print(f"PREDICTION: {class_name} (Confidence: {explanation['confidence']:.4f})")
    print("-"*80)
    print(f"RATIONALE: \"{explanation['rationale']}\"")
    print(f"Rationale length: {explanation['rationale_percentage']*100:.1f}% of text")
    print("-"*80)
    print("TOP CONCEPTS:")
    for concept, prob in explanation['top_concepts']:
        print(f"  {concept}: {prob:.4f}")
    print("="*80 + "\n")

def print_intervention(intervention, dataset_name=None):
    """Pretty print an intervention result"""
    # Define class names for CEBAB (1-5 stars)
    class_names = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    
    # Get original and new predictions
    orig_pred = intervention['original_prediction']
    new_pred = intervention['intervened_prediction']
    
    # Get class names
    orig_class = class_names[orig_pred] if orig_pred < len(class_names) else str(orig_pred)
    new_class = class_names[new_pred] if new_pred < len(class_names) else str(new_pred)
    
    # Get concept info
    concept = intervention['concept_name']
    orig_val = intervention['concept_value']['original']
    new_val = intervention['concept_value']['modified']
    
    print(f"Intervening on {concept}:")
    print(f"  Original value: {orig_val:.4f}")
    print(f"  Setting to {new_val:.4f}:")
    print(f"    Original prediction: {orig_class}")
    print(f"    New prediction: {new_class}")
    
    if orig_pred != new_pred:
        print(f"    *** PREDICTION CHANGED ***")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate explanations from CEBAB-trained model")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to explain (if not provided, will use examples)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["basic", "intervention", "all"],
                        help="Type of explanation to generate")
    parser.add_argument("--output_file", type=str,
                        help="Path to save explanations as JSON")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path, args.config_path
    )
    
    # Use provided text or examples
    texts = []
    if args.text:
        texts.append(args.text)
    else:
        # Example texts for CEBAB (restaurant reviews)
        texts = [
            "The food was delicious and the service was excellent. I would definitely come back!",
            "Terrible experience. The food was cold and the waiter was rude.",
            "Average restaurant. Food was okay but nothing special. Service was good though.",
            "Great ambiance but the food was mediocre. The prices were too high for the quality.",
            "The best meal I've had in years! Everything from the appetizers to the dessert was perfect."
        ]
    
    # Generate explanations
    all_explanations = []
    
    for text in texts:
        print(f"\nGenerating explanations for: \"{text}\"")
        
        explanations = {"text": text}
        
        # Basic explanation
        if args.mode in ["basic", "all"]:
            basic_explanation = generate_explanation(model, tokenizer, text)
            explanations["basic"] = basic_explanation
            print_explanation(basic_explanation)
        
        # Concept interventions
        if args.mode in ["intervention", "all"]:
            interventions = generate_concept_interventions(model, tokenizer, text)
            explanations["interventions"] = interventions
            
            print("\nCONCEPT INTERVENTIONS:")
            for intervention in interventions:
                print_intervention(intervention)
        
        all_explanations.append(explanations)
    
    # Save explanations to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_explanations, f, indent=2)
        print(f"\nExplanations saved to {args.output_file}")

if __name__ == "__main__":
    main()
