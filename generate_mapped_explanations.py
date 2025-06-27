#!/usr/bin/env python
"""
Generate Explanations with Mapped Concepts

This script generates explanations for text inputs using a model trained on CEBAB,
with concepts mapped to human-readable descriptions.
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
    """Load a trained model from checkpoint"""
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

def load_concept_mappings(mappings_path):
    """Load concept mappings from file"""
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    return mappings

def generate_explanation(model, tokenizer, text, concept_mappings=None, min_concept_prob=0.5):
    """Generate an explanation for a text input with mapped concepts"""
    # Get basic explanation
    explanation = model.explain_prediction(tokenizer, text, min_concept_prob)
    
    # Map concepts if mappings are provided
    if concept_mappings:
        mapped_concepts = []
        for concept, prob in explanation['top_concepts']:
            if concept in concept_mappings:
                description = concept_mappings[concept]['description']
                mapped_concepts.append({
                    'concept': concept,
                    'description': description,
                    'probability': prob
                })
            else:
                mapped_concepts.append({
                    'concept': concept,
                    'description': 'Unknown concept',
                    'probability': prob
                })
        
        explanation['mapped_concepts'] = mapped_concepts
    
    return explanation

def generate_concept_interventions(model, tokenizer, text, concept_mappings=None, num_concepts=5):
    """Generate explanations by intervening on different concepts"""
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
        
        # Try setting the concept to 1 (fully on)
        one_intervention = model.intervene_on_concepts(tokenizer, text, concept_idx, 1.0)
        
        # Add mapped description if available
        concept_name = f"concept_{concept_idx}"
        if concept_mappings and concept_name in concept_mappings:
            description = concept_mappings[concept_name]['description']
            zero_intervention['concept_description'] = description
            one_intervention['concept_description'] = description
        
        interventions.append(zero_intervention)
        interventions.append(one_intervention)
    
    return interventions

def print_explanation(explanation):
    """Pretty print an explanation with mapped concepts"""
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
    
    if 'mapped_concepts' in explanation:
        print("TOP CONCEPTS (WITH DESCRIPTIONS):")
        for concept in explanation['mapped_concepts']:
            print(f"  {concept['concept']} ({concept['description']}): {concept['probability']:.4f}")
    else:
        print("TOP CONCEPTS:")
        for concept, prob in explanation['top_concepts']:
            print(f"  {concept}: {prob:.4f}")
    
    print("="*80 + "\n")

def print_intervention(intervention):
    """Pretty print an intervention result with mapped concepts"""
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
    concept_desc = intervention.get('concept_description', 'Unknown concept')
    orig_val = intervention['concept_value']['original']
    new_val = intervention['concept_value']['modified']
    
    print(f"Intervening on {concept} ({concept_desc}):")
    print(f"  Original value: {orig_val:.4f}")
    print(f"  Setting to {new_val:.4f}:")
    print(f"    Original prediction: {orig_class}")
    print(f"    New prediction: {new_class}")
    
    if orig_pred != new_pred:
        print(f"    *** PREDICTION CHANGED ***")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate explanations with mapped concepts")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    parser.add_argument("--mappings_path", type=str, required=True,
                        help="Path to concept mappings JSON file")
    
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
    
    # Load concept mappings
    concept_mappings = load_concept_mappings(args.mappings_path)
    logger.info(f"Loaded {len(concept_mappings)} concept mappings")
    
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
            basic_explanation = generate_explanation(model, tokenizer, text, concept_mappings)
            explanations["basic"] = basic_explanation
            print_explanation(basic_explanation)
        
        # Concept interventions
        if args.mode in ["intervention", "all"]:
            interventions = generate_concept_interventions(model, tokenizer, text, concept_mappings)
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
