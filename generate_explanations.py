#!/usr/bin/env python
"""
Generate Explanations from Rationale-Concept Model Checkpoint

This script loads a trained Rationale-Concept Bottleneck Model checkpoint
and generates various types of explanations for input text.

Usage:
    python generate_explanations.py --checkpoint_path PATH_TO_CHECKPOINT --config_path PATH_TO_CONFIG
"""

import os
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
print(f"Using device: {device}")

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

def generate_basic_explanation(model, tokenizer, text):
    """
    Generate a basic explanation for a text input
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for preprocessing
        text: Input text
        
    Returns:
        explanation: Dictionary with explanation details
    """
    explanation = model.explain_prediction(tokenizer, text)
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

def generate_counterfactual_explanation(model, tokenizer, text, target_class):
    """
    Generate a counterfactual explanation by finding concepts that would
    change the prediction to the target class
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for preprocessing
        text: Input text
        target_class: Target class index
        
    Returns:
        counterfactual: Dictionary with counterfactual explanation
    """
    # First get the basic explanation
    basic_explanation = model.explain_prediction(tokenizer, text)
    original_prediction = basic_explanation['prediction']
    
    # If already predicting the target class, return
    if original_prediction == target_class:
        return {
            "message": "Input already classified as target class",
            "original_prediction": original_prediction,
            "target_class": target_class
        }
    
    # Try intervening on each concept
    counterfactual_concepts = []
    
    # Tokenize input once
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=model.config.max_seq_length, 
        padding="max_length", 
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get original outputs
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        concept_probs = outputs["concept_probs"].clone()
    
    # Try modifying each concept
    for concept_idx in range(model.num_concepts):
        # Try setting to 1.0
        modified_probs = concept_probs.clone()
        modified_probs[0, concept_idx] = 1.0
        
        # Predict with modified concepts
        with torch.no_grad():
            if model.use_skip_connection:
                combined = torch.cat([modified_probs, outputs["cls_embedding"]], dim=1)
                new_logits = model.classifier(combined)
            else:
                new_logits = model.classifier(modified_probs)
            
            new_prediction = torch.argmax(new_logits, dim=1).item()
        
        # If prediction changed to target class
        if new_prediction == target_class:
            counterfactual_concepts.append({
                "concept_idx": concept_idx,
                "concept_name": f"concept_{concept_idx}",
                "original_value": outputs["concept_probs"][0, concept_idx].item(),
                "modified_value": 1.0
            })
    
    # Try setting to 0.0 for concepts that didn't work at 1.0
    if not counterfactual_concepts:
        for concept_idx in range(model.num_concepts):
            # Try setting to 0.0
            modified_probs = concept_probs.clone()
            modified_probs[0, concept_idx] = 0.0
            
            # Predict with modified concepts
            with torch.no_grad():
                if model.use_skip_connection:
                    combined = torch.cat([modified_probs, outputs["cls_embedding"]], dim=1)
                    new_logits = model.classifier(combined)
                else:
                    new_logits = model.classifier(modified_probs)
                
                new_prediction = torch.argmax(new_logits, dim=1).item()
            
            # If prediction changed to target class
            if new_prediction == target_class:
                counterfactual_concepts.append({
                    "concept_idx": concept_idx,
                    "concept_name": f"concept_{concept_idx}",
                    "original_value": outputs["concept_probs"][0, concept_idx].item(),
                    "modified_value": 0.0
                })
    
    return {
        "original_prediction": original_prediction,
        "target_class": target_class,
        "counterfactual_concepts": counterfactual_concepts,
        "success": len(counterfactual_concepts) > 0
    }

def print_explanation(explanation, dataset_name=None):
    """Pretty print an explanation"""
    # Define class names for common datasets
    class_names = {
        'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'yelp_polarity': ['Negative', 'Positive'],
        'sst2': ['Negative', 'Positive'],
        'dbpedia': ['Company', 'Educational Institution', 'Artist', 'Athlete', 
                   'Office Holder', 'Mean Of Transportation', 'Building', 'Natural Place', 
                   'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
    }
    
    # Get class name if dataset is known
    prediction_class = explanation['prediction']
    if dataset_name and dataset_name in class_names:
        prediction_class = f"{prediction_class} ({class_names[dataset_name][prediction_class]})"
    
    print("\n" + "="*80)
    print(f"PREDICTION: {prediction_class} (Confidence: {explanation['confidence']:.4f})")
    print("-"*80)
    print(f"RATIONALE: \"{explanation['rationale']}\"")
    print(f"Rationale length: {explanation['rationale_percentage']*100:.1f}% of text")
    print("-"*80)
    print("TOP CONCEPTS:")
    for concept, prob in explanation['top_concepts']:
        print(f"  {concept}: {prob:.4f}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate explanations from a trained model")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to explain (if not provided, will use examples)")
    parser.add_argument("--dataset", type=str, choices=['ag_news', 'yelp_polarity', 'sst2', 'dbpedia'],
                        help="Dataset name for class labels")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["basic", "intervention", "counterfactual", "all"],
                        help="Type of explanation to generate")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for counterfactual explanations")
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
        # Example texts for different datasets
        texts = [
            "The company announced a new product that will revolutionize the market.",
            "The team won the championship after an incredible comeback in the final minutes.",
            "This restaurant has amazing food and excellent service. Highly recommended!",
            "The movie was boring and predictable. I wouldn't recommend it."
        ]
    
    # Generate explanations
    all_explanations = []
    
    for text in texts:
        print(f"\nGenerating explanations for: \"{text}\"")
        
        explanations = {"text": text}
        
        # Basic explanation
        if args.mode in ["basic", "all"]:
            basic_explanation = generate_basic_explanation(model, tokenizer, text)
            explanations["basic"] = basic_explanation
            print_explanation(basic_explanation, args.dataset)
        
        # Concept interventions
        if args.mode in ["intervention", "all"]:
            interventions = generate_concept_interventions(model, tokenizer, text)
            explanations["interventions"] = interventions
            
            print("\nCONCEPT INTERVENTIONS:")
            for intervention in interventions:
                concept = intervention["concept_name"]
                orig_val = intervention["concept_value"]["original"]
                new_val = intervention["concept_value"]["modified"]
                orig_pred = intervention["original_prediction"]
                new_pred = intervention["intervened_prediction"]
                
                # Get class names if dataset is known
                if args.dataset and args.dataset in {
                    'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
                    'yelp_polarity': ['Negative', 'Positive'],
                    'sst2': ['Negative', 'Positive']
                }:
                    class_names = {
                        'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
                        'yelp_polarity': ['Negative', 'Positive'],
                        'sst2': ['Negative', 'Positive']
                    }[args.dataset]
                    orig_pred_name = f"{orig_pred} ({class_names[orig_pred]})"
                    new_pred_name = f"{new_pred} ({class_names[new_pred]})"
                else:
                    orig_pred_name = str(orig_pred)
                    new_pred_name = str(new_pred)
                
                print(f"  {concept}: {orig_val:.4f} → {new_val:.4f}")
                print(f"    Prediction: {orig_pred_name} → {new_pred_name}")
                if orig_pred != new_pred:
                    print(f"    *** PREDICTION CHANGED ***")
                print()
        
        # Counterfactual explanation
        if args.mode in ["counterfactual", "all"]:
            counterfactual = generate_counterfactual_explanation(
                model, tokenizer, text, args.target_class
            )
            explanations["counterfactual"] = counterfactual
            
            print("\nCOUNTERFACTUAL EXPLANATION:")
            print(f"Target class: {args.target_class}")
            
            if counterfactual.get("message"):
                print(f"  {counterfactual['message']}")
            elif counterfactual["success"]:
                print("  Found concepts that change the prediction to target class:")
                for concept in counterfactual["counterfactual_concepts"]:
                    print(f"  {concept['concept_name']}: {concept['original_value']:.4f} → {concept['modified_value']:.4f}")
            else:
                print("  Could not find a single concept that changes the prediction to target class.")
                print("  Try modifying multiple concepts simultaneously or a different target class.")
        
        all_explanations.append(explanations)
    
    # Save explanations to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_explanations, f, indent=2)
        print(f"\nExplanations saved to {args.output_file}")

if __name__ == "__main__":
    main()
