#!/usr/bin/env python
"""
Explore Concept Space and Generate Advanced Explanations

This script provides an interactive way to explore the concept space
of a trained Rationale-Concept Bottleneck Model and generate various
types of explanations.

Features:
- Analyze concept importance across multiple examples
- Generate contrastive explanations between different texts
- Find minimal sets of concepts that change predictions
- Visualize concept interactions
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
print(f"Using device: {device}")

def load_model_from_checkpoint(checkpoint_path, config_path):
    """Load model from checkpoint"""
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

def analyze_concept_importance(model, tokenizer, texts, output_dir=None):
    """
    Analyze which concepts are most important for classification
    across multiple examples
    """
    all_concept_effects = {}
    
    for text in texts:
        # Get basic explanation
        explanation = model.explain_prediction(tokenizer, text)
        original_prediction = explanation['prediction']
        
        # Try modifying each concept
        for concept_idx in range(model.num_concepts):
            # Set concept to 0
            intervention_0 = model.intervene_on_concepts(tokenizer, text, concept_idx, 0.0)
            # Set concept to 1
            intervention_1 = model.intervene_on_concepts(tokenizer, text, concept_idx, 1.0)
            
            # Check if prediction changed
            changed_to_0 = intervention_0['intervened_prediction'] != original_prediction
            changed_to_1 = intervention_1['intervened_prediction'] != original_prediction
            
            # Record effect
            concept_name = f"concept_{concept_idx}"
            if concept_name not in all_concept_effects:
                all_concept_effects[concept_name] = {
                    'changed_to_0': 0,
                    'changed_to_1': 0,
                    'total': 0
                }
            
            all_concept_effects[concept_name]['changed_to_0'] += int(changed_to_0)
            all_concept_effects[concept_name]['changed_to_1'] += int(changed_to_1)
            all_concept_effects[concept_name]['total'] += 1
    
    # Calculate importance scores
    for concept in all_concept_effects:
        total = all_concept_effects[concept]['total']
        all_concept_effects[concept]['importance'] = (
            all_concept_effects[concept]['changed_to_0'] + 
            all_concept_effects[concept]['changed_to_1']
        ) / (2 * total) if total > 0 else 0
    
    # Sort by importance
    sorted_concepts = sorted(
        all_concept_effects.items(),
        key=lambda x: x[1]['importance'],
        reverse=True
    )
    
    # Print results
    print("\nCONCEPT IMPORTANCE ANALYSIS")
    print("=" * 60)
    print(f"{'Concept':<15} {'Importance':<10} {'Changed to 0':<15} {'Changed to 1':<15}")
    print("-" * 60)
    
    for concept, stats in sorted_concepts[:20]:  # Show top 20
        print(f"{concept:<15} {stats['importance']:.4f}      {stats['changed_to_0']}/{stats['total']}           {stats['changed_to_1']}/{stats['total']}")
    
    # Plot results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get top 20 concepts
        top_concepts = [c for c, _ in sorted_concepts[:20]]
        importance_scores = [stats['importance'] for _, stats in sorted_concepts[:20]]
        
        plt.figure(figsize=(12, 8))
        plt.bar(top_concepts, importance_scores)
        plt.xlabel('Concept')
        plt.ylabel('Importance Score')
        plt.title('Concept Importance Analysis')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'concept_importance.png'))
        
        # Save results
        with open(os.path.join(output_dir, 'concept_importance.json'), 'w') as f:
            json.dump(all_concept_effects, f, indent=2)
    
    return all_concept_effects

def generate_contrastive_explanation(model, tokenizer, text1, text2):
    """
    Generate a contrastive explanation between two texts
    """
    # Get explanations for both texts
    explanation1 = model.explain_prediction(tokenizer, text1)
    explanation2 = model.explain_prediction(tokenizer, text2)
    
    # Get predictions
    pred1 = explanation1['prediction']
    pred2 = explanation2['prediction']
    
    # Get concept probabilities
    concept_probs1 = {c[0]: c[1] for c in explanation1['top_concepts']}
    concept_probs2 = {c[0]: c[1] for c in explanation2['top_concepts']}
    
    # Find concepts that differ significantly
    all_concepts = set(concept_probs1.keys()) | set(concept_probs2.keys())
    differing_concepts = []
    
    for concept in all_concepts:
        val1 = concept_probs1.get(concept, 0)
        val2 = concept_probs2.get(concept, 0)
        diff = abs(val1 - val2)
        
        if diff > 0.2:  # Threshold for significant difference
            differing_concepts.append({
                'concept': concept,
                'text1_value': val1,
                'text2_value': val2,
                'difference': diff
            })
    
    # Sort by difference
    differing_concepts.sort(key=lambda x: x['difference'], reverse=True)
    
    # Get rationales
    rationale1 = explanation1['rationale']
    rationale2 = explanation2['rationale']
    
    # Create contrastive explanation
    contrastive = {
        'text1': text1,
        'text2': text2,
        'prediction1': pred1,
        'prediction2': pred2,
        'rationale1': rationale1,
        'rationale2': rationale2,
        'differing_concepts': differing_concepts
    }
    
    # Print contrastive explanation
    print("\nCONTRASTIVE EXPLANATION")
    print("=" * 80)
    print(f"Text 1: \"{text1}\"")
    print(f"Prediction: {pred1}")
    print(f"Rationale: \"{rationale1}\"")
    print("-" * 80)
    print(f"Text 2: \"{text2}\"")
    print(f"Prediction: {pred2}")
    print(f"Rationale: \"{rationale2}\"")
    print("-" * 80)
    print("Key differences in concepts:")
    
    for diff in differing_concepts[:5]:  # Show top 5
        concept = diff['concept']
        val1 = diff['text1_value']
        val2 = diff['text2_value']
        print(f"  {concept}: {val1:.4f} vs {val2:.4f} (diff: {diff['difference']:.4f})")
    
    return contrastive

def find_minimal_concept_set(model, tokenizer, text, target_class, max_concepts=3):
    """
    Find a minimal set of concepts that, when modified together,
    change the prediction to the target class
    """
    # Get basic explanation
    explanation = model.explain_prediction(tokenizer, text)
    original_prediction = explanation['prediction']
    
    # If already predicting target class, return
    if original_prediction == target_class:
        return {
            "message": "Input already classified as target class",
            "original_prediction": original_prediction,
            "target_class": target_class
        }
    
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
    
    # First try single concepts (already implemented in counterfactual explanation)
    # ...
    
    # Try pairs of concepts
    if max_concepts >= 2:
        for i in range(model.num_concepts):
            for j in range(i+1, model.num_concepts):
                # Try setting both to 1.0
                modified_probs = concept_probs.clone()
                modified_probs[0, i] = 1.0
                modified_probs[0, j] = 1.0
                
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
                    return {
                        "original_prediction": original_prediction,
                        "target_class": target_class,
                        "concept_set": [
                            {
                                "concept_idx": i,
                                "concept_name": f"concept_{i}",
                                "original_value": outputs["concept_probs"][0, i].item(),
                                "modified_value": 1.0
                            },
                            {
                                "concept_idx": j,
                                "concept_name": f"concept_{j}",
                                "original_value": outputs["concept_probs"][0, j].item(),
                                "modified_value": 1.0
                            }
                        ],
                        "set_size": 2
                    }
    
    # Try triplets of concepts
    if max_concepts >= 3:
        for i in range(model.num_concepts):
            for j in range(i+1, model.num_concepts):
                for k in range(j+1, model.num_concepts):
                    # Try setting all three to 1.0
                    modified_probs = concept_probs.clone()
                    modified_probs[0, i] = 1.0
                    modified_probs[0, j] = 1.0
                    modified_probs[0, k] = 1.0
                    
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
                        return {
                            "original_prediction": original_prediction,
                            "target_class": target_class,
                            "concept_set": [
                                {
                                    "concept_idx": i,
                                    "concept_name": f"concept_{i}",
                                    "original_value": outputs["concept_probs"][0, i].item(),
                                    "modified_value": 1.0
                                },
                                {
                                    "concept_idx": j,
                                    "concept_name": f"concept_{j}",
                                    "original_value": outputs["concept_probs"][0, j].item(),
                                    "modified_value": 1.0
                                },
                                {
                                    "concept_idx": k,
                                    "concept_name": f"concept_{k}",
                                    "original_value": outputs["concept_probs"][0, k].item(),
                                    "modified_value": 1.0
                                }
                            ],
                            "set_size": 3
                        }
    
    # Could not find a minimal set
    return {
        "original_prediction": original_prediction,
        "target_class": target_class,
        "message": f"Could not find a set of up to {max_concepts} concepts that changes the prediction to target class.",
        "success": False
    }

def main():
    parser = argparse.ArgumentParser(description="Explore concept space and generate advanced explanations")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--mode", type=str, default="importance",
                        choices=["importance", "contrastive", "minimal_set"],
                        help="Type of analysis to perform")
    parser.add_argument("--text", type=str,
                        help="Text to analyze (required for contrastive and minimal_set modes)")
    parser.add_argument("--text2", type=str,
                        help="Second text for contrastive analysis")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for minimal set analysis")
    parser.add_argument("--output_dir", type=str, default="concept_analysis",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path, args.config_path
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected analysis
    if args.mode == "importance":
        # Use example texts if none provided
        texts = []
        if args.text:
            texts.append(args.text)
        if args.text2:
            texts.append(args.text2)
            
        if not texts:
            texts = [
                "The company announced a new product that will revolutionize the market.",
                "The team won the championship after an incredible comeback in the final minutes.",
                "This restaurant has amazing food and excellent service. Highly recommended!",
                "The movie was boring and predictable. I wouldn't recommend it."
            ]
        
        analyze_concept_importance(model, tokenizer, texts, args.output_dir)
        
    elif args.mode == "contrastive":
        if not args.text or not args.text2:
            print("Error: Both --text and --text2 are required for contrastive analysis")
            return
        
        generate_contrastive_explanation(model, tokenizer, args.text, args.text2)
        
    elif args.mode == "minimal_set":
        if not args.text:
            print("Error: --text is required for minimal set analysis")
            return
        
        result = find_minimal_concept_set(model, tokenizer, args.text, args.target_class)
        
        print("\nMINIMAL CONCEPT SET ANALYSIS")
        print("=" * 80)
        print(f"Text: \"{args.text}\"")
        print(f"Original prediction: {result['original_prediction']}")
        print(f"Target class: {args.target_class}")
        
        if "message" in result:
            print(f"Result: {result['message']}")
        elif "concept_set" in result:
            print(f"Found minimal set of {result['set_size']} concepts:")
            for concept in result["concept_set"]:
                print(f"  {concept['concept_name']}: {concept['original_value']:.4f} → {concept['modified_value']:.4f}")
        
        # Save result
        with open(os.path.join(args.output_dir, 'minimal_concept_set.json'), 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
