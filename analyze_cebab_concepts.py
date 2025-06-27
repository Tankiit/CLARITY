#!/usr/bin/env python
"""
Analyze CEBAB Concepts

This script analyzes the concepts learned by a model trained on CEBAB
and maps them to human-readable descriptions.
"""

import os
import argparse
import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset
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

def load_cebab_dataset():
    """Load CEBAB dataset"""
    logger.info("Loading CEBAB dataset...")
    
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("CEBaB/CEBaB")
        logger.info("Successfully loaded CEBAB from Hugging Face")
    except Exception as e:
        logger.warning(f"Error loading from Hugging Face: {e}")
        logger.info("Trying to load from local files...")
        
        # Try loading from local files
        dataset = {
            'train': load_dataset('json', data_files='train_inclusive.json', split='train'),
            'validation': load_dataset('json', data_files='dev.json', split='train'),
            'test': load_dataset('json', data_files='test.json', split='train')
        }
        logger.info("Successfully loaded CEBAB from local files")
    
    return dataset

def analyze_concepts(model, tokenizer, dataset, num_samples=100, num_concepts=10):
    """
    Analyze concepts and map them to human-readable descriptions
    
    This function:
    1. Samples reviews from the dataset
    2. Extracts concepts for each review
    3. Analyzes which aspects (food, service, etc.) correlate with each concept
    4. Maps concepts to human-readable descriptions
    """
    logger.info(f"Analyzing concepts using {num_samples} samples...")
    
    # Sample reviews from the dataset
    samples = dataset['test'].shuffle(seed=42).select(range(num_samples))
    
    # Extract concepts for each review
    concept_activations = []
    aspect_ratings = []
    
    for sample in samples:
        # Get review text
        text = sample['review']
        
        # Get aspect ratings
        aspects = {
            'food': sample.get('food_rating', 0),
            'service': sample.get('service_rating', 0),
            'ambiance': sample.get('ambiance_rating', 0),
            'noise': sample.get('noise_rating', 0),
            'price': sample.get('price_rating', 0)
        }
        
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=128, 
            padding="max_length", 
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
        
        # Get concept probabilities
        concept_probs = outputs["concept_probs"][0].cpu().numpy()
        
        # Store concept activations and aspect ratings
        concept_activations.append(concept_probs)
        aspect_ratings.append(aspects)
    
    # Convert to numpy arrays
    concept_activations = np.array(concept_activations)
    
    # Calculate correlation between concepts and aspects
    concept_aspect_correlations = defaultdict(dict)
    aspect_names = ['food', 'service', 'ambiance', 'noise', 'price']
    
    for aspect in aspect_names:
        # Get aspect ratings (skip missing ratings)
        ratings = [sample[aspect] for sample in aspect_ratings if sample[aspect] > 0]
        if not ratings:
            continue
        
        # Get corresponding concept activations
        indices = [i for i, sample in enumerate(aspect_ratings) if sample[aspect] > 0]
        activations = concept_activations[indices]
        
        # Calculate correlation for each concept
        for concept_idx in range(model.num_concepts):
            concept_values = activations[:, concept_idx]
            
            # Calculate correlation if we have enough data
            if len(ratings) > 5:
                correlation = np.corrcoef(concept_values, ratings)[0, 1]
                concept_aspect_correlations[f"concept_{concept_idx}"][aspect] = correlation
    
    # Map concepts to human-readable descriptions
    concept_descriptions = {}
    
    for concept, correlations in concept_aspect_correlations.items():
        # Sort aspects by absolute correlation
        sorted_aspects = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Create description based on top correlations
        if not sorted_aspects:
            description = "Unknown concept"
        else:
            top_aspect, top_corr = sorted_aspects[0]
            direction = "positive" if top_corr > 0 else "negative"
            description = f"{direction} {top_aspect}"
            
            # Add second aspect if correlation is significant
            if len(sorted_aspects) > 1 and abs(sorted_aspects[1][1]) > 0.2:
                second_aspect, second_corr = sorted_aspects[1]
                second_direction = "positive" if second_corr > 0 else "negative"
                description += f" and {second_direction} {second_aspect}"
        
        concept_descriptions[concept] = {
            "description": description,
            "correlations": dict(sorted_aspects)
        }
    
    # Sort concepts by maximum correlation
    sorted_concepts = sorted(
        concept_descriptions.items(),
        key=lambda x: max([abs(corr) for corr in x[1]["correlations"].values()], default=0),
        reverse=True
    )
    
    return dict(sorted_concepts[:num_concepts]), concept_activations, aspect_ratings

def visualize_concepts(concept_descriptions, output_dir):
    """Visualize concept correlations with aspects"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a heatmap of concept-aspect correlations
    aspects = ['food', 'service', 'ambiance', 'noise', 'price']
    concepts = list(concept_descriptions.keys())
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(concepts), len(aspects)))
    
    for i, concept in enumerate(concepts):
        for j, aspect in enumerate(aspects):
            correlation_matrix[i, j] = concept_descriptions[concept]["correlations"].get(aspect, 0)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add labels
    plt.xticks(np.arange(len(aspects)), aspects, rotation=45)
    plt.yticks(np.arange(len(concepts)), [f"{c}: {concept_descriptions[c]['description']}" for c in concepts])
    
    # Add colorbar
    plt.colorbar(label='Correlation')
    
    # Add title and labels
    plt.title('Concept-Aspect Correlations')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'concept_correlations.png'))
    plt.close()
    
    # Save concept descriptions to file
    with open(os.path.join(output_dir, 'concept_descriptions.json'), 'w') as f:
        json.dump(concept_descriptions, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Analyze CEBAB concepts")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to use for analysis")
    parser.add_argument("--num_concepts", type=int, default=10,
                        help="Number of top concepts to analyze")
    parser.add_argument("--output_dir", type=str, default="concept_analysis",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path, args.config_path
    )
    
    # Load CEBAB dataset
    dataset = load_cebab_dataset()
    
    # Analyze concepts
    concept_descriptions, concept_activations, aspect_ratings = analyze_concepts(
        model, tokenizer, dataset, args.num_samples, args.num_concepts
    )
    
    # Print concept descriptions
    print("\nConcept Descriptions:")
    print("=" * 80)
    for concept, info in concept_descriptions.items():
        print(f"{concept}: {info['description']}")
        print("  Correlations:")
        for aspect, corr in info['correlations'].items():
            print(f"    {aspect}: {corr:.4f}")
        print()
    
    # Visualize concepts
    visualize_concepts(concept_descriptions, args.output_dir)
    logger.info(f"Concept analysis saved to {args.output_dir}")

if __name__ == "__main__":
    main()
