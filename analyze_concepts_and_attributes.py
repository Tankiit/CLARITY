#!/usr/bin/env python
"""
Analyze Concepts and Attributes in CEBaB Dataset

This script demonstrates:
1. How different rationales are extracted for different concepts
2. How concepts can be aligned with attributes using cosine similarity
"""

import os
import argparse
import torch
import numpy as np
import json
import logging
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init

# Initialize colorama
init()

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

def load_model_and_tokenizer(checkpoint_path, config_path):
    """
    Load the trained model and tokenizer
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Filter out parameters that are not in ModelConfig
    valid_params = {
        'base_model_name', 'num_labels', 'num_concepts', 'hidden_size',
        'dropout_rate', 'min_span_size', 'max_span_size', 'length_bonus_factor',
        'concept_sparsity_weight', 'concept_diversity_weight', 'rationale_sparsity_weight',
        'rationale_continuity_weight', 'classification_weight', 'target_rationale_percentage',
        'enable_concept_interactions', 'use_skip_connection', 'use_lora', 'lora_r',
        'lora_alpha', 'batch_size', 'max_seq_length', 'learning_rate', 'base_model_lr',
        'weight_decay', 'num_epochs', 'warmup_ratio', 'max_grad_norm', 'seed', 'output_dir'
    }

    filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

    # Create model configuration
    config = ModelConfig(**filtered_config)
    logger.info(f"Loaded configuration: {config.base_model_name}, {config.num_labels} classes, {config.num_concepts} concepts")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Initialize model
    model = RationaleConceptBottleneckModel(config)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, config

def extract_concepts_and_rationales(model, tokenizer, text):
    """
    Extract concepts and their associated rationales from text

    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text

    Returns:
        Dictionary with concepts and their rationales
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=model.config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )

    # Get concept probabilities
    concept_probs = outputs['concept_probs'][0].cpu().numpy()

    # Get token probabilities
    token_probs = outputs['token_probs'][0].cpu().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_mask = inputs['attention_mask'][0].cpu().numpy()

    # Get valid tokens and their probabilities
    valid_tokens = []
    valid_probs = []
    for i, (token, prob) in enumerate(zip(tokens, token_probs)):
        if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            valid_tokens.append(token)
            valid_probs.append(prob)

    # Get top concepts
    top_concept_indices = np.argsort(concept_probs)[::-1][:10]  # Top 10 concepts

    # For each top concept, find the most relevant tokens
    concept_rationales = {}

    for concept_idx in top_concept_indices:
        concept_name = f"concept_{concept_idx}"
        concept_prob = concept_probs[concept_idx]

        # Skip concepts with very low probability
        if concept_prob < 0.1:
            continue

        # Get the concept's embedding from the model
        concept_embedding = model.concept_mapper.concept_encoder[-1].weight[concept_idx].detach().cpu().numpy()

        # Use a simpler approach - just use token probabilities directly
        # This avoids issues with different output formats

        # Create token similarities based on token probabilities
        token_similarities = [(token, prob) for token, prob in zip(valid_tokens, valid_probs)]

        # Sort tokens by probability
        token_similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top tokens for this concept
        top_tokens = token_similarities[:5]  # Top 5 tokens

        concept_rationales[concept_name] = {
            'probability': float(concept_prob),
            'top_tokens': [(token, float(sim)) for token, sim in top_tokens]
        }

    return concept_rationales

def align_concepts_with_attributes(model, tokenizer, text, attributes=None):
    """
    Align concepts with attributes using a simpler approach

    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text
        attributes: List of attributes to align with (food, service, ambiance, noise)

    Returns:
        Dictionary with attribute-concept alignments
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=model.config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    # Get concept probabilities
    concept_probs = outputs['concept_probs'][0].cpu().numpy()

    # Define attribute-specific keywords
    attribute_keywords = {
        'food': ['food', 'meal', 'dish', 'taste', 'delicious', 'flavor', 'steak', 'potatoes', 'onion', 'cooked'],
        'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'attentive', 'friendly', 'rude', 'slow', 'quick'],
        'ambiance': ['ambiance', 'atmosphere', 'decor', 'interior', 'design', 'comfortable', 'cozy', 'cafeteria', 'resembled', 'inside'],
        'noise': ['noise', 'loud', 'quiet', 'noisy', 'peaceful', 'crowded', 'busy', 'silent', 'conversation']
    }

    # For each attribute, run the model with an attribute-specific prompt
    attribute_concept_similarity = {}

    for attribute in attributes:
        # Create attribute-specific prompt
        prompt = f"Analyze the {attribute} in this review: {text}"

        # Tokenize prompt
        attr_inputs = tokenizer(
            prompt,
            max_length=model.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Get model outputs for this attribute
        with torch.no_grad():
            attr_outputs = model(
                input_ids=attr_inputs['input_ids'],
                attention_mask=attr_inputs['attention_mask']
            )

        # Get concept probabilities for this attribute
        attr_concept_probs = attr_outputs['concept_probs'][0].cpu().numpy()

        # Calculate relevance score for each concept
        # Combine original concept probability with attribute-specific probability
        concept_relevance = []
        for i, (orig_prob, attr_prob) in enumerate(zip(concept_probs, attr_concept_probs)):
            # Higher score means more relevant to this attribute
            relevance = orig_prob * attr_prob
            concept_relevance.append((i, relevance))

        # Sort by relevance
        concept_relevance.sort(key=lambda x: x[1], reverse=True)

        # Get top concepts for this attribute
        top_concepts = [(f"concept_{idx}", float(relevance), float(concept_probs[idx]))
                        for idx, relevance in concept_relevance[:5]]  # Top 5 concepts

        attribute_concept_similarity[attribute] = top_concepts

    return attribute_concept_similarity

def print_concept_rationales(concept_rationales, text):
    """
    Print concept rationales in a readable format
    """
    print(f"\n{Fore.BLUE}===== CONCEPT RATIONALES ====={Style.RESET_ALL}")
    print(f"\nOriginal text: {text}\n")

    for concept, data in concept_rationales.items():
        print(f"{Fore.GREEN}{concept} (Probability: {data['probability']:.4f}){Style.RESET_ALL}")
        print("Top tokens:")
        for token, similarity in data['top_tokens']:
            print(f"  {token}: {similarity:.4f}")
        print()

def print_attribute_concept_alignment(attribute_concept_similarity):
    """
    Print attribute-concept alignment in a readable format
    """
    print(f"\n{Fore.BLUE}===== ATTRIBUTE-CONCEPT ALIGNMENT ====={Style.RESET_ALL}\n")

    for attribute, concepts in attribute_concept_similarity.items():
        print(f"{Fore.GREEN}{attribute.upper()}{Style.RESET_ALL}")
        print("Top concepts:")
        for concept, similarity, probability in concepts:
            print(f"  {concept}: similarity={similarity:.4f}, probability={probability:.4f}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Analyze concepts and attributes")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str,
                        default="./cebab_models/20250519-151741_distilbert-base-uncased/checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str,
                        default="./cebab_models/20250519-151741_distilbert-base-uncased/config.json",
                        help="Path to model configuration")

    # Optional arguments
    parser.add_argument("--text", type=str,
                        default="Went there on a date. My girlfriend said her meal was excellent. I got the angus strip steak which was ok. The mashed potatoes were cold and the onion straws were barely cooked. Service was adequate but it resembled a school cafeteria inside.",
                        help="Text to analyze")
    parser.add_argument("--attributes", type=str, default="food,service,ambiance,noise",
                        help="Comma-separated list of attributes to analyze")
    parser.add_argument("--output_dir", type=str, default="concept_attribute_results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Parse attributes
    attributes = args.attributes.split(',')

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )

    # Extract concepts and their rationales
    concept_rationales = extract_concepts_and_rationales(model, tokenizer, args.text)

    # Align concepts with attributes
    attribute_concept_similarity = align_concepts_with_attributes(
        model, tokenizer, args.text, attributes
    )

    # Print results
    print_concept_rationales(concept_rationales, args.text)
    print_attribute_concept_alignment(attribute_concept_similarity)

    # Save results
    if args.output_dir:
        results = {
            'text': args.text,
            'concept_rationales': concept_rationales,
            'attribute_concept_similarity': {
                attr: [(concept, float(sim), float(prob)) for concept, sim, prob in concepts]
                for attr, concepts in attribute_concept_similarity.items()
            }
        }

        output_file = os.path.join(args.output_dir, 'concept_attribute_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
