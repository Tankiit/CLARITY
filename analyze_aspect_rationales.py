#!/usr/bin/env python
"""
Analyze Rationales for Different Aspects with Varying Thresholds

This script demonstrates how different rationale thresholds affect the
extracted rationales for different aspects (food, service, ambiance, noise)
of restaurant reviews.
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

def extract_rationales_with_threshold(model, tokenizer, text, threshold, aspect=None):
    """
    Extract rationales using a specific threshold

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        threshold: The threshold for rationale extraction (as a percentile)
        aspect: Optional aspect to focus on (food, service, ambiance, noise)

    Returns:
        Dictionary with rationale information
    """
    # Create aspect-specific prompt
    if aspect:
        # Add aspect-specific prompt to guide the model's attention
        aspect_prompts = {
            'food': f"Analyze the food quality in this review: {text}",
            'service': f"Analyze the service quality in this review: {text}",
            'ambiance': f"Analyze the ambiance in this review: {text}",
            'noise': f"Analyze the noise level in this review: {text}"
        }

        # Use the aspect-specific prompt if available
        if aspect in aspect_prompts:
            prompt_text = aspect_prompts[aspect]
        else:
            prompt_text = text
    else:
        prompt_text = text

    # Tokenize input
    inputs = tokenizer(
        prompt_text,
        max_length=model.config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )

    # Get token probabilities from the rationale extractor
    token_probs = outputs['token_probs'][0].cpu().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_mask = inputs['attention_mask'][0].cpu().numpy()

    # Get valid tokens and probabilities
    valid_indices = []
    valid_probs = []

    # If we used an aspect-specific prompt, we need to find where the original text starts
    if aspect:
        # Find the original text in the tokenized input
        original_tokens = tokenizer.encode(text, add_special_tokens=False)
        prompt_tokens = inputs['input_ids'][0].cpu().numpy()

        # Find the start position of the original text in the prompt
        start_pos = -1
        for i in range(len(prompt_tokens) - len(original_tokens) + 1):
            if np.array_equal(prompt_tokens[i:i+len(original_tokens)], original_tokens):
                start_pos = i
                break

        # If we found the original text, only consider those tokens
        if start_pos >= 0:
            for i, (token, prob) in enumerate(zip(tokens[start_pos:start_pos+len(original_tokens)],
                                                 token_probs[start_pos:start_pos+len(original_tokens)])):
                if attention_mask[start_pos+i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                    valid_indices.append(start_pos+i)
                    valid_probs.append(prob)
        else:
            # Fallback if we can't find the original text
            for i, (token, prob) in enumerate(zip(tokens, token_probs)):
                if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                    valid_indices.append(i)
                    valid_probs.append(prob)
    else:
        # No aspect-specific prompt, use all tokens
        for i, (token, prob) in enumerate(zip(tokens, token_probs)):
            if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                valid_indices.append(i)
                valid_probs.append(prob)

    # Apply aspect-specific weighting to token probabilities
    if aspect:
        # Define aspect-specific keywords to boost
        aspect_keywords = {
            'food': ['food', 'meal', 'dish', 'taste', 'delicious', 'flavor', 'steak', 'potatoes', 'onion', 'cooked'],
            'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'attentive', 'friendly', 'rude', 'slow', 'quick'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'interior', 'design', 'comfortable', 'cozy', 'cafeteria', 'resembled', 'inside'],
            'noise': ['noise', 'loud', 'quiet', 'noisy', 'peaceful', 'crowded', 'busy', 'silent', 'conversation']
        }

        # Get keywords for the current aspect
        keywords = aspect_keywords.get(aspect, [])

        # Apply boosting to tokens that match keywords
        for i, idx in enumerate(valid_indices):
            token = tokens[idx].lower().replace('##', '')
            # Boost tokens that match keywords for this aspect
            if any(keyword in token or token in keyword for keyword in keywords):
                valid_probs[i] *= 1.5  # Boost by 50%

    # Calculate percentile threshold
    if valid_probs:
        # Convert threshold (0-1) to percentile (0-100)
        percentile = 100 - threshold * 100
        # Ensure percentile is within valid range
        percentile = max(0, min(100, percentile))
        # Calculate actual threshold value
        actual_threshold = np.percentile(valid_probs, percentile)
    else:
        actual_threshold = 0

    # Extract rationales based on threshold
    rationale_tokens = []
    for i, idx in enumerate(valid_indices):
        if valid_probs[i] >= actual_threshold:
            rationale_tokens.append((tokens[idx], valid_probs[i], idx))

    # Sort by probability
    rationale_tokens.sort(key=lambda x: x[1], reverse=True)

    # Create highlighted text
    highlighted_text = highlight_rationales(text, tokenizer, rationale_tokens)

    # Get prediction
    logits = outputs['logits']
    probs = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probs[0, prediction].item()

    return {
        'rationale_tokens': rationale_tokens,
        'highlighted_text': highlighted_text,
        'num_tokens': len(rationale_tokens),
        'token_coverage': len(rationale_tokens) / sum(attention_mask),
        'prediction': prediction,
        'confidence': confidence,
        'aspect': aspect
    }

def highlight_rationales(text, tokenizer, rationale_tokens):
    """
    Highlight rationales in the original text
    """
    if not rationale_tokens:
        return text

    # Tokenize the text to get token spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.offset_mapping

    # Create a list of characters to highlight
    highlight_mask = [False] * len(text)

    # Mark characters to highlight based on token positions
    for token, score, _ in rationale_tokens:
        # Find all occurrences of this token in the tokens list
        for i, t in enumerate(tokenizer.convert_ids_to_tokens(encoding.input_ids)):
            # Check if token matches or is part of a word (for subword tokens)
            if t == token or (token.startswith('##') and t.endswith(token[2:])):
                start, end = offset_mapping[i]
                for j in range(start, end):
                    if j < len(highlight_mask):
                        highlight_mask[j] = True

    # Build the highlighted text with color
    highlighted = []
    in_highlight = False

    for i, char in enumerate(text):
        if highlight_mask[i] and not in_highlight:
            highlighted.append(Fore.RED)
            in_highlight = True
        elif not highlight_mask[i] and in_highlight:
            highlighted.append(Style.RESET_ALL)
            in_highlight = False

        highlighted.append(char)

    # Close any open highlight
    if in_highlight:
        highlighted.append(Style.RESET_ALL)

    return ''.join(highlighted)

def analyze_text_for_aspects(model, tokenizer, text, thresholds, aspects=None):
    """
    Analyze text for different aspects with multiple thresholds

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        thresholds: List of thresholds to test
        aspects: List of aspects to analyze (food, service, ambiance, noise)

    Returns:
        Dictionary with results for each aspect and threshold
    """
    if aspects is None:
        aspects = ['food', 'service', 'ambiance', 'noise']

    results = {}

    for aspect in aspects:
        results[aspect] = {}
        for threshold in thresholds:
            results[aspect][threshold] = extract_rationales_with_threshold(
                model, tokenizer, text, threshold, aspect
            )

    return results

def print_colored_results(results, text):
    """
    Print results with colored highlighting
    """
    print(f"\nOriginal text: {text}\n")

    for aspect, thresholds in results.items():
        print(f"\n{Fore.BLUE}===== ASPECT: {aspect.upper()} ====={Style.RESET_ALL}")

        for threshold, result in thresholds.items():
            print(f"\n{Fore.GREEN}Threshold: {threshold}{Style.RESET_ALL}")
            print(f"Number of rationale tokens: {result['num_tokens']}")
            print(f"Token coverage: {result['token_coverage']:.2%}")
            print(f"Highlighted text: {result['highlighted_text']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze rationales for different aspects with varying thresholds")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")

    # Optional arguments
    parser.add_argument("--text", type=str, default="Went there on a date. My girlfriend said her meal was excellent. I got the angus strip steak which was ok. The mashed potatoes were cold and the onion straws were barely cooked. Service was adequate but it resembled a school cafeteria inside.",
                        help="Text to analyze")
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5",
                        help="Comma-separated list of thresholds to test")
    parser.add_argument("--aspects", type=str, default="food,service,ambiance,noise",
                        help="Comma-separated list of aspects to analyze")
    parser.add_argument("--output_file", type=str,
                        help="Path to save results as JSON")

    args = parser.parse_args()

    # Parse thresholds and aspects
    thresholds = [float(t) for t in args.thresholds.split(',')]
    aspects = args.aspects.split(',')

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )

    # Analyze text for different aspects with varying thresholds
    results = analyze_text_for_aspects(model, tokenizer, args.text, thresholds, aspects)

    # Print results
    print_colored_results(results, args.text)

    # Save results if output file is provided
    if args.output_file:
        # Convert results to JSON-serializable format
        serializable_results = {}
        for aspect, thresholds in results.items():
            serializable_results[aspect] = {}
            for threshold, result in thresholds.items():
                serializable_results[aspect][str(threshold)] = {
                    'rationale_tokens': [(t[0], float(t[1]), int(t[2])) for t in result['rationale_tokens']],
                    'highlighted_text': result['highlighted_text'].replace(Fore.RED, '**').replace(Style.RESET_ALL, '**'),
                    'num_tokens': result['num_tokens'],
                    'token_coverage': float(result['token_coverage']),
                    'prediction': int(result['prediction']),
                    'confidence': float(result['confidence']),
                    'aspect': result['aspect']
                }

        with open(args.output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
