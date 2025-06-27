#!/usr/bin/env python
"""
Analyze Rationales by Aspect with Different Thresholds

This script demonstrates how different rationale thresholds affect the
extracted rationales for different aspects of restaurant reviews.
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

def extract_rationales_with_threshold(model, tokenizer, text, threshold):
    """
    Extract rationales using a specific threshold
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
    
    # Extract rationales based on threshold
    rationale_tokens = []
    for i, (token, prob) in enumerate(zip(tokens, token_probs)):
        if prob > threshold and attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            rationale_tokens.append((token, prob, i))
    
    # Sort by probability
    rationale_tokens.sort(key=lambda x: x[1], reverse=True)
    
    # Create highlighted text
    highlighted_text = highlight_rationales(text, tokenizer, rationale_tokens)
    
    return {
        'rationale_tokens': rationale_tokens,
        'highlighted_text': highlighted_text,
        'num_tokens': len(rationale_tokens),
        'token_coverage': len(rationale_tokens) / sum(attention_mask)
    }

def highlight_rationales(text, tokenizer, rationale_tokens, highlight_char='**'):
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
    
    # Build the highlighted text
    highlighted = []
    in_highlight = False
    
    for i, char in enumerate(text):
        if highlight_mask[i] and not in_highlight:
            highlighted.append(highlight_char)
            in_highlight = True
        elif not highlight_mask[i] and in_highlight:
            highlighted.append(highlight_char)
            in_highlight = False
        
        highlighted.append(char)
    
    # Close any open highlight
    if in_highlight:
        highlighted.append(highlight_char)
    
    return ''.join(highlighted)

def analyze_text_with_thresholds(model, tokenizer, text, thresholds):
    """
    Analyze text with multiple thresholds
    """
    results = {}
    
    for threshold in thresholds:
        results[threshold] = extract_rationales_with_threshold(model, tokenizer, text, threshold)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze rationales by aspect with different thresholds")
    
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
    parser.add_argument("--output_file", type=str,
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )
    
    # Analyze text with different thresholds
    results = analyze_text_with_thresholds(model, tokenizer, args.text, thresholds)
    
    # Print results
    print(f"\nAnalyzing text: \"{args.text}\"\n")
    print("Results by threshold:")
    
    for threshold, result in results.items():
        print(f"\nThreshold: {threshold}")
        print(f"Number of rationale tokens: {result['num_tokens']}")
        print(f"Token coverage: {result['token_coverage']:.2%}")
        print(f"Highlighted text: {result['highlighted_text']}")
    
    # Save results if output file is provided
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
