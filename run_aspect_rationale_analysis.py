#!/usr/bin/env python
"""
Run Aspect Rationale Analysis

This script runs the aspect rationale analysis on a sample text with different
thresholds for each aspect (food, service, ambiance, noise).
"""

import os
import argparse
import json
import logging
import torch
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from analyze_aspect_rationales import load_model_and_tokenizer, analyze_text_for_aspects

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def visualize_aspect_rationales(results, text, output_dir=None):
    """
    Visualize rationales for different aspects and thresholds

    Args:
        results: Results from analyze_text_for_aspects
        text: Original text
        output_dir: Directory to save visualizations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a figure for each aspect
    for aspect, thresholds in results.items():
        plt.figure(figsize=(12, 8))

        # Create a heatmap-like visualization
        words = text.split()
        thresholds_list = sorted(thresholds.keys())

        # Create a matrix of word importance for each threshold
        importance_matrix = np.zeros((len(thresholds_list), len(words)))

        for i, threshold in enumerate(thresholds_list):
            result = thresholds[threshold]

            # Create a mapping from tokens to importance
            token_importance = {token: score for token, score, _ in result['rationale_tokens']}

            # Map words to importance
            for j, word in enumerate(words):
                # Check if any token is part of this word
                for token in token_importance:
                    if token.lower().replace('##', '') in word.lower():
                        importance_matrix[i, j] = token_importance[token]

        # Create a DataFrame for visualization
        df = pd.DataFrame(importance_matrix,
                         index=[f'Threshold {t}' for t in thresholds_list],
                         columns=words)

        # Plot heatmap
        plt.figure(figsize=(20, 6))
        plt.imshow(df.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Token Importance')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.yticks(range(len(thresholds_list)), df.index)
        plt.title(f'Rationale Importance by Threshold for {aspect.capitalize()} Aspect')
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{aspect}_rationales.png'))
        else:
            plt.show()

        plt.close()

def print_aspect_comparison(results, text):
    """
    Print a comparison of rationales for different aspects

    Args:
        results: Results from analyze_text_for_aspects
        text: Original text
    """
    print(f"\n{Fore.BLUE}===== ASPECT COMPARISON ====={Style.RESET_ALL}")
    print(f"\nOriginal text: {text}\n")

    # Choose a middle threshold for comparison
    thresholds = list(next(iter(results.values())).keys())
    middle_threshold = thresholds[len(thresholds) // 2]

    print(f"Comparing aspects at threshold {middle_threshold}:\n")

    # Print highlighted text for each aspect
    for aspect, thresholds_dict in results.items():
        result = thresholds_dict[middle_threshold]
        print(f"{Fore.GREEN}{aspect.upper()}:{Style.RESET_ALL} {result['highlighted_text']}")
        print(f"  Token coverage: {result['token_coverage']:.2%}")
        print()

    # Print a summary of which words are important for which aspects
    print(f"\n{Fore.BLUE}Word importance by aspect:{Style.RESET_ALL}")

    # Get all unique tokens across aspects
    all_tokens = set()
    for aspect, thresholds_dict in results.items():
        result = thresholds_dict[middle_threshold]
        all_tokens.update([token for token, _, _ in result['rationale_tokens']])

    # Create a mapping of tokens to aspects
    token_aspects = {token: [] for token in all_tokens}

    for aspect, thresholds_dict in results.items():
        result = thresholds_dict[middle_threshold]
        for token, _, _ in result['rationale_tokens']:
            token_aspects[token].append(aspect)

    # Print tokens by number of aspects they appear in
    for num_aspects in range(len(results), 0, -1):
        tokens_with_n_aspects = [token for token, aspects in token_aspects.items()
                                if len(aspects) == num_aspects]

        if tokens_with_n_aspects:
            if num_aspects == 1:
                print(f"\nTokens unique to a single aspect:")
            else:
                print(f"\nTokens important for {num_aspects} aspects:")

            for token in sorted(tokens_with_n_aspects):
                aspects_str = ', '.join(token_aspects[token])
                print(f"  {token}: {aspects_str}")

def analyze_token_probabilities(model, tokenizer, text):
    """
    Analyze raw token probabilities to understand the distribution
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

    # Get token probabilities
    token_probs = outputs['token_probs'][0].cpu().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_mask = inputs['attention_mask'][0].cpu().numpy()

    # Print token probabilities
    print("\nToken probabilities:")
    valid_probs = []

    for i, (token, prob) in enumerate(zip(tokens, token_probs)):
        if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            print(f"  {token}: {prob:.6f}")
            valid_probs.append(prob)

    # Print statistics
    if valid_probs:
        print("\nProbability statistics:")
        print(f"  Min: {min(valid_probs):.6f}")
        print(f"  Max: {max(valid_probs):.6f}")
        print(f"  Mean: {np.mean(valid_probs):.6f}")
        print(f"  Median: {np.median(valid_probs):.6f}")

        # Suggest thresholds
        percentiles = [50, 75, 90, 95, 99]
        print("\nSuggested thresholds (percentiles):")
        for p in percentiles:
            threshold = np.percentile(valid_probs, p)
            print(f"  {p}th percentile: {threshold:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Run aspect rationale analysis")

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
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.2,0.3,0.5,0.7",
                        help="Comma-separated list of thresholds to test (as percentiles, e.g., 0.1 means top 10%)")
    parser.add_argument("--aspects", type=str, default="food,service,ambiance,noise",
                        help="Comma-separated list of aspects to analyze")
    parser.add_argument("--output_dir", type=str, default="aspect_rationale_results",
                        help="Directory to save results and visualizations")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--analyze_probs", action="store_true",
                        help="Analyze token probabilities")

    args = parser.parse_args()

    # Parse thresholds and aspects
    thresholds = [float(t) for t in args.thresholds.split(',')]
    aspects = args.aspects.split(',')

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )

    # Analyze token probabilities if requested
    if args.analyze_probs:
        analyze_token_probabilities(model, tokenizer, args.text)

    # Analyze text for different aspects with varying thresholds
    results = analyze_text_for_aspects(model, tokenizer, args.text, thresholds, aspects)

    # Print aspect comparison
    print_aspect_comparison(results, args.text)

    # Generate visualizations if requested
    if args.visualize:
        visualize_aspect_rationales(results, args.text, args.output_dir)

    # Save results
    if args.output_dir:
        # Convert results to JSON-serializable format
        serializable_results = {}
        for aspect, thresholds_dict in results.items():
            serializable_results[aspect] = {}
            for threshold, result in thresholds_dict.items():
                serializable_results[aspect][str(threshold)] = {
                    'rationale_tokens': [(t[0], float(t[1]), int(t[2])) for t in result['rationale_tokens']],
                    'highlighted_text': result['highlighted_text'].replace(Fore.RED, '**').replace(Style.RESET_ALL, '**'),
                    'num_tokens': result['num_tokens'],
                    'token_coverage': float(result['token_coverage']),
                    'prediction': int(result['prediction']),
                    'confidence': float(result['confidence']),
                    'aspect': result['aspect']
                }

        output_file = os.path.join(args.output_dir, 'aspect_rationale_results.json')
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
