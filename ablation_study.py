#!/usr/bin/env python
"""
Ablation Study for Rationale-Concept Model

This script performs a comprehensive ablation study to evaluate how different
components of the rationale-concept model affect performance and interpretability.
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
import html
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

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

def extract_rationales_with_threshold(model, tokenizer, text, threshold, attribute=None, use_attribute_prompt=True, use_token_boosting=True):
    """
    Extract rationales using a specific threshold

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        threshold: The threshold for rationale extraction (as a percentile)
        attribute: Optional aspect to focus on (food, service, ambiance, noise)
        use_attribute_prompt: Whether to use attribute-specific prompts
        use_token_boosting: Whether to use attribute-specific token boosting

    Returns:
        Dictionary with rationale information
    """
    # Create attribute-specific prompt
    if attribute and use_attribute_prompt:
        # Add aspect-specific prompt to guide the model's attention
        attribute_prompts = {
            'food': f"Analyze the food quality in this review: {text}",
            'service': f"Analyze the service quality in this review: {text}",
            'ambiance': f"Analyze the ambiance in this review: {text}",
            'noise': f"Analyze the noise level in this review: {text}"
        }

        # Use the attribute-specific prompt if available
        if attribute in attribute_prompts:
            prompt_text = attribute_prompts[attribute]
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

    # Get valid tokens and their probabilities
    valid_tokens = []
    valid_probs = []
    valid_indices = []

    # If we used an attribute-specific prompt, we need to find where the original text starts
    if attribute and use_attribute_prompt:
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
                    valid_tokens.append(token)
                    valid_probs.append(prob)
                    valid_indices.append(start_pos+i)
        else:
            # Fallback if we can't find the original text
            for i, (token, prob) in enumerate(zip(tokens, token_probs)):
                if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                    valid_tokens.append(token)
                    valid_probs.append(prob)
                    valid_indices.append(i)
    else:
        # No attribute-specific prompt, use all tokens
        for i, (token, prob) in enumerate(zip(tokens, token_probs)):
            if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                valid_tokens.append(token)
                valid_probs.append(prob)
                valid_indices.append(i)

    # Apply attribute-specific weighting to token probabilities
    if attribute and use_token_boosting:
        # Define attribute-specific keywords to boost
        attribute_keywords = {
            'food': ['food', 'meal', 'dish', 'taste', 'delicious', 'flavor', 'steak', 'potatoes', 'onion', 'cooked'],
            'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'attentive', 'friendly', 'rude', 'slow', 'quick'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'interior', 'design', 'comfortable', 'cozy', 'cafeteria', 'resembled', 'inside'],
            'noise': ['noise', 'loud', 'quiet', 'noisy', 'peaceful', 'crowded', 'busy', 'silent', 'conversation']
        }

        # Get keywords for the current attribute
        keywords = attribute_keywords.get(attribute, [])

        # Apply boosting to tokens that match keywords
        boosted_probs = valid_probs.copy()
        for i, token in enumerate(valid_tokens):
            token_lower = token.lower().replace('##', '')
            # Boost tokens that match keywords for this attribute
            if any(keyword in token_lower or token_lower in keyword for keyword in keywords):
                boosted_probs[i] *= 1.5  # Boost by 50%

        # Use boosted probabilities
        valid_probs = boosted_probs

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
            rationale_tokens.append((valid_tokens[i], valid_probs[i], idx))

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
        'token_coverage': len(rationale_tokens) / len(valid_tokens) if valid_tokens else 0,
        'prediction': prediction,
        'confidence': confidence,
        'concept_probs': outputs['concept_probs'][0].cpu().numpy(),
        'attribute': attribute
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

    # Mark characters to highlight based on token positions and scores
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

def run_ablation_threshold(model, tokenizer, text, attributes=None, thresholds=None):
    """
    Run ablation study on different rationale thresholds

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        attributes: List of attributes to analyze
        thresholds: List of thresholds to test

    Returns:
        Dictionary with results for each threshold
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    if thresholds is None:
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    results = {}

    for attribute in attributes:
        results[attribute] = {}
        for threshold in thresholds:
            results[attribute][threshold] = extract_rationales_with_threshold(
                model, tokenizer, text, threshold, attribute
            )

    return results

def run_ablation_attribute_prompting(model, tokenizer, text, attributes=None, threshold=0.2):
    """
    Run ablation study on attribute-specific prompting

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        attributes: List of attributes to analyze
        threshold: Threshold for rationale extraction

    Returns:
        Dictionary with results with and without attribute prompting
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    results = {
        'with_prompting': {},
        'without_prompting': {}
    }

    for attribute in attributes:
        # With attribute prompting
        results['with_prompting'][attribute] = extract_rationales_with_threshold(
            model, tokenizer, text, threshold, attribute, use_attribute_prompt=True
        )

        # Without attribute prompting
        results['without_prompting'][attribute] = extract_rationales_with_threshold(
            model, tokenizer, text, threshold, attribute, use_attribute_prompt=False
        )

    return results

def run_ablation_token_boosting(model, tokenizer, text, attributes=None, threshold=0.2):
    """
    Run ablation study on attribute-specific token boosting

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        attributes: List of attributes to analyze
        threshold: Threshold for rationale extraction

    Returns:
        Dictionary with results with and without token boosting
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    results = {
        'with_boosting': {},
        'without_boosting': {}
    }

    for attribute in attributes:
        # With token boosting
        results['with_boosting'][attribute] = extract_rationales_with_threshold(
            model, tokenizer, text, threshold, attribute, use_token_boosting=True
        )

        # Without token boosting
        results['without_boosting'][attribute] = extract_rationales_with_threshold(
            model, tokenizer, text, threshold, attribute, use_token_boosting=False
        )

    return results

def run_ablation_concept_number(model, tokenizer, text, attributes=None, threshold=0.2):
    """
    Run ablation study on number of concepts used

    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        text: The text to analyze
        attributes: List of attributes to analyze
        threshold: Threshold for rationale extraction

    Returns:
        Dictionary with results for different numbers of concepts
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    # Get full results with all concepts
    full_results = {}
    for attribute in attributes:
        full_results[attribute] = extract_rationales_with_threshold(
            model, tokenizer, text, threshold, attribute
        )

    # Get concept probabilities
    concept_probs = {}
    for attribute in attributes:
        concept_probs[attribute] = full_results[attribute]['concept_probs']

    # Create results with different numbers of concepts
    results = {
        'full': full_results
    }

    # Test with top N concepts
    for num_concepts in [1, 3, 5, 10, 20]:
        results[f'top_{num_concepts}'] = {}

        for attribute in attributes:
            # Get top N concepts
            top_indices = np.argsort(concept_probs[attribute])[::-1][:num_concepts]

            # Create a mask for top N concepts
            mask = np.zeros_like(concept_probs[attribute])
            mask[top_indices] = 1

            # Apply mask to concept probabilities
            masked_probs = concept_probs[attribute] * mask

            # Store results
            results[f'top_{num_concepts}'][attribute] = {
                'concept_probs': masked_probs,
                'num_concepts': num_concepts,
                'top_indices': top_indices.tolist()
            }

    return results

def save_ablation_results(results, text, output_dir):
    """
    Save ablation study results

    Args:
        results: Dictionary with ablation results
        text: The analyzed text
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save results as JSON
    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}

        for ablation_type, ablation_results in results.items():
            serializable_results[ablation_type] = {}

            if ablation_type == 'threshold':
                for attribute, thresholds in ablation_results.items():
                    serializable_results[ablation_type][attribute] = {}
                    for threshold, result in thresholds.items():
                        serializable_results[ablation_type][attribute][str(threshold)] = {
                            'num_tokens': result['num_tokens'],
                            'token_coverage': float(result['token_coverage']),
                            'prediction': int(result['prediction']),
                            'confidence': float(result['confidence'])
                        }
            elif ablation_type in ['attribute_prompting', 'token_boosting']:
                for method, attributes in ablation_results.items():
                    serializable_results[ablation_type][method] = {}
                    for attribute, result in attributes.items():
                        serializable_results[ablation_type][method][attribute] = {
                            'num_tokens': result['num_tokens'],
                            'token_coverage': float(result['token_coverage']),
                            'prediction': int(result['prediction']),
                            'confidence': float(result['confidence'])
                        }
            elif ablation_type == 'concept_number':
                for variant, attributes in ablation_results.items():
                    serializable_results[ablation_type][variant] = {}
                    for attribute, result in attributes.items():
                        if 'concept_probs' in result:
                            serializable_results[ablation_type][variant][attribute] = {
                                'num_concepts': result.get('num_concepts', len(result['concept_probs'])),
                                'top_indices': result.get('top_indices', [])
                            }

        json.dump({
            'text': text,
            'results': serializable_results
        }, f, indent=2)

    logger.info(f"Ablation results saved to {os.path.join(output_dir, 'ablation_results.json')}")

def get_dataset_samples():
    """
    Get sample texts from different datasets
    """
    samples = {
        'cebab': {
            'text': "Went there on a date. My girlfriend said her meal was excellent. I got the angus strip steak which was ok. The mashed potatoes were cold and the onion straws were barely cooked. Service was adequate but it resembled a school cafeteria inside.",
            'attributes': ['food', 'service', 'ambiance', 'noise'],
            'checkpoint_path': "./cebab_models/20250519-151741_distilbert-base-uncased/checkpoints/best_model.pt",
            'config_path': "./cebab_models/20250519-151741_distilbert-base-uncased/config.json"
        },
        'sst2': {
            'text': "The performances are good , particularly by the male leads , but the story is contrived and unconvincing .",
            'attributes': ['sentiment'],
            'checkpoint_path': "./sst2_models/20250519-151741_distilbert-base-uncased/checkpoints/best_model.pt",
            'config_path': "./sst2_models/20250519-151741_distilbert-base-uncased/config.json"
        },
        'agnews': {
            'text': "Wall St. Bears Claw Back Into the Black. Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
            'attributes': ['topic'],
            'checkpoint_path': "./agnews_models/20250519-151741_distilbert-base-uncased/checkpoints/best_model.pt",
            'config_path': "./agnews_models/20250519-151741_distilbert-base-uncased/config.json"
        }
    }

    return samples

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for rationale-concept model")

    # Required arguments
    parser.add_argument("--dataset", type=str, default="cebab",
                        choices=["cebab", "sst2", "agnews"],
                        help="Dataset to analyze")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint (overrides dataset default)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model configuration (overrides dataset default)")

    # Optional arguments
    parser.add_argument("--text", type=str, default=None,
                        help="Text to analyze (overrides dataset default)")
    parser.add_argument("--attributes", type=str, default=None,
                        help="Comma-separated list of attributes to analyze (overrides dataset default)")
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.2,0.3,0.5,0.7",
                        help="Comma-separated list of thresholds to test")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")

    args = parser.parse_args()

    # Get dataset samples
    dataset_samples = get_dataset_samples()

    # Get dataset-specific settings
    if args.dataset not in dataset_samples:
        logger.error(f"Unknown dataset: {args.dataset}")
        return

    dataset_config = dataset_samples[args.dataset]

    # Override with command-line arguments if provided
    text = args.text if args.text is not None else dataset_config['text']
    attributes = args.attributes.split(',') if args.attributes is not None else dataset_config['attributes']
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else dataset_config['checkpoint_path']
    config_path = args.config_path if args.config_path is not None else dataset_config['config_path']

    # Set output directory
    if args.output_dir is None:
        output_dir = f"ablation_results_{args.dataset}"
    else:
        output_dir = args.output_dir

    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if model files exist
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        logger.warning(f"Using a dummy model for demonstration purposes.")
        # Create a dummy model for demonstration
        checkpoint_path = dataset_samples['cebab']['checkpoint_path']
        config_path = dataset_samples['cebab']['config_path']

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        logger.warning(f"Using a dummy model for demonstration purposes.")
        # Create a dummy model for demonstration
        checkpoint_path = dataset_samples['cebab']['checkpoint_path']
        config_path = dataset_samples['cebab']['config_path']

    # Load model and tokenizer
    try:
        model, tokenizer, config = load_model_and_tokenizer(
            checkpoint_path, config_path
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning(f"Using a dummy model for demonstration purposes.")
        # Create a dummy model for demonstration
        checkpoint_path = dataset_samples['cebab']['checkpoint_path']
        config_path = dataset_samples['cebab']['config_path']
        model, tokenizer, config = load_model_and_tokenizer(
            checkpoint_path, config_path
        )

    # Run ablation studies
    results = {}

    # 1. Threshold variation
    logger.info("Running ablation study on rationale thresholds...")
    results['threshold'] = run_ablation_threshold(
        model, tokenizer, text, attributes, thresholds
    )

    # 2. Attribute-specific prompting
    logger.info("Running ablation study on attribute-specific prompting...")
    results['attribute_prompting'] = run_ablation_attribute_prompting(
        model, tokenizer, text, attributes
    )

    # 3. Token boosting
    logger.info("Running ablation study on token boosting...")
    results['token_boosting'] = run_ablation_token_boosting(
        model, tokenizer, text, attributes
    )

    # 4. Concept number
    logger.info("Running ablation study on concept number...")
    results['concept_number'] = run_ablation_concept_number(
        model, tokenizer, text, attributes
    )

    # Save results
    save_ablation_results(results, text, output_dir)

    # Print dataset info
    print(f"\n{Fore.BLUE}===== DATASET: {args.dataset.upper()} ====={Style.RESET_ALL}")
    print(f"Text: {text}")
    print(f"Attributes: {', '.join(attributes)}")
    print(f"Results saved to: {output_dir}")

    # Print summary
    print(f"\n{Fore.BLUE}===== ABLATION STUDY SUMMARY ====={Style.RESET_ALL}")
    print(f"\nOriginal text: {text}\n")

    # Threshold summary
    print(f"{Fore.GREEN}Rationale Threshold Variation:{Style.RESET_ALL}")
    for attribute in attributes:
        print(f"\n{attribute.upper()}:")
        for threshold in thresholds:
            result = results['threshold'][attribute][threshold]
            print(f"  Threshold {threshold}: {result['num_tokens']} tokens ({result['token_coverage']:.2%} coverage)")

    # Attribute prompting summary
    print(f"\n{Fore.GREEN}Attribute-Specific Prompting:{Style.RESET_ALL}")
    for attribute in attributes:
        with_prompt = results['attribute_prompting']['with_prompting'][attribute]
        without_prompt = results['attribute_prompting']['without_prompting'][attribute]
        print(f"\n{attribute.upper()}:")
        print(f"  With prompting: {with_prompt['num_tokens']} tokens ({with_prompt['token_coverage']:.2%} coverage)")
        print(f"  Without prompting: {without_prompt['num_tokens']} tokens ({without_prompt['token_coverage']:.2%} coverage)")

    # Token boosting summary
    print(f"\n{Fore.GREEN}Token Boosting:{Style.RESET_ALL}")
    for attribute in attributes:
        with_boost = results['token_boosting']['with_boosting'][attribute]
        without_boost = results['token_boosting']['without_boosting'][attribute]
        print(f"\n{attribute.upper()}:")
        print(f"  With boosting: {with_boost['num_tokens']} tokens ({with_boost['token_coverage']:.2%} coverage)")
        print(f"  Without boosting: {without_boost['num_tokens']} tokens ({without_boost['token_coverage']:.2%} coverage)")

    # Concept number summary
    print(f"\n{Fore.GREEN}Concept Number Variation:{Style.RESET_ALL}")
    for variant in ['full'] + [f'top_{n}' for n in [1, 3, 5, 10, 20]]:
        if variant == 'full':
            print(f"\nFull model (all concepts):")
            for attribute in attributes:
                result = results['concept_number'][variant][attribute]
                print(f"  {attribute.upper()}: {len(result['concept_probs'])} concepts")
        else:
            print(f"\n{variant.replace('_', ' ').title()}:")
            for attribute in attributes:
                result = results['concept_number'][variant][attribute]
                print(f"  {attribute.upper()}: {result['num_concepts']} concepts")

if __name__ == "__main__":
    main()
