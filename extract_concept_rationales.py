#!/usr/bin/env python
"""
Extract Rationales for Each Concept Based on Token Percentile

This script demonstrates how to extract rationales for each concept
using a token percentile approach.
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

def get_concept_token_importance(model, tokenizer, text, concept_idx, method='gradient'):
    """
    Calculate the importance of each token for a specific concept
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text
        concept_idx: The index of the concept to analyze
        method: Method to calculate token importance ('gradient', 'attention', or 'intervention')
        
    Returns:
        List of (token, importance_score) pairs
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
    
    if method == 'gradient':
        # Gradient-based approach
        model.zero_grad()
        
        # Get embeddings with gradient tracking
        token_embeddings = model.encoder.embeddings.word_embeddings(inputs['input_ids'])
        token_embeddings.requires_grad_(True)
        
        # Forward pass with embeddings
        outputs = model(
            inputs_embeds=token_embeddings,
            attention_mask=inputs['attention_mask']
        )
        
        # Get concept probability
        concept_prob = outputs['concept_probs'][0, concept_idx]
        
        # Compute gradient of concept probability with respect to token embeddings
        concept_prob.backward()
        
        # Get gradient magnitudes as importance scores
        token_importance = torch.norm(token_embeddings.grad, dim=2)[0].detach().cpu().numpy()
        
    elif method == 'attention':
        # Attention-based approach
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )
        
        # Use attention weights as importance scores
        # For simplicity, use the last layer's attention weights
        if 'attentions' in outputs:
            # Average across attention heads
            attention = outputs['attentions'][-1].mean(dim=1)[0].cpu().numpy()
            # Use attention from CLS token to all other tokens
            token_importance = attention[0]
        else:
            # Fallback to token probabilities
            token_importance = outputs['token_probs'][0].cpu().numpy()
    
    elif method == 'intervention':
        # Intervention-based approach
        with torch.no_grad():
            # Get baseline concept probability
            baseline_outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            baseline_prob = baseline_outputs['concept_probs'][0, concept_idx].item()
            
            # Initialize importance scores
            token_importance = np.zeros(inputs['input_ids'].size(1))
            
            # For each token, mask it and see how it affects the concept probability
            for i in range(inputs['input_ids'].size(1)):
                if inputs['attention_mask'][0, i] == 0:
                    continue  # Skip padding tokens
                
                # Create a copy with this token masked
                masked_attention = inputs['attention_mask'].clone()
                masked_attention[0, i] = 0
                
                # Get prediction with masked token
                masked_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=masked_attention
                )
                masked_prob = masked_outputs['concept_probs'][0, concept_idx].item()
                
                # Calculate impact on concept probability
                token_importance[i] = abs(baseline_prob - masked_prob)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_mask = inputs['attention_mask'][0].cpu().numpy()
    
    # Create token importance pairs
    token_importance_pairs = []
    for i, (token, importance) in enumerate(zip(tokens, token_importance)):
        if attention_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            token_importance_pairs.append((token, float(importance)))
    
    return token_importance_pairs

def extract_concept_rationales_by_percentile(token_importance_pairs, percentile_threshold):
    """
    Extract rationales based on token percentile threshold
    
    Args:
        token_importance_pairs: List of (token, importance_score) pairs
        percentile_threshold: Percentile threshold (0-1) for selecting tokens
        
    Returns:
        List of (token, importance_score) pairs above the threshold
    """
    # Sort by importance
    sorted_pairs = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)
    
    # Calculate number of tokens to include based on percentile
    num_tokens = len(sorted_pairs)
    num_to_include = max(1, int(num_tokens * percentile_threshold))
    
    # Select top tokens
    rationale_tokens = sorted_pairs[:num_to_include]
    
    return rationale_tokens

def highlight_concept_rationales(text, tokenizer, rationale_tokens):
    """
    Highlight rationales in the original text
    
    Args:
        text: Original text
        tokenizer: Tokenizer
        rationale_tokens: List of (token, importance) pairs
        
    Returns:
        HTML string with highlighted rationales
    """
    # Tokenize the text to get token spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.offset_mapping
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    
    # Create a list of characters to highlight with their intensity
    highlight_scores = [0.0] * len(text)
    
    # Get max importance for normalization
    max_importance = max([importance for _, importance in rationale_tokens]) if rationale_tokens else 1.0
    
    # Mark characters to highlight based on token positions and scores
    for token, importance in rationale_tokens:
        # Normalize importance
        norm_importance = importance / max_importance
        
        # Find all occurrences of this token in the tokens list
        for i, t in enumerate(tokens):
            if t == token or (token.startswith('##') and t.endswith(token[2:])):
                start, end = offset_mapping[i]
                for j in range(start, end):
                    if j < len(highlight_scores):
                        highlight_scores[j] = max(highlight_scores[j], norm_importance)
    
    # Generate HTML with colored highlighting
    html_parts = []
    in_highlight = False
    current_intensity = 0
    
    for i, char in enumerate(text):
        score = highlight_scores[i]
        if score > 0.1:  # Threshold for highlighting
            intensity = min(255, int(score * 255))
            if not in_highlight or abs(intensity - current_intensity) > 20:
                if in_highlight:
                    html_parts.append('</span>')
                html_parts.append(f'<span style="background-color: rgba(255,0,0,{score:.2f});">')
                in_highlight = True
                current_intensity = intensity
        elif in_highlight:
            html_parts.append('</span>')
            in_highlight = False
        
        html_parts.append(html.escape(char))
    
    if in_highlight:
        html_parts.append('</span>')
    
    return ''.join(html_parts)

def analyze_concept_rationales(model, tokenizer, text, concept_indices=None, percentile_thresholds=None, method='gradient'):
    """
    Analyze rationales for multiple concepts with different percentile thresholds
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text
        concept_indices: List of concept indices to analyze (if None, use top activated concepts)
        percentile_thresholds: List of percentile thresholds to test
        method: Method to calculate token importance
        
    Returns:
        Dictionary with concept rationales at different thresholds
    """
    if percentile_thresholds is None:
        percentile_thresholds = [0.1, 0.2, 0.3, 0.5]
    
    # Tokenize input for initial forward pass
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
    
    # Get concept probabilities
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    concept_probs = outputs['concept_probs'][0].cpu().numpy()
    
    # If concept_indices not provided, use top activated concepts
    if concept_indices is None:
        # Get top 5 concepts
        concept_indices = np.argsort(concept_probs)[::-1][:5]
    
    # Analyze each concept
    results = {}
    
    for concept_idx in concept_indices:
        concept_name = f"concept_{concept_idx}"
        concept_prob = concept_probs[concept_idx]
        
        # Skip concepts with very low probability
        if concept_prob < 0.1:
            continue
        
        # Get token importance for this concept
        token_importance_pairs = get_concept_token_importance(
            model, tokenizer, text, concept_idx, method
        )
        
        # Extract rationales at different percentile thresholds
        concept_results = {
            'probability': float(concept_prob),
            'thresholds': {}
        }
        
        for threshold in percentile_thresholds:
            rationale_tokens = extract_concept_rationales_by_percentile(
                token_importance_pairs, threshold
            )
            
            # Generate highlighted text
            highlighted_html = highlight_concept_rationales(
                text, tokenizer, rationale_tokens
            )
            
            concept_results['thresholds'][threshold] = {
                'rationale_tokens': [(token, float(importance)) for token, importance in rationale_tokens],
                'highlighted_html': highlighted_html,
                'num_tokens': len(rationale_tokens)
            }
        
        results[concept_name] = concept_results
    
    return results

def save_html_report(results, text, output_path):
    """
    Save results as an HTML report
    
    Args:
        results: Results from analyze_concept_rationales
        text: Original text
        output_path: Path to save the HTML report
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Concept Rationales Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .concept {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .concept-header {{ font-weight: bold; margin-bottom: 10px; }}
            .threshold-section {{ margin-bottom: 15px; }}
            .rationale-text {{ padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Concept Rationales Analysis</h1>
        <h2>Original Text:</h2>
        <p>{html.escape(text)}</p>
        
        <h2>Concept Rationales:</h2>
    """
    
    # Sort concepts by probability
    sorted_concepts = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    for concept_name, concept_data in sorted_concepts:
        html_content += f"""
        <div class="concept">
            <div class="concept-header">{concept_name} (Probability: {concept_data['probability']:.4f})</div>
        """
        
        # Sort thresholds
        thresholds = sorted(concept_data['thresholds'].keys())
        
        for threshold in thresholds:
            threshold_data = concept_data['thresholds'][threshold]
            
            html_content += f"""
            <div class="threshold-section">
                <h3>Threshold: {threshold} (Top {int(threshold * 100)}%)</h3>
                <p>Number of tokens: {threshold_data['num_tokens']}</p>
                <p>Top tokens: {', '.join([f"{token} ({importance:.4f})" for token, importance in threshold_data['rationale_tokens'][:5]])}</p>
                <div class="rationale-text">
                    {threshold_data['highlighted_html']}
                </div>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract concept rationales based on token percentile")
    
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
    parser.add_argument("--concepts", type=str, default="",
                        help="Comma-separated list of concept indices to analyze (if empty, use top activated concepts)")
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.5",
                        help="Comma-separated list of percentile thresholds")
    parser.add_argument("--method", type=str, default="gradient",
                        choices=["gradient", "attention", "intervention"],
                        help="Method to calculate token importance")
    parser.add_argument("--output_dir", type=str, default="concept_rationale_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parse arguments
    percentile_thresholds = [float(t) for t in args.thresholds.split(',')]
    concept_indices = [int(c) for c in args.concepts.split(',')] if args.concepts else None
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )
    
    # Analyze concept rationales
    results = analyze_concept_rationales(
        model, tokenizer, args.text, concept_indices, percentile_thresholds, args.method
    )
    
    # Print results
    print(f"\n{Fore.BLUE}===== CONCEPT RATIONALES ====={Style.RESET_ALL}")
    print(f"\nOriginal text: {args.text}\n")
    
    for concept_name, concept_data in results.items():
        print(f"{Fore.GREEN}{concept_name} (Probability: {concept_data['probability']:.4f}){Style.RESET_ALL}")
        
        for threshold, threshold_data in concept_data['thresholds'].items():
            print(f"\nThreshold: {threshold} (Top {int(threshold * 100)}%)")
            print(f"Number of tokens: {threshold_data['num_tokens']}")
            print("Top tokens:")
            for token, importance in threshold_data['rationale_tokens'][:5]:
                print(f"  {token}: {importance:.4f}")
        
        print()
    
    # Save results
    if args.output_dir:
        # Save HTML report
        html_path = os.path.join(args.output_dir, 'concept_rationales.html')
        save_html_report(results, args.text, html_path)
        
        # Save JSON results
        json_path = os.path.join(args.output_dir, 'concept_rationales.json')
        
        # Convert to JSON-serializable format
        json_results = {
            'text': args.text,
            'concepts': {}
        }
        
        for concept_name, concept_data in results.items():
            json_results['concepts'][concept_name] = {
                'probability': concept_data['probability'],
                'thresholds': {}
            }
            
            for threshold, threshold_data in concept_data['thresholds'].items():
                json_results['concepts'][concept_name]['thresholds'][str(threshold)] = {
                    'rationale_tokens': threshold_data['rationale_tokens'],
                    'num_tokens': threshold_data['num_tokens']
                }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {json_path}")

if __name__ == "__main__":
    main()
