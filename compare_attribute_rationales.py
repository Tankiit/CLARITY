#!/usr/bin/env python
"""
Compare Rationales and Concepts Across Attributes

This script compares how rationales are generated for different attributes
(food, service, ambiance, noise) and identifies the top concepts for each attribute.
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

def get_attribute_specific_rationales(model, tokenizer, text, attribute, percentile_threshold=0.2):
    """
    Get attribute-specific rationales

    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text
        attribute: The attribute to analyze (food, service, ambiance, noise)
        percentile_threshold: Percentile threshold for selecting tokens

    Returns:
        Dictionary with rationales and concept information
    """
    # Create attribute-specific prompt
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

    # Tokenize input
    inputs = tokenizer(
        prompt_text,
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

    # Find the original text in the tokenized input (for attribute-specific prompts)
    original_tokens = tokenizer.encode(text, add_special_tokens=False)
    prompt_tokens = inputs['input_ids'][0].cpu().numpy()

    # Find the start position of the original text in the prompt
    start_pos = -1
    for i in range(len(prompt_tokens) - len(original_tokens) + 1):
        if np.array_equal(prompt_tokens[i:i+len(original_tokens)], original_tokens):
            start_pos = i
            break

    # Get valid tokens and their probabilities
    valid_tokens = []
    valid_probs = []
    valid_indices = []

    if start_pos >= 0:
        # Only consider tokens from the original text
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

    # Apply attribute-specific weighting to token probabilities
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

    # Sort tokens by boosted probability
    token_importance = list(zip(valid_tokens, boosted_probs, valid_indices))
    token_importance.sort(key=lambda x: x[1], reverse=True)

    # Select top tokens based on percentile threshold
    num_tokens = len(token_importance)
    num_to_include = max(1, int(num_tokens * percentile_threshold))
    top_tokens = token_importance[:num_to_include]

    # Get top concepts
    top_concept_indices = np.argsort(concept_probs)[::-1][:5]  # Top 5 concepts
    top_concepts = [(f"concept_{idx}", float(concept_probs[idx])) for idx in top_concept_indices]

    # Create highlighted text
    highlighted_html = highlight_attribute_rationales(text, tokenizer, top_tokens, attribute)

    return {
        'attribute': attribute,
        'top_tokens': [(token, float(prob)) for token, prob, _ in top_tokens],
        'top_concepts': top_concepts,
        'highlighted_html': highlighted_html
    }

def highlight_attribute_rationales(text, tokenizer, rationale_tokens, attribute):
    """
    Highlight rationales in the original text with attribute-specific colors

    Args:
        text: Original text
        tokenizer: Tokenizer
        rationale_tokens: List of (token, importance, index) tuples
        attribute: The attribute being analyzed

    Returns:
        HTML string with highlighted rationales
    """
    # Define attribute-specific colors
    attribute_colors = {
        'food': 'rgba(255,0,0,{})',  # Red
        'service': 'rgba(0,128,0,{})',  # Green
        'ambiance': 'rgba(0,0,255,{})',  # Blue
        'noise': 'rgba(128,0,128,{})'  # Purple
    }

    color_format = attribute_colors.get(attribute, 'rgba(128,128,128,{})')  # Default gray

    # Tokenize the text to get token spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.offset_mapping
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)

    # Create a list of characters to highlight with their intensity
    highlight_scores = [0.0] * len(text)

    # Get max importance for normalization
    max_importance = max([importance for _, importance, _ in rationale_tokens]) if rationale_tokens else 1.0

    # Mark characters to highlight based on token positions and scores
    for token, importance, _ in rationale_tokens:
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
                html_parts.append(f'<span style="background-color: {color_format.format(score)}">')
                in_highlight = True
                current_intensity = intensity
        elif in_highlight:
            html_parts.append('</span>')
            in_highlight = False

        html_parts.append(html.escape(char))

    if in_highlight:
        html_parts.append('</span>')

    return ''.join(html_parts)

def compare_attributes(model, tokenizer, text, attributes=None, percentile_threshold=0.2):
    """
    Compare rationales and concepts across different attributes

    Args:
        model: The trained model
        tokenizer: The tokenizer
        text: The input text
        attributes: List of attributes to compare
        percentile_threshold: Percentile threshold for selecting tokens

    Returns:
        Dictionary with attribute-specific rationales and concepts
    """
    if attributes is None:
        attributes = ['food', 'service', 'ambiance', 'noise']

    results = {}

    for attribute in attributes:
        results[attribute] = get_attribute_specific_rationales(
            model, tokenizer, text, attribute, percentile_threshold
        )

    return results

def save_html_comparison(results, text, output_path):
    """
    Save attribute comparison as an HTML report with enhanced visualizations

    Args:
        results: Results from compare_attributes
        text: Original text
        output_path: Path to save the HTML report
    """
    # Create a combined visualization of all attributes
    combined_html = create_combined_visualization(results, text)

    # Create HTML header with styles
    html_header = """<!DOCTYPE html>
<html>
<head>
    <title>Detailed Attribute-Concept Visualization</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
        h1, h2, h3 { color: #2c3e50; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
        .attribute { flex: 1; min-width: 300px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .food { border-top: 5px solid #e74c3c; }
        .service { border-top: 5px solid #2ecc71; }
        .ambiance { border-top: 5px solid #3498db; }
        .noise { border-top: 5px solid #9b59b6; }
        .attribute-header { font-weight: bold; margin-bottom: 15px; font-size: 1.2em; }
        .rationale-text { padding: 15px; background-color: #f9f9f9; border-radius: 5px; margin-bottom: 15px; line-height: 1.8; }
        .token-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
        .token { padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }
        .token-food { background-color: rgba(231, 76, 60, 0.2); border: 1px solid rgba(231, 76, 60, 0.4); }
        .token-service { background-color: rgba(46, 204, 113, 0.2); border: 1px solid rgba(46, 204, 113, 0.4); }
        .token-ambiance { background-color: rgba(52, 152, 219, 0.2); border: 1px solid rgba(52, 152, 219, 0.4); }
        .token-noise { background-color: rgba(155, 89, 182, 0.2); border: 1px solid rgba(155, 89, 182, 0.4); }
        .concepts { margin-top: 15px; }
        .concept { margin-right: 10px; padding: 8px 12px; border-radius: 5px; display: inline-block; margin-bottom: 8px; }
        .concept-food { background-color: rgba(231, 76, 60, 0.1); border: 1px solid rgba(231, 76, 60, 0.3); }
        .concept-service { background-color: rgba(46, 204, 113, 0.1); border: 1px solid rgba(46, 204, 113, 0.3); }
        .concept-ambiance { background-color: rgba(52, 152, 219, 0.1); border: 1px solid rgba(52, 152, 219, 0.3); }
        .concept-noise { background-color: rgba(155, 89, 182, 0.1); border: 1px solid rgba(155, 89, 182, 0.3); }
        .combined-view { padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 30px; border: 1px solid #ddd; }
        .combined-text { line-height: 2; font-size: 1.1em; padding: 15px; background-color: white; border-radius: 5px; }
        .concept-matrix { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .concept-matrix th, .concept-matrix td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        .concept-matrix th { background-color: #f2f2f2; }
        .concept-matrix td.highlight { font-weight: bold; }
        .concept-matrix .heat-0 { background-color: #ffffff; }
        .concept-matrix .heat-1 { background-color: #fee5d9; }
        .concept-matrix .heat-2 { background-color: #fcbba1; }
        .concept-matrix .heat-3 { background-color: #fc9272; }
        .concept-matrix .heat-4 { background-color: #fb6a4a; }
        .concept-matrix .heat-5 { background-color: #de2d26; }
        .legend { display: flex; align-items: center; justify-content: center; margin: 15px 0; }
        .legend-item { display: flex; align-items: center; margin: 0 10px; }
        .legend-color { width: 20px; height: 20px; margin-right: 5px; border-radius: 3px; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border: 1px solid #ddd; border-bottom: none; border-radius: 5px 5px 0 0; margin-right: 5px; }
        .tab.active { background-color: #f8f9fa; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .token-importance { width: 100%; height: 30px; background-color: #f2f2f2; border-radius: 15px; margin-top: 5px; position: relative; overflow: hidden; }
        .token-importance-bar { height: 100%; position: absolute; left: 0; }
        .token-importance-label { position: absolute; right: 10px; top: 5px; font-size: 0.8em; }
    </style>
    <script>
        function switchTab(tabName) {
            /* Hide all tab contents */
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }

            /* Deactivate all tabs */
            const tabs = document.getElementsByClassName('tab');
            for (let i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }

            /* Activate selected tab and content */
            document.getElementById(tabName).classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }

        function highlightConcept(conceptId) {
            /* Reset all highlights */
            const cells = document.querySelectorAll('.concept-cell');
            for (let i = 0; i < cells.length; i++) {
                cells[i].style.outline = 'none';
            }

            /* Highlight cells for this concept */
            const conceptCells = document.querySelectorAll('.' + conceptId);
            for (let i = 0; i < conceptCells.length; i++) {
                conceptCells[i].style.outline = '3px solid #3498db';
            }
        }

        function resetHighlights() {
            const cells = document.querySelectorAll('.concept-cell');
            for (let i = 0; i < cells.length; i++) {
                cells[i].style.outline = 'none';
            }
        }
    </script>
</head>
<body>
    <h1>Detailed Attribute-Concept Visualization</h1>

    <div class="tabs">
        <div id="combined-tab" class="tab active" onclick="switchTab('combined')">Combined View</div>
        <div id="individual-tab" class="tab" onclick="switchTab('individual')">Individual Attributes</div>
        <div id="concept-tab" class="tab" onclick="switchTab('concept')">Concept Analysis</div>
        <div id="token-tab" class="tab" onclick="switchTab('token')">Token Importance</div>
    </div>

    <div id="combined" class="tab-content active">
        <h2>Combined Attribute Visualization</h2>
        <p>This view shows how different parts of the text relate to different attributes (food, service, ambiance, noise).</p>

        <div class="combined-view">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(231, 76, 60, 0.5);"></div>
                    <span>Food</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(46, 204, 113, 0.5);"></div>
                    <span>Service</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(52, 152, 219, 0.5);"></div>
                    <span>Ambiance</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(155, 89, 182, 0.5);"></div>
                    <span>Noise</span>
                </div>
            </div>

            <div class="combined-text">
"""

    # Create the combined HTML content
    html_content = html_header + combined_html + """
            </div>
        </div>
    </div>

    <div id="individual" class="tab-content">
        <h2>Individual Attribute Analysis</h2>
        <p>This view shows how each attribute highlights different parts of the text.</p>

        <div class="container">
"""

    for attribute, data in results.items():
        html_content += f"""
                <div class="attribute {attribute}">
                    <div class="attribute-header">{attribute.upper()}</div>
                    <div class="rationale-text">
                        {data['highlighted_html']}
                    </div>
                    <h3>Top Tokens:</h3>
                    <div class="token-list">
                        {' '.join([f'<span class="token token-{attribute}">{token} ({importance:.4f})</span>' for token, importance in data['top_tokens'][:8]])}
                    </div>
                    <h3>Top Concepts:</h3>
                    <div class="concepts">
                        {' '.join([f'<span class="concept concept-{attribute}">{concept} ({prob:.4f})</span>' for concept, prob in data['top_concepts'][:5]])}
                    </div>
                </div>
        """

    html_content += """
            </div>
        </div>

        <div id="concept" class="tab-content">
            <h2>Concept-Attribute Alignment Matrix</h2>
            <p>This matrix shows how different concepts align with different attributes. Hover over a concept to highlight it across all attributes.</p>
    """

    # Collect all concepts
    all_concepts = {}
    for attribute, data in results.items():
        for concept, prob in data['top_concepts']:
            if concept not in all_concepts:
                all_concepts[concept] = {'attributes': [], 'probabilities': {}}
            all_concepts[concept]['attributes'].append(attribute)
            all_concepts[concept]['probabilities'][attribute] = prob

    # Sort concepts by number of attributes they appear in
    sorted_concepts = sorted(all_concepts.items(),
                            key=lambda x: (len(x[1]['attributes']),
                                          sum(x[1]['probabilities'].values())),
                            reverse=True)

    # Create concept matrix
    html_content += """
            <table class="concept-matrix">
                <tr>
                    <th>Concept</th>
                    <th>Food</th>
                    <th>Service</th>
                    <th>Ambiance</th>
                    <th>Noise</th>
                    <th>Attributes</th>
                </tr>
    """

    for concept, data in sorted_concepts:
        concept_id = concept.replace('_', '-')
        html_content += f"""
                <tr onmouseover="highlightConcept('{concept_id}')" onmouseout="resetHighlights()">
                    <td><strong>{concept}</strong></td>
        """

        for attribute in ['food', 'service', 'ambiance', 'noise']:
            if attribute in data['probabilities']:
                prob = data['probabilities'][attribute]
                # Determine heat level based on probability
                heat_level = min(5, int(prob * 6))
                html_content += f"""
                    <td class="concept-cell {concept_id} heat-{heat_level}">{prob:.4f}</td>
                """
            else:
                html_content += """
                    <td class="concept-cell heat-0">-</td>
                """

        html_content += f"""
                    <td>{len(data['attributes'])}</td>
                </tr>
        """

    html_content += """
            </table>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color heat-0"></div>
                    <span>0.0</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color heat-1"></div>
                    <span>0.2</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color heat-2"></div>
                    <span>0.4</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color heat-3"></div>
                    <span>0.6</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color heat-4"></div>
                    <span>0.8</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color heat-5"></div>
                    <span>1.0</span>
                </div>
            </div>

            <h3>Concept Overlap Analysis</h3>
    """

    # Display concepts by number of attributes
    for num_attrs in range(len(results), 0, -1):
        concepts_with_n_attrs = [c for c, data in sorted_concepts if len(data['attributes']) == num_attrs]

        if concepts_with_n_attrs:
            if num_attrs == 1:
                html_content += f"<h4>Concepts unique to a single attribute:</h4>"
            else:
                html_content += f"<h4>Concepts appearing in {num_attrs} attributes:</h4>"

            html_content += "<ul>"
            for concept in concepts_with_n_attrs:
                concept_data = all_concepts[concept]
                attrs_str = ', '.join(concept_data['attributes'])
                html_content += f"""
                    <li>
                        <strong>{concept}</strong> - Attributes: {attrs_str}
                        <div style="margin-top: 5px;">
                """

                for attr in concept_data['attributes']:
                    prob = concept_data['probabilities'][attr]
                    width = int(prob * 100)
                    html_content += f"""
                            <div style="margin-bottom: 5px;">
                                <span style="display: inline-block; width: 80px;">{attr}:</span>
                                <div class="token-importance">
                                    <div class="token-importance-bar" style="width: {width}%; background-color: rgba(52, 152, 219, 0.7);"></div>
                                    <span class="token-importance-label">{prob:.4f}</span>
                                </div>
                            </div>
                    """

                html_content += """
                        </div>
                    </li>
                """
            html_content += "</ul>"

    html_content += """
        </div>

        <div id="token" class="tab-content">
            <h2>Token Importance Analysis</h2>
            <p>This view shows the importance of each token for different attributes.</p>
    """

    # Collect all unique tokens across attributes
    all_tokens = {}
    for attribute, data in results.items():
        for token, importance in data['top_tokens']:
            if token not in all_tokens:
                all_tokens[token] = {}
            all_tokens[token][attribute] = importance

    # Sort tokens by average importance
    sorted_tokens = []
    for token, attr_scores in all_tokens.items():
        avg_score = sum(attr_scores.values()) / len(attr_scores)
        sorted_tokens.append((token, avg_score, attr_scores))

    sorted_tokens.sort(key=lambda x: x[1], reverse=True)

    # Display token importance
    html_content += """
            <table class="concept-matrix">
                <tr>
                    <th>Token</th>
                    <th>Food</th>
                    <th>Service</th>
                    <th>Ambiance</th>
                    <th>Noise</th>
                    <th>Average</th>
                </tr>
    """

    for token, avg_score, attr_scores in sorted_tokens[:20]:  # Show top 20 tokens
        html_content += f"""
                <tr>
                    <td><strong>{token}</strong></td>
        """

        for attribute in ['food', 'service', 'ambiance', 'noise']:
            if attribute in attr_scores:
                importance = attr_scores[attribute]
                # Determine heat level based on importance
                heat_level = min(5, int(importance / 0.01))  # Adjust based on your importance scale
                html_content += f"""
                    <td class="heat-{heat_level}">{importance:.4f}</td>
                """
            else:
                html_content += """
                    <td class="heat-0">-</td>
                """

        html_content += f"""
                    <td>{avg_score:.4f}</td>
                </tr>
        """

    html_content += """
            </table>

            <h3>Token Importance Visualization</h3>
            <p>This visualization shows the importance of each token for different attributes.</p>
    """

    # Display token importance bars
    for token, avg_score, attr_scores in sorted_tokens[:15]:  # Show top 15 tokens
        html_content += f"""
            <div style="margin-bottom: 20px;">
                <h4>{token}</h4>
        """

        for attribute in ['food', 'service', 'ambiance', 'noise']:
            if attribute in attr_scores:
                importance = attr_scores[attribute]
                width = int(importance * 5000)  # Scale up for better visualization

                # Get attribute color
                attr_colors = {
                    'food': 'rgba(231, 76, 60, 0.7)',
                    'service': 'rgba(46, 204, 113, 0.7)',
                    'ambiance': 'rgba(52, 152, 219, 0.7)',
                    'noise': 'rgba(155, 89, 182, 0.7)'
                }
                color = attr_colors.get(attribute, 'rgba(128, 128, 128, 0.7)')

                html_content += f"""
                <div style="margin-bottom: 5px;">
                    <span style="display: inline-block; width: 80px;">{attribute}:</span>
                    <div class="token-importance">
                        <div class="token-importance-bar" style="width: {width}%; background-color: {color};"></div>
                        <span class="token-importance-label">{importance:.4f}</span>
                    </div>
                </div>
                """
            else:
                html_content += f"""
                <div style="margin-bottom: 5px;">
                    <span style="display: inline-block; width: 80px;">{attribute}:</span>
                    <div class="token-importance">
                        <span class="token-importance-label">0.0000</span>
                    </div>
                </div>
                """

        html_content += """
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Enhanced HTML visualization saved to {output_path}")

def create_combined_visualization(results, text):
    """
    Create a combined visualization of all attributes

    Args:
        results: Results from compare_attributes
        text: Original text

    Returns:
        HTML string with combined visualization
    """
    # Tokenize the text to get character positions
    words = []
    word_positions = []
    current_word = ""
    current_start = 0

    for i, char in enumerate(text):
        if char.isalnum() or char == "'":
            if not current_word:
                current_start = i
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                word_positions.append((current_start, i - 1))
                current_word = ""
            words.append(char)
            word_positions.append((i, i))

    if current_word:
        words.append(current_word)
        word_positions.append((current_start, len(text) - 1))

    # Create attribute-specific importance scores for each word
    word_importance = {}
    for attribute, data in results.items():
        word_importance[attribute] = [0.0] * len(words)

        # Get token importance
        token_importance = {token: importance for token, importance in data['top_tokens']}

        # Map token importance to words
        for i, word in enumerate(words):
            word_lower = word.lower()
            for token, importance in token_importance.items():
                token_lower = token.lower().replace('##', '')
                if token_lower in word_lower or word_lower in token_lower:
                    word_importance[attribute][i] = max(word_importance[attribute][i], importance)

    # Create HTML with multi-color highlighting
    html_parts = []

    for i, word in enumerate(words):
        # Get importance for each attribute
        food_imp = word_importance.get('food', [0.0] * len(words))[i]
        service_imp = word_importance.get('service', [0.0] * len(words))[i]
        ambiance_imp = word_importance.get('ambiance', [0.0] * len(words))[i]
        noise_imp = word_importance.get('noise', [0.0] * len(words))[i]

        # Normalize importance values
        max_imp = max(food_imp, service_imp, ambiance_imp, noise_imp)
        if max_imp > 0:
            food_imp = food_imp / max_imp
            service_imp = service_imp / max_imp
            ambiance_imp = ambiance_imp / max_imp
            noise_imp = noise_imp / max_imp

        # Create background with multiple colors
        style = ""
        if max_imp > 0:
            # Create gradient background
            gradient_parts = []
            if food_imp > 0.1:
                gradient_parts.append(f"rgba(231, 76, 60, {food_imp * 0.5})")
            if service_imp > 0.1:
                gradient_parts.append(f"rgba(46, 204, 113, {service_imp * 0.5})")
            if ambiance_imp > 0.1:
                gradient_parts.append(f"rgba(52, 152, 219, {ambiance_imp * 0.5})")
            if noise_imp > 0.1:
                gradient_parts.append(f"rgba(155, 89, 182, {noise_imp * 0.5})")

            if gradient_parts:
                if len(gradient_parts) == 1:
                    style = f"background-color: {gradient_parts[0]};"
                else:
                    gradient = ", ".join(gradient_parts)
                    style = f"background: linear-gradient(to right, {gradient});"

        # Add border if this word is important for any attribute
        if max_imp > 0.3:
            style += "border-bottom: 2px dotted #333;"

        # Add tooltip with importance values
        tooltip = f"Food: {food_imp:.2f}, Service: {service_imp:.2f}, Ambiance: {ambiance_imp:.2f}, Noise: {noise_imp:.2f}"

        if style:
            html_parts.append(f'<span style="{style}" title="{tooltip}">{html.escape(word)}</span>')
        else:
            html_parts.append(html.escape(word))

    return ''.join(html_parts)

    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"HTML comparison saved to {output_path}")

def print_attribute_comparison(results, text):
    """
    Print attribute comparison in a readable format
    """
    print(f"\n{Fore.BLUE}===== ATTRIBUTE RATIONALES COMPARISON ====={Style.RESET_ALL}")
    print(f"\nOriginal text: {text}\n")

    attribute_colors = {
        'food': Fore.RED,
        'service': Fore.GREEN,
        'ambiance': Fore.BLUE,
        'noise': Fore.MAGENTA
    }

    for attribute, data in results.items():
        color = attribute_colors.get(attribute, Fore.WHITE)
        print(f"{color}{attribute.upper()}{Style.RESET_ALL}")

        print("Top tokens:")
        for token, importance in data['top_tokens'][:5]:
            print(f"  {token}: {importance:.4f}")

        print("Top concepts:")
        for concept, prob in data['top_concepts'][:3]:
            print(f"  {concept}: {prob:.4f}")

        print()

    # Print concept overlap analysis
    print(f"{Fore.BLUE}Concept Overlap Analysis:{Style.RESET_ALL}")

    # Collect all concepts
    all_concepts = {}
    for attribute, data in results.items():
        for concept, prob in data['top_concepts']:
            if concept not in all_concepts:
                all_concepts[concept] = {'attributes': [], 'probabilities': {}}
            all_concepts[concept]['attributes'].append(attribute)
            all_concepts[concept]['probabilities'][attribute] = prob

    # Sort concepts by number of attributes they appear in
    sorted_concepts = sorted(all_concepts.items(),
                            key=lambda x: (len(x[1]['attributes']),
                                          sum(x[1]['probabilities'].values())),
                            reverse=True)

    # Display concepts by number of attributes
    for num_attrs in range(len(results), 0, -1):
        concepts_with_n_attrs = [c for c, data in sorted_concepts if len(data['attributes']) == num_attrs]

        if concepts_with_n_attrs:
            if num_attrs == 1:
                print(f"\nConcepts unique to a single attribute:")
            else:
                print(f"\nConcepts appearing in {num_attrs} attributes:")

            for concept in concepts_with_n_attrs:
                concept_data = all_concepts[concept]
                attrs_str = ', '.join(concept_data['attributes'])
                print(f"  {concept} - Attributes: {attrs_str}")
                for attr in concept_data['attributes']:
                    color = attribute_colors.get(attr, Fore.WHITE)
                    print(f"    {color}{attr}: {concept_data['probabilities'][attr]:.4f}{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Compare rationales and concepts across attributes")

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
                        help="Comma-separated list of attributes to compare")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Percentile threshold for selecting tokens")
    parser.add_argument("--output_dir", type=str, default="attribute_comparison_results",
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

    # Compare attributes
    results = compare_attributes(
        model, tokenizer, args.text, attributes, args.threshold
    )

    # Print comparison
    print_attribute_comparison(results, args.text)

    # Save results
    if args.output_dir:
        # Save HTML comparison
        html_path = os.path.join(args.output_dir, 'attribute_comparison.html')
        save_html_comparison(results, args.text, html_path)

        # Save JSON results
        json_path = os.path.join(args.output_dir, 'attribute_comparison.json')

        # Convert to JSON-serializable format
        json_results = {
            'text': args.text,
            'attributes': {}
        }

        for attribute, data in results.items():
            json_results['attributes'][attribute] = {
                'top_tokens': data['top_tokens'],
                'top_concepts': data['top_concepts']
            }

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {json_path}")

if __name__ == "__main__":
    main()
