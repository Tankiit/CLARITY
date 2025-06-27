#!/usr/bin/env python
"""
Extract Rationales and Concepts from Trained Model

This script loads a trained RationaleConceptBottleneckModel and extracts
rationales and concepts from input text without relying on supervisory signals.
"""

import os
import argparse
import torch
import numpy as np
import json
import logging
from transformers import AutoTokenizer
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

def load_model_and_tokenizer(checkpoint_path, config_path):
    """
    Load the trained model and tokenizer

    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the model configuration

    Returns:
        model: Loaded model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Filter out parameters that are not in ModelConfig
    # This handles additional parameters like gradient_accumulation_steps
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

def extract_rationales_and_concepts(model, tokenizer, text, top_k_concepts=5, rationale_threshold=0.1):
    """
    Extract rationales and concepts from input text

    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        text: Input text
        top_k_concepts: Number of top concepts to return
        rationale_threshold: Threshold for rationale extraction

    Returns:
        Dictionary with prediction, rationales, and concepts
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
    model.zero_grad()
    with torch.no_grad():
        # Try to get attention weights if the model supports it
        try:
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True  # Request attention weights
            )
        except TypeError:
            # If model doesn't support output_attentions, call without it
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Try to extract attention from the model's base model if possible
            try:
                # For models based on transformers, we can try to get attention from the base model
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
                    # For DistilBERT
                    base_outputs = model.base_model.transformer(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_attentions=True
                    )
                    if 'attentions' in base_outputs:
                        outputs['attentions'] = base_outputs['attentions']
            except:
                # If this fails, we'll fall back to other methods
                pass

    # Get prediction
    logits = outputs['logits']
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_class = torch.argmax(logits, dim=1).item()
    confidence = probs[0, pred_class].item()

    # Extract rationales using attention weights
    rationales = []

    # Check if attention weights are available
    if 'attentions' in outputs:
        # Get attention weights from the last layer
        # For DistilBERT, there are 6 layers, and we take the last one
        attention = outputs['attentions'][-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Average across attention heads
        attention = attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]

        # Get attention from [CLS] token to all other tokens
        # This represents how much each token contributes to the final classification
        cls_attention = attention[0, 0, :]  # Shape: [seq_len]

        # Convert to numpy for easier processing
        attention_scores = cls_attention.detach().cpu().numpy()

        # Get token IDs and attention mask
        token_ids = inputs['input_ids'][0].detach().cpu().numpy()
        attention_mask = inputs['attention_mask'][0].detach().cpu().numpy()

        # Get tokens from IDs
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Find rationales (tokens with high attention scores)
        mask = attention_mask.astype(bool)
        scores = attention_scores[mask]
        valid_tokens = [t for t, m in zip(tokens, mask) if m]

        # Normalize scores
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            # Lower the threshold to capture more rationales
            adjusted_threshold = max(0.05, rationale_threshold * 0.5)

            # Extract rationales above threshold
            for i, (token, score) in enumerate(zip(valid_tokens, scores)):
                if score > adjusted_threshold and not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]']:
                    rationales.append({
                        'token': token,
                        'score': float(score),
                        'position': i
                    })

    # If no rationales found using attention, try using rationale_scores if available
    if not rationales and 'rationale_scores' in outputs:
        rationale_scores = outputs['rationale_scores'][0].detach().cpu().numpy()

        # Get token IDs and attention mask
        token_ids = inputs['input_ids'][0].detach().cpu().numpy()
        attention_mask = inputs['attention_mask'][0].detach().cpu().numpy()

        # Get tokens from IDs
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Find rationales (spans with high scores)
        mask = attention_mask.astype(bool)
        scores = rationale_scores[mask]
        valid_tokens = [t for t, m in zip(tokens, mask) if m]

        # Normalize scores
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            # Extract rationales above threshold
            for i, (token, score) in enumerate(zip(valid_tokens, scores)):
                if score > rationale_threshold and not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]']:
                    rationales.append({
                        'token': token,
                        'score': float(score),
                        'position': i
                    })

    # If still no rationales, use a gradient-based approach as a last resort
    if not rationales:
        # Try to compute gradients with respect to embeddings
        model.zero_grad()

        # Get embeddings from the model's base model
        try:
            # Create a copy of inputs with requires_grad=True for embeddings
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Get the embedding layer
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'embeddings'):
                    # For BERT-like models
                    embedding_layer = model.base_model.embeddings
                elif hasattr(model.base_model, 'transformer') and hasattr(model.base_model.transformer, 'embeddings'):
                    # For DistilBERT
                    embedding_layer = model.base_model.transformer.embeddings
                else:
                    embedding_layer = None
            else:
                embedding_layer = None

            if embedding_layer is not None:
                # Get embeddings with gradient tracking
                with torch.set_grad_enabled(True):
                    # Get word embeddings
                    embeddings = embedding_layer.word_embeddings(input_ids)
                    embeddings.retain_grad()

                    # Forward pass through the model
                    if hasattr(model, 'forward_with_embeddings'):
                        outputs = model.forward_with_embeddings(embeddings, attention_mask)
                    else:
                        # Try to manually recreate the forward pass
                        if hasattr(embedding_layer, 'forward'):
                            # Get full embeddings (including position, etc.)
                            full_embeddings = embedding_layer(input_ids)
                        else:
                            # If we can't get full embeddings, use word embeddings
                            full_embeddings = embeddings

                        # Forward pass through the model
                        outputs = model(inputs_embeds=full_embeddings, attention_mask=attention_mask)

                    # Get prediction
                    logits = outputs['logits']
                    pred_class = torch.argmax(logits, dim=1).item()

                    # Compute gradient of the predicted class with respect to embeddings
                    logits[0, pred_class].backward()

                    # Get gradients
                    if embeddings.grad is not None:
                        # Compute importance scores (L2 norm of gradients)
                        grad_norm = torch.norm(embeddings.grad, dim=2)[0].detach().cpu().numpy()

                        # Get token IDs and attention mask
                        token_ids = input_ids[0].detach().cpu().numpy()
                        mask = attention_mask[0].detach().cpu().numpy().astype(bool)

                        # Get tokens from IDs
                        tokens = tokenizer.convert_ids_to_tokens(token_ids)

                        # Get scores for valid tokens
                        scores = grad_norm[mask]
                        valid_tokens = [t for t, m in zip(tokens, mask) if m]

                        # Normalize scores
                        if len(scores) > 0:
                            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

                            # Extract rationales above threshold
                            for i, (token, score) in enumerate(zip(valid_tokens, scores)):
                                if score > rationale_threshold and not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]']:
                                    rationales.append({
                                        'token': token,
                                        'score': float(score),
                                        'position': i
                                    })
        except Exception as e:
            # If gradient-based approach fails, we'll use a simple heuristic
            logger.warning(f"Gradient-based rationale extraction failed: {e}")

            # Use a simple heuristic: extract important words based on common sentiment words
            # This is a fallback when all other methods fail
            sentiment_words = {
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect', 'delicious', 'favorite'],
                'negative': ['bad', 'terrible', 'awful', 'worst', 'poor', 'horrible', 'disappointing', 'mediocre', 'cold', 'rude']
            }

            # Get prediction
            pred_class = torch.argmax(outputs['logits'], dim=1).item()
            sentiment = 'positive' if pred_class == 1 else 'negative'

            # Tokenize text
            words = text.lower().split()

            # Find sentiment words in the text
            for i, word in enumerate(words):
                # Remove punctuation
                clean_word = ''.join(c for c in word if c.isalnum())

                # Check if it's a sentiment word matching the prediction
                if clean_word in sentiment_words[sentiment]:
                    rationales.append({
                        'token': clean_word,
                        'score': 0.8,  # Arbitrary high score
                        'position': i
                    })

    # Extract concepts
    concept_scores = outputs.get('concept_scores', None)
    concepts = []

    if concept_scores is not None:
        # Convert to numpy for easier processing
        concept_scores = concept_scores[0].detach().cpu().numpy()

        # Get top k concepts
        top_indices = np.argsort(concept_scores)[-top_k_concepts:][::-1]

        for idx in top_indices:
            concepts.append({
                'concept_id': int(idx),
                'score': float(concept_scores[idx])
            })

    # Create highlighted text with rationales
    highlighted_text = highlight_rationales(text, tokenizer, rationales)

    return {
        'text': text,
        'prediction': pred_class,
        'prediction_label': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': confidence,
        'rationales': rationales,
        'concepts': concepts,
        'highlighted_text': highlighted_text
    }

def highlight_rationales(text, tokenizer, rationales, highlight_char='**'):
    """
    Highlight rationales in the original text

    Args:
        text: Original text
        tokenizer: Tokenizer used
        rationales: List of rationale dictionaries
        highlight_char: Character(s) to use for highlighting

    Returns:
        Text with rationales highlighted
    """
    if not rationales:
        return text

    # Sort rationales by score
    sorted_rationales = sorted(rationales, key=lambda x: x['score'], reverse=True)

    # Get the top rationales (to avoid too much highlighting)
    top_rationales = sorted_rationales[:min(15, len(sorted_rationales))]

    # Tokenize the text to get token spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.offset_mapping
    tokens = [tokenizer.convert_ids_to_tokens(id) for id in encoding.input_ids]

    # Create a list of characters to highlight
    highlight_mask = [False] * len(text)

    # Mark characters to highlight based on token positions
    for rationale in top_rationales:
        token = rationale['token']

        # Find all occurrences of this token in the tokens list
        for i, t in enumerate(tokens):
            # Check if token matches or is part of a word (for subword tokens)
            if t == token or (token.startswith('##') and t.endswith(token[2:])):
                start, end = offset_mapping[i]
                for j in range(start, end):
                    if j < len(highlight_mask):
                        highlight_mask[j] = True

    # Expand highlights to include adjacent highlighted characters
    # This helps with subword tokenization issues
    expanded_mask = highlight_mask.copy()
    for i in range(1, len(highlight_mask) - 1):
        if highlight_mask[i-1] and highlight_mask[i+1]:
            expanded_mask[i] = True

    # Build the highlighted text
    highlighted = []
    in_highlight = False

    for i, char in enumerate(text):
        if expanded_mask[i] and not in_highlight:
            highlighted.append(highlight_char)
            in_highlight = True
        elif not expanded_mask[i] and in_highlight:
            highlighted.append(highlight_char)
            in_highlight = False

        highlighted.append(char)

    # Close any open highlight
    if in_highlight:
        highlighted.append(highlight_char)

    # If no highlights were added, try a more aggressive approach
    if '**' not in ''.join(highlighted) and rationales:
        # Just highlight the top 3 tokens regardless of threshold
        top_tokens = [r['token'] for r in sorted_rationales[:3]]
        words_to_highlight = []

        # Find these words in the original text
        for word in text.split():
            lower_word = word.lower()
            if any(token.lower().replace('##', '') in lower_word for token in top_tokens):
                words_to_highlight.append(word)

        # Highlight these words
        for word in words_to_highlight:
            text = text.replace(word, f"{highlight_char}{word}{highlight_char}")

        return text

    return ''.join(highlighted)

def print_explanation(explanation, show_rationales=True, show_concepts=True):
    """
    Print the explanation in a readable format

    Args:
        explanation: Explanation dictionary
        show_rationales: Whether to show rationales
        show_concepts: Whether to show concepts
    """
    print("\n" + "="*80)
    print(f"TEXT: \"{explanation['text']}\"")
    print("-"*80)
    print(f"PREDICTION: {explanation['prediction']} ({explanation['prediction_label']}) (Confidence: {explanation['confidence']:.4f})")

    if show_rationales:
        print("-"*80)
        print("RATIONALES:")
        if explanation['rationales']:
            print(explanation['highlighted_text'])
        else:
            print("No significant rationales found.")

    if show_concepts:
        print("-"*80)
        print("TOP CONCEPTS:")
        if explanation['concepts']:
            for i, concept in enumerate(explanation['concepts']):
                print(f"  Concept {concept['concept_id']}: {concept['score']:.4f}")
        else:
            print("No significant concepts found.")

    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Extract rationales and concepts from trained model")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")

    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to analyze (if not provided, will use example texts)")
    parser.add_argument("--top_k_concepts", type=int, default=5,
                        help="Number of top concepts to show")
    parser.add_argument("--rationale_threshold", type=float, default=0.1,
                        help="Threshold for rationale extraction")
    parser.add_argument("--output_file", type=str,
                        help="Path to save explanations as JSON")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )

    # Use provided text or examples
    texts = []
    if args.text:
        texts.append(args.text)
    else:
        # Example texts
        texts = [
            "The food was delicious and the service was excellent. I would definitely come back!",
            "Terrible experience. The food was cold and the waiter was rude.",
            "The restaurant had great food but poor service.",
            "Overbooked and did not honor reservation time, put on wait list with walk-ins.",
            "Beautiful ambiance but the noise level was too high to have a conversation."
        ]

    # Generate explanations for each text
    explanations = []
    for text in texts:
        explanation = extract_rationales_and_concepts(
            model, tokenizer, text,
            top_k_concepts=args.top_k_concepts,
            rationale_threshold=args.rationale_threshold
        )
        explanations.append(explanation)
        print_explanation(explanation)

    # Save explanations if output file is provided
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(explanations, f, indent=2)
        logger.info(f"Explanations saved to {args.output_file}")

if __name__ == "__main__":
    main()
