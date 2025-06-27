#!/usr/bin/env python
"""
Simple Explanations for Text Classification

This script generates simple explanations for text classification predictions
without relying on the complex rationale-concept model architecture.
"""

import os
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import logging

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

def load_config_from_json(config_path):
    """Load configuration from a JSON file"""
    # Load the JSON file
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Extract model name
    base_model_name = config_data['model']['model_name']
    if base_model_name.startswith('models/'):
        base_model_name = base_model_name[7:]  # Remove 'models/' prefix
    
    # Extract dataset name
    dataset_name = config_data.get('dataset_name', 'sst2')
    
    # Extract number of classes for the dataset
    num_classes_str = config_data['model']['num_classes']
    # Convert string representation to dict if needed
    if isinstance(num_classes_str, str) and '{' in num_classes_str:
        import ast
        num_classes_dict = ast.literal_eval(num_classes_str)
        num_labels = num_classes_dict.get(dataset_name, 2)
    else:
        num_labels = int(num_classes_str)
    
    return {
        'base_model_name': base_model_name,
        'dataset_name': dataset_name,
        'num_labels': num_labels
    }

def generate_attention_based_explanation(text, model_name, dataset_name):
    """
    Generate a simple explanation based on attention weights
    
    This function uses the base model's attention weights to identify
    important words in the text.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs with attention weights
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention weights from the last layer
    # Shape: [batch_size, num_heads, seq_length, seq_length]
    attention = outputs.attentions[-1]
    
    # Average attention across heads
    # Shape: [batch_size, seq_length, seq_length]
    attention_avg = attention.mean(dim=1)
    
    # Get attention from CLS token to other tokens
    # Shape: [batch_size, seq_length]
    cls_attention = attention_avg[:, 0, :]
    
    # Convert to numpy for easier processing
    cls_attention = cls_attention[0].cpu().numpy()
    
    # Get token IDs and attention mask
    token_ids = inputs["input_ids"][0].cpu().numpy()
    attention_mask = inputs["attention_mask"][0].cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Create a list of (token, attention) pairs for tokens with attention mask
    token_attention = []
    for i, (token, attn, mask) in enumerate(zip(tokens, cls_attention, attention_mask)):
        if mask > 0 and token not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            token_attention.append((token, attn, i))
    
    # Sort by attention score
    token_attention.sort(key=lambda x: x[1], reverse=True)
    
    # Get top tokens (up to 10)
    top_tokens = token_attention[:10]
    
    # Create a simple explanation
    explanation = {
        "important_tokens": [(token, float(score)) for token, score, _ in top_tokens],
        "token_indices": [idx for _, _, idx in top_tokens]
    }
    
    return explanation

def classify_text(text, model_name, dataset_name, num_labels):
    """
    Classify text using a pre-trained model
    
    This function uses a simple text classification pipeline to classify the text.
    """
    # Define class names for datasets
    class_names = {
        'sst2': ['Negative', 'Positive'],
        'yelp_polarity': ['Negative', 'Positive'],
        'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'dbpedia': ['Company', 'Educational Institution', 'Artist', 'Athlete', 
                   'Office Holder', 'Mean Of Transportation', 'Building', 'Natural Place', 
                   'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
    }
    
    # Create a text classification pipeline
    classifier = pipeline(
        "text-classification", 
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Classify the text
    result = classifier(text)
    
    # Extract prediction and score
    label = result[0]['label']
    score = result[0]['score']
    
    # Map label to class index and name
    if 'LABEL_' in label:
        class_idx = int(label.split('_')[1])
    else:
        class_idx = 1 if label.lower() == 'positive' else 0  # Default mapping for sentiment
    
    # Get class name if available
    if dataset_name in class_names and class_idx < len(class_names[dataset_name]):
        class_name = class_names[dataset_name][class_idx]
    else:
        class_name = label
    
    return {
        "prediction": class_idx,
        "class_name": class_name,
        "confidence": score
    }

def main():
    parser = argparse.ArgumentParser(description="Generate simple explanations for text classification")
    
    # Required arguments
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to explain (if not provided, will use examples)")
    parser.add_argument("--dataset", type=str, choices=['sst2', 'yelp_polarity', 'ag_news', 'dbpedia'],
                        help="Dataset name for class labels")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_json(args.config_path)
    if args.dataset:
        config['dataset_name'] = args.dataset
    logger.info(f"Loaded configuration: {config['base_model_name']}, {config['num_labels']} classes")
    
    # Use provided text or examples
    texts = []
    if args.text:
        texts.append(args.text)
    else:
        # Example texts for different datasets
        if config['dataset_name'] == 'sst2' or config['dataset_name'] == 'yelp_polarity':
            texts = [
                "This movie was absolutely amazing. The acting was superb.",
                "I hated every minute of this terrible film.",
                "The restaurant had great food but poor service."
            ]
        elif config['dataset_name'] == 'ag_news':
            texts = [
                "The company announced a new product that will revolutionize the market.",
                "The team won the championship after an incredible comeback.",
                "Scientists discovered a new species in the Amazon rainforest."
            ]
        else:
            texts = [
                "The company announced a new product that will revolutionize the market.",
                "The team won the championship after an incredible comeback.",
                "This restaurant has amazing food and excellent service.",
                "The movie was boring and predictable. I wouldn't recommend it."
            ]
    
    # Process each text
    for text in texts:
        print(f"\nText: \"{text}\"")
        
        # Classify the text
        classification = classify_text(
            text, 
            config['base_model_name'], 
            config['dataset_name'],
            config['num_labels']
        )
        
        # Generate explanation
        explanation = generate_attention_based_explanation(
            text,
            config['base_model_name'],
            config['dataset_name']
        )
        
        # Print results
        print(f"Prediction: {classification['prediction']} ({classification['class_name']})")
        print(f"Confidence: {classification['confidence']:.4f}")
        print("\nImportant words (based on attention):")
        for token, score in explanation['important_tokens']:
            print(f"  {token}: {score:.4f}")
        
        # Highlight important words in the text
        tokens = []
        for i, token in enumerate(text.split()):
            if any(important_token in token.lower() for important_token, _ in explanation['important_tokens'][:5]):
                tokens.append(f"**{token}**")
            else:
                tokens.append(token)
        
        highlighted_text = " ".join(tokens)
        print(f"\nHighlighted text:\n{highlighted_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()
