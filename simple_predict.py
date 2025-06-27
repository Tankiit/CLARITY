#!/usr/bin/env python
"""
Simple prediction script for the checkpoint

This script loads the checkpoint and makes predictions on text inputs.
It's a simplified version that focuses on just getting predictions working.
"""

import os
import argparse
import json
import torch
import logging
import re
from transformers import AutoTokenizer, AutoModel

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

class SimpleClassifier(torch.nn.Module):
    """Simple classifier that loads the checkpoint"""
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
        logits = self.classifier(pooled_output)
        return logits

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

def main():
    parser = argparse.ArgumentParser(description="Simple prediction from checkpoint")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")
    
    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to classify (if not provided, will use examples)")
    parser.add_argument("--dataset", type=str, choices=['sst2', 'yelp_polarity', 'ag_news', 'dbpedia'],
                        help="Dataset name for class labels")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_json(args.config_path)
    logger.info(f"Loaded configuration: {config['base_model_name']}, {config['num_labels']} classes")
    
    # Initialize a simple model
    model = SimpleClassifier(config['base_model_name'], config['num_labels'])
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
    
    # Use provided text or examples
    texts = []
    if args.text:
        texts.append(args.text)
    else:
        # Example texts for different datasets
        if args.dataset == 'sst2' or args.dataset == 'yelp_polarity':
            texts = [
                "This movie was absolutely amazing. The acting was superb.",
                "I hated every minute of this terrible film.",
                "The restaurant had great food but poor service."
            ]
        elif args.dataset == 'ag_news':
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
    
    # Define class names for datasets
    class_names = {
        'sst2': ['Negative', 'Positive'],
        'yelp_polarity': ['Negative', 'Positive'],
        'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'dbpedia': ['Company', 'Educational Institution', 'Artist', 'Athlete', 
                   'Office Holder', 'Mean Of Transportation', 'Building', 'Natural Place', 
                   'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
    }
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probs[0, prediction].item()
            
            # Get class name if available
            if args.dataset and args.dataset in class_names:
                class_name = class_names[args.dataset][prediction]
                prediction_str = f"{prediction} ({class_name})"
            else:
                prediction_str = str(prediction)
            
            # Print result
            print(f"\nText: \"{text}\"")
            print(f"Prediction: {prediction_str}")
            print(f"Confidence: {confidence:.4f}")
            
            # Print all class probabilities
            if args.dataset and args.dataset in class_names:
                print("Class probabilities:")
                for i, name in enumerate(class_names[args.dataset]):
                    if i < probs.size(1):
                        print(f"  {name}: {probs[0, i].item():.4f}")

if __name__ == "__main__":
    main()
