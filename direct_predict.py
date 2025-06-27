#!/usr/bin/env python
"""
Direct Prediction from Checkpoint

This script directly uses the checkpoint to make predictions without
trying to extract rationales or concepts. It's a simplified approach
that focuses on getting accurate predictions.
"""

import os
import argparse
import json
import torch
import logging
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

class DirectClassifier(torch.nn.Module):
    """
    Direct classifier that uses the checkpoint
    
    This model directly loads the encoder and classifier from the checkpoint
    without trying to use the rationale extractor or concept mapper.
    """
    def __init__(self, model_name, dataset_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dataset_name = dataset_name
        
        # Create a classifier that matches the checkpoint format
        self.classifiers = torch.nn.ModuleDict()
        self.classifiers[dataset_name] = torch.nn.Sequential(
            torch.nn.Linear(768, 384),  # Simplified to use CLS token directly
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(384, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        # Encode input
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
        
        # Apply classifier
        logits = self.classifiers[self.dataset_name](pooled_output)
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

def load_checkpoint_selectively(model, checkpoint_path, dataset_name):
    """Load only the encoder and classifier weights from the checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a new state dict
    new_state_dict = {}
    
    # Copy encoder weights
    for key in checkpoint:
        if key.startswith('encoder.'):
            new_state_dict[key] = checkpoint[key]
    
    # Copy classifier weights with remapping
    for key in checkpoint:
        if key.startswith(f'classifiers.{dataset_name}.'):
            # Remove the dataset name from the key path
            new_key = f'classifiers.{dataset_name}.' + key.split('.')[-1]
            new_state_dict[new_key] = checkpoint[key]
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Direct prediction from checkpoint")
    
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
    if args.dataset:
        config['dataset_name'] = args.dataset
    logger.info(f"Loaded configuration: {config['base_model_name']}, {config['num_labels']} classes")
    
    # Initialize model
    model = DirectClassifier(
        config['base_model_name'],
        config['dataset_name'],
        config['num_labels']
    )
    
    # Load checkpoint selectively
    model = load_checkpoint_selectively(model, args.checkpoint_path, config['dataset_name'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
    
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
            if config['dataset_name'] in class_names:
                class_name = class_names[config['dataset_name']][prediction]
                prediction_str = f"{prediction} ({class_name})"
            else:
                prediction_str = str(prediction)
            
            # Print result
            print(f"\nText: \"{text}\"")
            print(f"Prediction: {prediction_str}")
            print(f"Confidence: {confidence:.4f}")
            
            # Print all class probabilities
            if config['dataset_name'] in class_names:
                print("Class probabilities:")
                for i, name in enumerate(class_names[config['dataset_name']]):
                    if i < probs.size(1):
                        print(f"  {name}: {probs[0, i].item():.4f}")

if __name__ == "__main__":
    main()
