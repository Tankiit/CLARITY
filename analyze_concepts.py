#!/usr/bin/env python
"""
Analyze Concepts from Trained Model

This script analyzes the concepts learned by a trained RationaleConceptBottleneckModel
to understand what each concept represents.
"""

import os
import argparse
import torch
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig
from torch.utils.data import DataLoader, Dataset

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

class SimpleDataset(Dataset):
    """Simple dataset for text examples"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add text for reference
        encoding['text'] = text

        return encoding

def get_concept_activations(model, dataloader, num_concepts):
    """
    Get concept activations for a set of examples

    Args:
        model: Trained model
        dataloader: DataLoader with examples
        num_concepts: Number of concepts in the model

    Returns:
        Dictionary mapping concept IDs to lists of (text, activation) pairs
    """
    concept_activations = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing examples"):
            # Get texts
            texts = batch.pop('text')

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Get model outputs
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Get concept scores
            concept_scores = outputs.get('concept_scores', None)

            if concept_scores is not None:
                # For each example in the batch
                for i, text in enumerate(texts):
                    # For each concept
                    for concept_id in range(num_concepts):
                        score = concept_scores[i, concept_id].item()
                        concept_activations[concept_id].append((text, score))

    return concept_activations

def analyze_concepts(concept_activations, top_k=5):
    """
    Analyze concepts based on their activations

    Args:
        concept_activations: Dictionary mapping concept IDs to lists of (text, activation) pairs
        top_k: Number of top examples to consider for each concept

    Returns:
        Dictionary with concept analysis
    """
    concept_analysis = {}

    for concept_id, activations in concept_activations.items():
        # Sort activations by score (descending)
        sorted_activations = sorted(activations, key=lambda x: x[1], reverse=True)

        # Get top and bottom examples
        top_examples = sorted_activations[:top_k]
        bottom_examples = sorted_activations[-top_k:]

        # Calculate statistics
        all_scores = [score for _, score in activations]
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        concept_analysis[concept_id] = {
            'top_examples': top_examples,
            'bottom_examples': bottom_examples,
            'mean_score': mean_score,
            'std_score': std_score
        }

    return concept_analysis

def visualize_concept(concept_id, concept_info, output_dir):
    """
    Visualize a concept with its top and bottom examples

    Args:
        concept_id: ID of the concept
        concept_info: Information about the concept
        output_dir: Directory to save visualization
    """
    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot top examples
    plt.subplot(2, 1, 1)
    texts = [text[:50] + "..." if len(text) > 50 else text for text, _ in concept_info['top_examples']]
    scores = [score for _, score in concept_info['top_examples']]

    plt.barh(range(len(texts)), scores, color='green')
    plt.yticks(range(len(texts)), texts)
    plt.title(f"Top Examples for Concept {concept_id}")
    plt.xlabel("Activation")
    plt.tight_layout()

    # Plot bottom examples
    plt.subplot(2, 1, 2)
    texts = [text[:50] + "..." if len(text) > 50 else text for text, _ in concept_info['bottom_examples']]
    scores = [score for _, score in concept_info['bottom_examples']]

    plt.barh(range(len(texts)), scores, color='red')
    plt.yticks(range(len(texts)), texts)
    plt.title(f"Bottom Examples for Concept {concept_id}")
    plt.xlabel("Activation")
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"concept_{concept_id}.png"))
    plt.close()

def generate_concept_report(concept_analysis, output_dir):
    """
    Generate a report of concept analysis

    Args:
        concept_analysis: Dictionary with concept analysis
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "concept_report.md")

    with open(report_path, 'w') as f:
        f.write("# Concept Analysis Report\n\n")

        for concept_id, concept_info in sorted(concept_analysis.items()):
            f.write(f"## Concept {concept_id}\n\n")
            f.write(f"Mean activation: {concept_info['mean_score']:.4f}\n\n")
            f.write(f"Standard deviation: {concept_info['std_score']:.4f}\n\n")

            f.write("### Top Examples\n\n")
            for i, (text, score) in enumerate(concept_info['top_examples']):
                f.write(f"{i+1}. \"{text}\" (Score: {score:.4f})\n\n")

            f.write("### Bottom Examples\n\n")
            for i, (text, score) in enumerate(concept_info['bottom_examples']):
                f.write(f"{i+1}. \"{text}\" (Score: {score:.4f})\n\n")

            f.write("\n---\n\n")

    logger.info(f"Concept report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Analyze concepts from trained model")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")

    # Optional arguments
    parser.add_argument("--examples_file", type=str,
                        help="Path to file with example texts (one per line)")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to generate if no examples file is provided")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top examples to show for each concept")
    parser.add_argument("--output_dir", type=str, default="concept_analysis",
                        help="Directory to save analysis results")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_path, args.config_path
    )

    # Get example texts
    if args.examples_file and os.path.exists(args.examples_file):
        with open(args.examples_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} examples from {args.examples_file}")
    else:
        # Generate example texts
        from datasets import load_dataset

        try:
            # Try to load CEBAB dataset
            dataset = load_dataset("CEBaB/CEBaB")
            examples = []

            # Get examples from different splits
            for split in ['train_inclusive', 'validation', 'test']:
                if split in dataset:
                    split_examples = dataset[split]['description'][:args.num_examples // 3]
                    examples.extend(split_examples)

            texts = examples[:args.num_examples]
            logger.info(f"Generated {len(texts)} examples from CEBAB dataset")
        except:
            # Fallback to example texts
            texts = [
                "The food was delicious and the service was excellent.",
                "Terrible experience. The food was cold and the waiter was rude.",
                "The restaurant had great food but poor service.",
                "Overbooked and did not honor reservation time.",
                "Beautiful ambiance but the noise level was too high."
            ] * 20  # Repeat to get more examples
            texts = texts[:args.num_examples]
            logger.info(f"Using {len(texts)} example texts")

    # Create dataset and dataloader
    dataset = SimpleDataset(texts, tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Get concept activations
    concept_activations = get_concept_activations(model, dataloader, config.num_concepts)

    # Analyze concepts
    concept_analysis = analyze_concepts(concept_activations, top_k=args.top_k)

    # Visualize concepts
    for concept_id, concept_info in concept_analysis.items():
        visualize_concept(concept_id, concept_info, args.output_dir)

    # Generate report
    report_path = generate_concept_report(concept_analysis, args.output_dir)

    logger.info(f"Concept analysis complete. Results saved to {args.output_dir}")
    logger.info(f"To view the report, open {report_path}")

if __name__ == "__main__":
    main()
