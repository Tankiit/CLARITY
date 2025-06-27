"""
Visualization script for trained rationale-concept bottleneck models.

This script loads a trained model and generates visualizations of explanations
without having to retrain the model.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import time
from transformers import AutoTokenizer
from tqdm import tqdm

# Custom JSON encoder for NumPy and PyTorch types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

# Import the implementation
from optimized_rationale_concept_model import (
    ModelConfig,
    RationaleConceptBottleneckModel,
    device
)

# Import visualization functions from the experiment script
from yelp_dbpedia_experiment import (
    visualize_explanation,
    create_explanations_summary,
    get_dataset_examples
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize explanations from a trained model")

    # Model checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to saved model configuration")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="yelp_polarity",
                       choices=["yelp_polarity", "dbpedia", "sst2", "agnews"],
                       help="Dataset to use for examples")

    # Custom examples
    parser.add_argument("--custom_examples", type=str, default=None,
                       help="Path to JSON file with custom examples to visualize")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Directory to save visualizations")

    # Rationale threshold
    parser.add_argument("--rationale_threshold", type=float, default=None,
                       help="Override the model's rationale threshold")

    # Concept threshold
    parser.add_argument("--concept_threshold", type=float, default=None,
                       help="Threshold for showing concepts (default: 0.1)")

    # Number of concepts to show
    parser.add_argument("--num_concepts", type=int, default=5,
                       help="Number of top concepts to show in visualizations")

    return parser.parse_args()

def load_model_and_config(model_path, config_path):
    """Load model and configuration from saved files"""
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Check if this is the new format or the old format
    if "model" in config_dict and "training" in config_dict:
        # This is the old format used in the existing models
        print("Detected old config format, converting to new format...")

        # Extract dataset name
        dataset_name = config_dict.get("dataset_name", "sst2")

        # Extract model parameters
        model_params = config_dict["model"]
        training_params = config_dict["training"]

        # Parse num_classes
        num_classes_str = model_params.get("num_classes", "{'sst2': 2}")
        num_classes_dict = eval(num_classes_str)  # Convert string to dict
        num_labels = num_classes_dict.get(dataset_name, 2)

        # Parse concept counts
        concept_config_str = model_params.get("concept", "")
        concept_counts = 100  # Default
        if "concept_counts" in concept_config_str:
            try:
                # Extract concept_counts from the string
                concept_counts_str = concept_config_str.split("concept_counts=")[1].split(",")[0]
                concept_counts_dict = eval(concept_counts_str)
                concept_counts = concept_counts_dict.get(dataset_name, 100)
            except:
                print("Could not parse concept_counts, using default value of 100")

        # Create a new config dictionary
        new_config = {
            "base_model_name": model_params.get("model_name", "distilbert-base-uncased").replace("models/", ""),
            "num_labels": num_labels,
            "num_concepts": concept_counts,
            "batch_size": int(training_params.get("batch_size", 16)),
            "max_seq_length": 128,  # Default
            "learning_rate": float(training_params.get("learning_rate", 2e-5)),
            "base_model_lr": float(training_params.get("base_model_lr", 5e-6)),
            "num_epochs": int(training_params.get("epochs", 100)),
            "seed": int(training_params.get("seed", 42)),
            "output_dir": training_params.get("save_dir", "models"),
            "warmup_ratio": float(training_params.get("warmup_ratio", 0.1)),
            "weight_decay": float(training_params.get("weight_decay", 0.01)),
            "max_grad_norm": float(training_params.get("max_grad_norm", 1.0)),
            "classification_weight": float(training_params.get("classification_weight", 1.0)),
            "rationale_sparsity_weight": float(training_params.get("rationale_sparsity_weight", 0.05)),
            "rationale_continuity_weight": float(training_params.get("rationale_continuity_weight", 0.2)),
            "concept_diversity_weight": float(training_params.get("concept_diversity_weight", 0.05)),
            "use_skip_connection": model_params.get("use_skip_connection", "True").lower() == "true",
            "enable_concept_interactions": "enable_concept_interactions=True" in model_params.get("concept", ""),
            "target_rationale_percentage": 0.2  # Default
        }

        config_dict = new_config

    # Create config object
    config = ModelConfig(**config_dict)

    # Create model
    model = RationaleConceptBottleneckModel(config)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config

def load_custom_examples(file_path):
    """Load custom examples from a JSON file"""
    with open(file_path, 'r') as f:
        examples = json.load(f)

    # Check format
    if isinstance(examples, list):
        # If it's a list of strings, use as is
        if all(isinstance(ex, str) for ex in examples):
            return examples
        # If it's a list of objects, extract the 'text' field
        elif all(isinstance(ex, dict) and 'text' in ex for ex in examples):
            return [ex['text'] for ex in examples]

    raise ValueError("Custom examples file must contain a list of strings or objects with 'text' field")

def generate_explanations(model, tokenizer, examples, class_names, rationale_threshold=None):
    """Generate explanations for a list of examples"""
    print(f"\nGenerating explanations for {len(examples)} examples...")

    # Set rationale threshold if provided
    original_threshold = model.config.target_rationale_percentage
    if rationale_threshold is not None:
        print(f"Setting rationale threshold to {rationale_threshold} (was {original_threshold})")
        model.config.target_rationale_percentage = rationale_threshold

    explanations = []

    for i, text in enumerate(tqdm(examples, desc="Generating explanations")):
        # Generate explanation
        explanation = model.explain_prediction(tokenizer, text)
        explanations.append({
            'text': text,
            'explanation': explanation,
            'class_name': class_names.get(explanation['prediction'], f'Class {explanation["prediction"]}')
        })

    # Reset threshold if changed
    if rationale_threshold is not None:
        model.config.target_rationale_percentage = original_threshold

    return explanations

def visualize_explanations(explanations, output_dir, class_names):
    """Create visualizations for explanations"""
    print(f"\nCreating visualizations in {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual visualizations
    for i, ex in enumerate(explanations):
        text = ex['text']
        explanation = ex['explanation']

        # Visualize explanation
        save_path = os.path.join(output_dir, f"example_{i+1}_explanation.png")
        visualize_explanation(text, explanation, class_names, save_path)
        predicted_class = explanation['prediction']
        print(f"  Explanation {i+1}: {class_names.get(predicted_class, f'Class {predicted_class}')} (saved to {save_path})")

    # Save explanations to JSON file
    explanations_file = os.path.join(output_dir, "explanations.json")
    with open(explanations_file, 'w') as f:
        json.dump(explanations, f, indent=2, cls=NumpyEncoder)
    print(f"  Explanations saved to: {explanations_file}")

    # Create a summary visualization
    summary_path = os.path.join(output_dir, "explanations_summary.png")
    create_explanations_summary(explanations, summary_path)
    print(f"  Summary visualization saved to: {summary_path}")

    return summary_path

def analyze_concept_activations(model, tokenizer, examples, output_dir):
    """Analyze concept activations across examples"""
    print("\nAnalyzing concept activations...")

    # Store concept activations
    concept_activations = []
    predictions = []

    # Process each example
    for text in tqdm(examples, desc="Processing examples"):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_seq_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get concept activations
        with torch.no_grad():
            outputs = model(**inputs)
            concept_probs = outputs['concept_probs'][0].cpu().numpy()
            concept_activations.append(concept_probs)
            predictions.append(outputs['logits'][0].argmax().item())

    # Convert to arrays
    concept_activations = np.array(concept_activations)
    predictions = np.array(predictions)

    # Calculate average activation per concept
    avg_activations = concept_activations.mean(axis=0)

    # Calculate average activation per concept per class
    unique_labels = np.unique(predictions)
    class_activations = {}
    for label in unique_labels:
        mask = predictions == label
        if np.any(mask):
            class_activations[label] = concept_activations[mask].mean(axis=0)

    # Plot average concept activations
    plt.figure(figsize=(12, 6))

    # Plot overall average
    plt.bar(range(len(avg_activations)), avg_activations, alpha=0.7, label='Overall')

    plt.xlabel('Concept Index')
    plt.ylabel('Average Activation')
    plt.title('Average Concept Activations Across Examples')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_activations_overall.png'))
    plt.close()

    # Plot class-specific activations
    plt.figure(figsize=(14, 8))

    # Plot for each class
    bar_width = 0.8 / len(class_activations)
    for i, (label, activations) in enumerate(class_activations.items()):
        plt.bar(
            [x + i * bar_width for x in range(len(activations))],
            activations,
            width=bar_width,
            alpha=0.7,
            label=f'Class {label}'
        )

    plt.xlabel('Concept Index')
    plt.ylabel('Average Activation')
    plt.title('Class-Specific Concept Activations')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_activations_by_class.png'))
    plt.close()

    # Create heatmap of concept activations by class
    plt.figure(figsize=(16, 10))

    # Prepare data for heatmap
    heatmap_data = np.zeros((len(unique_labels), len(avg_activations)))
    for i, label in enumerate(unique_labels):
        if label in class_activations:
            heatmap_data[i] = class_activations[label]

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        yticklabels=[f'Class {label}' for label in unique_labels],
        xticklabels=[f'C{i}' for i in range(len(avg_activations))],
        cbar_kws={'label': 'Activation'}
    )

    plt.title('Concept Activations by Class')
    plt.xlabel('Concept')
    plt.ylabel('Class')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_activations_heatmap.png'))
    plt.close()

    print(f"  Concept activation visualizations saved to {output_dir}")

    return {
        'avg_activations': avg_activations,
        'class_activations': class_activations
    }

def main():
    """Main function for visualization"""
    args = parse_arguments()

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and config
    print(f"Loading model from {args.model_path}")
    model, config = load_model_and_config(args.model_path, args.config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Get dataset info
    class_names, default_examples = get_dataset_examples(args.dataset)

    # Load examples
    if args.custom_examples:
        print(f"Loading custom examples from {args.custom_examples}")
        examples = load_custom_examples(args.custom_examples)
    else:
        print(f"Using default examples for {args.dataset}")
        examples = default_examples

    # Generate explanations
    explanations = generate_explanations(
        model, tokenizer, examples, class_names, args.rationale_threshold
    )

    # Visualize explanations
    summary_path = visualize_explanations(explanations, output_dir, class_names)

    # Analyze concept activations
    activation_results = analyze_concept_activations(model, tokenizer, examples, output_dir)

    print(f"\nVisualization complete. Results saved to {output_dir}")

    # Try to open the summary visualization
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(summary_path)}")
    except:
        pass

if __name__ == "__main__":
    main()
