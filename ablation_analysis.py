"""
Ablation analysis for the rationale-concept bottleneck model.

This script performs various ablation studies to understand the impact of
different components and hyperparameters on model performance.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import json
from transformers import set_seed, AutoTokenizer
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
    load_and_process_dataset,
    MetricsTracker,
    evaluate_model,
    device
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ablation analysis for rationale-concept model")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="yelp_polarity",
                       choices=["yelp_polarity", "dbpedia", "sst2", "agnews"],
                       help="Dataset to use for the experiment")

    # Model checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to saved model configuration")

    # Ablation options
    parser.add_argument("--ablation_type", type=str, default="all",
                       choices=["rationale_threshold", "concept_count", "interactions", "skip_connection", "all"],
                       help="Type of ablation study to perform")

    # Rationale threshold ablation
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5",
                       help="Comma-separated list of rationale thresholds to test")

    # Concept count ablation
    parser.add_argument("--concept_counts", type=str, default="10,20,30,40,50",
                       help="Comma-separated list of concept counts to test")

    # Test set size
    parser.add_argument("--max_test_samples", type=int, default=500,
                       help="Maximum number of test samples to use")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                       help="Directory to save ablation results")

    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

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

def get_dataset_info(dataset_name):
    """Get dataset-specific information"""
    if dataset_name == "yelp_polarity":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        num_labels = 2
    elif dataset_name == "dbpedia":
        class_names = {
            0: "Company", 1: "Educational Institution", 2: "Artist",
            3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
            6: "Building", 7: "Natural Place", 8: "Village",
            9: "Animal", 10: "Plant", 11: "Album",
            12: "Film", 13: "Written Work"
        }
        num_labels = 14
    elif dataset_name == "sst2":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        num_labels = 2
    elif dataset_name == "agnews":
        class_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        num_labels = 4
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return class_names, num_labels

def run_rationale_threshold_ablation(model, tokenizer, test_dataset, thresholds, output_dir):
    """Test the impact of different rationale thresholds"""
    print(f"\nRunning rationale threshold ablation with thresholds: {thresholds}")

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    # Store results
    results = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'avg_rationale_pct': []
    }

    # Original model target rationale percentage
    original_target_pct = model.config.target_rationale_percentage

    # Test each threshold
    for threshold in thresholds:
        print(f"\nTesting rationale threshold: {threshold}")

        # Modify model's target rationale percentage
        model.config.target_rationale_percentage = threshold

        # Evaluate model
        metrics = evaluate_model(model, test_loader)

        # Calculate average rationale percentage
        rationale_pcts = []
        for batch in tqdm(test_loader, desc="Calculating rationale percentages"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Get rationale masks
                rationale_masks = outputs['rationale_mask']
                attention_masks = batch['attention_mask']

                # Calculate rationale percentage for each example
                for i in range(len(rationale_masks)):
                    mask = rationale_masks[i]
                    attn = attention_masks[i]
                    valid_tokens = attn.sum().item()
                    rationale_tokens = (mask * attn).sum().item()
                    rationale_pcts.append(rationale_tokens / valid_tokens if valid_tokens > 0 else 0)

        avg_rationale_pct = sum(rationale_pcts) / len(rationale_pcts) if rationale_pcts else 0

        # Store results
        results['threshold'].append(threshold)
        results['accuracy'].append(metrics['accuracy'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f1'].append(metrics['f1'])
        results['avg_rationale_pct'].append(avg_rationale_pct)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Average Rationale %: {avg_rationale_pct:.2%}")

    # Reset model to original target percentage
    model.config.target_rationale_percentage = original_target_pct

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'rationale_threshold_results.csv'), index=False)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot accuracy and F1
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(results['threshold'], results['accuracy'], 'b-o', label='Accuracy')
    ax1.plot(results['threshold'], results['f1'], 'r-o', label='F1 Score')
    ax1.set_xlabel('Rationale Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Impact of Rationale Threshold on Performance')
    ax1.legend()
    ax1.grid(True)

    # Plot rationale percentage
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(results['threshold'], results['avg_rationale_pct'], 'g-o', label='Actual Rationale %')
    ax2.plot(results['threshold'], results['threshold'], 'k--', label='Target Rationale %')
    ax2.set_xlabel('Target Rationale Threshold')
    ax2.set_ylabel('Actual Rationale %')
    ax2.set_title('Target vs. Actual Rationale Percentage')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rationale_threshold_plot.png'))
    plt.close()

    return results_df

def run_concept_interaction_ablation(model, tokenizer, test_dataset, output_dir):
    """Test the impact of concept interactions"""
    print("\nRunning concept interaction ablation")

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    # Store results
    results = {
        'interactions_enabled': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Original interaction setting
    original_interactions = model.config.enable_concept_interactions

    # Test with and without interactions
    for enable_interactions in [False, True]:
        print(f"\nTesting with concept interactions: {enable_interactions}")

        # Modify model's interaction setting
        model.config.enable_concept_interactions = enable_interactions
        model.concept_mapper.enable_interactions = enable_interactions

        # Evaluate model
        metrics = evaluate_model(model, test_loader)

        # Store results
        results['interactions_enabled'].append(enable_interactions)
        results['accuracy'].append(metrics['accuracy'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f1'].append(metrics['f1'])

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

    # Reset model to original setting
    model.config.enable_concept_interactions = original_interactions
    model.concept_mapper.enable_interactions = original_interactions

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'concept_interaction_results.csv'), index=False)

    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot metrics
    x = ['Disabled', 'Enabled']
    plt.bar(x, results['accuracy'], width=0.2, label='Accuracy', color='blue', alpha=0.7)
    plt.bar([p + 0.2 for p in range(len(x))], results['f1'], width=0.2, label='F1 Score', color='red', alpha=0.7)

    plt.xlabel('Concept Interactions')
    plt.ylabel('Score')
    plt.title('Impact of Concept Interactions on Performance')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_interaction_plot.png'))
    plt.close()

    return results_df

def run_skip_connection_ablation(model, tokenizer, test_dataset, output_dir):
    """Test the impact of skip connections"""
    print("\nRunning skip connection ablation")

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    # Store results
    results = {
        'skip_connection': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Original skip connection setting
    original_skip = model.use_skip_connection

    # We can't easily modify the model architecture after it's created
    # So we'll just report the performance with the current setting
    print(f"\nTesting with skip connection: {original_skip}")

    # Evaluate model
    metrics = evaluate_model(model, test_loader)

    # Store results
    results['skip_connection'].append(original_skip)
    results['accuracy'].append(metrics['accuracy'])
    results['precision'].append(metrics['precision'])
    results['recall'].append(metrics['recall'])
    results['f1'].append(metrics['f1'])

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'skip_connection_results.csv'), index=False)

    return results_df

def analyze_concept_activations(model, tokenizer, test_dataset, output_dir, num_samples=100):
    """Analyze concept activations across the test set"""
    print(f"\nAnalyzing concept activations across {num_samples} test samples")

    # Limit samples
    if num_samples < len(test_dataset):
        test_subset = test_dataset.select(range(num_samples))
    else:
        test_subset = test_dataset

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=1,
        shuffle=False
    )

    # Store concept activations and labels
    concept_activations = []
    labels = []

    # Collect activations
    for batch in tqdm(test_loader, desc="Collecting concept activations"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Get concept probabilities
            concept_probs = outputs['concept_probs'][0].cpu().numpy()
            concept_activations.append(concept_probs)
            labels.append(batch['labels'].item())

    # Convert to arrays
    concept_activations = np.array(concept_activations)
    labels = np.array(labels)

    # Calculate average activation per concept
    avg_activations = concept_activations.mean(axis=0)

    # Calculate average activation per concept per class
    unique_labels = np.unique(labels)
    class_activations = {}
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            class_activations[label] = concept_activations[mask].mean(axis=0)

    # Plot average concept activations
    plt.figure(figsize=(12, 6))

    # Plot overall average
    plt.bar(range(len(avg_activations)), avg_activations, alpha=0.7, label='Overall')

    plt.xlabel('Concept Index')
    plt.ylabel('Average Activation')
    plt.title('Average Concept Activations Across Test Set')
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

    return {
        'avg_activations': avg_activations,
        'class_activations': class_activations
    }

def generate_summary_visualization(results_dict, output_dir):
    """Generate a summary visualization of all ablation results"""
    print("\nGenerating summary visualization...")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # Define grid layout
    gs = plt.GridSpec(3, 2, figure=fig)

    # 1. Rationale threshold results
    if 'rationale_threshold' in results_dict:
        ax1 = fig.add_subplot(gs[0, 0])
        df = results_dict['rationale_threshold']

        # Plot accuracy and F1 vs threshold
        ax1.plot(df['threshold'], df['accuracy'], 'b-o', label='Accuracy')
        ax1.plot(df['threshold'], df['f1'], 'r-o', label='F1 Score')
        ax1.set_xlabel('Rationale Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Impact of Rationale Threshold on Performance')
        ax1.legend()
        ax1.grid(True)

        # Plot rationale percentage
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['threshold'], df['avg_rationale_pct'], 'g-o', label='Actual Rationale %')
        ax2.plot(df['threshold'], df['threshold'], 'k--', label='Target Rationale %')
        ax2.set_xlabel('Target Rationale Threshold')
        ax2.set_ylabel('Actual Rationale %')
        ax2.set_title('Target vs. Actual Rationale Percentage')
        ax2.legend()
        ax2.grid(True)

    # 2. Concept interaction results
    if 'concept_interaction' in results_dict:
        ax3 = fig.add_subplot(gs[1, 0])
        df = results_dict['concept_interaction']

        # Convert boolean to string for better display
        x_labels = ['Disabled', 'Enabled']

        # Plot metrics
        x = range(len(x_labels))
        width = 0.35
        ax3.bar([p - width/2 for p in x], df['accuracy'], width=width, label='Accuracy', color='blue', alpha=0.7)
        ax3.bar([p + width/2 for p in x], df['f1'], width=width, label='F1 Score', color='red', alpha=0.7)

        ax3.set_xticks(x)
        ax3.set_xticklabels(x_labels)
        ax3.set_xlabel('Concept Interactions')
        ax3.set_ylabel('Score')
        ax3.set_title('Impact of Concept Interactions on Performance')
        ax3.legend()
        ax3.grid(True, axis='y')

    # 3. Concept activations heatmap
    if 'concept_activations' in results_dict:
        ax4 = fig.add_subplot(gs[1, 1])
        heatmap_data = results_dict['concept_activations']['heatmap_data']
        class_labels = results_dict['concept_activations']['class_labels']

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            yticklabels=class_labels,
            xticklabels=[f'C{i}' for i in range(heatmap_data.shape[1])],
            cbar_kws={'label': 'Activation'},
            ax=ax4
        )

        ax4.set_title('Concept Activations by Class')
        ax4.set_xlabel('Concept')
        ax4.set_ylabel('Class')

    # 4. Example rationales visualization
    if 'example_rationales' in results_dict:
        ax5 = fig.add_subplot(gs[2, :])
        examples = results_dict['example_rationales']

        # Create a table with example texts and their rationales
        cell_text = []
        for ex in examples[:3]:  # Show top 3 examples
            cell_text.append([
                ex['text'][:50] + "...",
                ex['rationale'],
                f"{ex['rationale_pct']:.1%}",
                f"{ex['accuracy']:.2f}"
            ])

        ax5.axis('off')
        table = ax5.table(
            cellText=cell_text,
            colLabels=['Text', 'Extracted Rationale', 'Rationale %', 'Accuracy'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4, 0.1, 0.1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax5.set_title('Example Rationales', pad=20)

    # Adjust layout and save
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'ablation_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary visualization saved to: {summary_path}")
    return summary_path

def extract_example_rationales(model, tokenizer, test_dataset, num_examples=3):
    """Extract example rationales from the test set"""
    print(f"\nExtracting {num_examples} example rationales...")

    # Create a small dataloader for examples
    example_loader = torch.utils.data.DataLoader(
        test_dataset.select(range(min(num_examples * 5, len(test_dataset)))),
        batch_size=1,
        shuffle=True
    )

    examples = []

    # Process examples
    for batch in example_loader:
        if len(examples) >= num_examples:
            break

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Get input text
        input_ids = batch['input_ids'][0]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Generate explanation
        with torch.no_grad():
            explanation = model.explain_prediction(tokenizer, text)

            # Forward pass to get predictions
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Calculate accuracy
            logits = outputs['logits']
            pred = torch.argmax(logits, dim=1).item()
            true_label = batch['labels'].item()
            accuracy = 1.0 if pred == true_label else 0.0

        # Add to examples
        examples.append({
            'text': text,
            'rationale': explanation['rationale'],
            'rationale_pct': explanation['rationale_percentage'],
            'prediction': explanation['prediction'],
            'true_label': true_label,
            'accuracy': accuracy
        })

    return examples

def main():
    """Main function for ablation analysis"""
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset info
    class_names, num_labels = get_dataset_info(args.dataset)

    # Load model and config
    print(f"Loading model from {args.model_path}")
    model, config = load_model_and_config(args.model_path, args.config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Load dataset
    print(f"Loading {args.dataset} dataset")
    datasets = load_and_process_dataset(args.dataset, tokenizer, config)

    # Limit test set size if specified
    if args.max_test_samples is not None:
        print(f"Limiting test set to {args.max_test_samples} samples")
        datasets['test'] = datasets['test'].select(range(min(args.max_test_samples, len(datasets['test']))))

    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]

    # Parse concept counts
    concept_counts = [int(c) for c in args.concept_counts.split(',')]

    # Store all results for summary visualization
    all_results = {}

    # Run ablation studies
    if args.ablation_type == 'rationale_threshold' or args.ablation_type == 'all':
        rationale_results = run_rationale_threshold_ablation(
            model, tokenizer, datasets['test'], thresholds, output_dir
        )
        all_results['rationale_threshold'] = rationale_results

    if args.ablation_type == 'interactions' or args.ablation_type == 'all':
        interaction_results = run_concept_interaction_ablation(
            model, tokenizer, datasets['test'], output_dir
        )
        all_results['concept_interaction'] = interaction_results

    if args.ablation_type == 'skip_connection' or args.ablation_type == 'all':
        skip_results = run_skip_connection_ablation(
            model, tokenizer, datasets['test'], output_dir
        )
        all_results['skip_connection'] = skip_results

    # Always analyze concept activations
    activation_results = analyze_concept_activations(
        model, tokenizer, datasets['test'], output_dir
    )

    # Extract example rationales
    example_rationales = extract_example_rationales(
        model, tokenizer, datasets['test']
    )

    # Save example rationales
    with open(os.path.join(output_dir, 'example_rationales.json'), 'w') as f:
        json.dump(example_rationales, f, indent=2, cls=NumpyEncoder)

    # Prepare data for summary visualization
    if 'concept_activations' in all_results:
        all_results['concept_activations'] = {
            'heatmap_data': np.array([act for _, act in activation_results['class_activations'].items()]),
            'class_labels': [f'Class {label}' for label in activation_results['class_activations'].keys()]
        }

    all_results['example_rationales'] = example_rationales

    # Generate summary visualization
    summary_path = generate_summary_visualization(all_results, output_dir)

    print(f"\nAblation analysis complete. Results saved to {output_dir}")
    print(f"Summary visualization saved to: {summary_path}")

    # Open the summary visualization if on a system with a display
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(summary_path)}")
    except:
        pass

if __name__ == "__main__":
    main()
