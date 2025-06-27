"""
Example script for training and evaluating the optimized rationale-concept bottleneck model
on the Yelp Polarity and DBpedia datasets with visualization of results.
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
    train_model,
    evaluate_model,
    device
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train rationale-concept model on Yelp Polarity and DBpedia")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="yelp_polarity",
                       choices=["yelp_polarity", "dbpedia", "sst2", "agnews", "all"],
                       help="Dataset to use for the experiment")

    # Key parameters that can be adjusted
    parser.add_argument("--small", action="store_true",
                       help="Use smaller configuration for faster training")
    parser.add_argument("--fast", action="store_true",
                       help="Use LoRA and other optimizations for faster training")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations for sample predictions")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--inference_only", action="store_true",
                       help="Skip training and only run inference from a saved model")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model for inference only mode")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (when early stopping is not enabled)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping based on validation loss")
    parser.add_argument("--patience", type=int, default=2,
                       help="Number of epochs to wait for improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                       help="Minimum change in validation loss to qualify as improvement")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples to use (None for all)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                       help="Maximum number of validation samples to use (None for all)")

    return parser.parse_args()

def create_config(args, dataset_name):
    """Create model configuration based on arguments and dataset"""

    # Dataset-specific configurations
    dataset_configs = {
        'yelp_polarity': {
            'num_labels': 2,
            'output_dir': "yelp_models"
        },
        'dbpedia': {
            'num_labels': 14,
            'output_dir': "dbpedia_models"
        },
        'sst2': {
            'num_labels': 2,
            'output_dir': "sst2_models"
        },
        'agnews': {
            'num_labels': 4,
            'output_dir': "agnews_models"
        }
    }

    # Get dataset configuration
    ds_config = dataset_configs[dataset_name]

    # Determine number of epochs
    # Use more epochs when early stopping is enabled
    num_epochs = 10 if args.early_stopping else args.epochs

    # Base configuration
    config = ModelConfig(
        base_model_name="distilbert-base-uncased",
        num_labels=ds_config['num_labels'],
        num_concepts=50,
        batch_size=args.batch_size,
        max_seq_length=128,
        learning_rate=2e-5,
        base_model_lr=1e-5,
        num_epochs=num_epochs,
        seed=args.seed,
        output_dir=ds_config['output_dir']
    )

    # Small configuration for quick testing
    if args.small:
        config.num_concepts = 20
        config.batch_size = 16
        config.max_seq_length = 64
        config.num_epochs = 2 if not args.early_stopping else 5

    # Fast training configuration
    if args.fast:
        config.use_lora = True
        config.lora_r = 8
        config.enable_concept_interactions = False  # Simpler model

    return config

def visualize_explanation(text, explanation, class_names, save_path=None):
    """Visualize model explanation for a prediction"""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # 1. Visualize rationale
    rationale = explanation["rationale"]
    rationale_pct = explanation["rationale_percentage"]

    # Extract rationale tokens
    words = text.split()
    rationale_words = rationale.split()

    # Create a binary mask for highlighting
    mask = []
    for word in words:
        if any(r.lower().startswith(word.lower()) for r in rationale_words):
            mask.append(1)
        else:
            mask.append(0)

    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Word': words,
        'Is_Rationale': mask
    })

    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray

    # Plot text
    for i, row in df.iterrows():
        word = row['Word']
        color = highlight_color if row['Is_Rationale'] == 1 else normal_color
        ax1.text(i, 0, word,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=10)

    # Hide axes but keep frame
    ax1.set_xlim(-1, len(words))
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Extracted Rationale ({rationale_pct:.1%} of text)", fontsize=14)

    # 2. Visualize top concepts
    concepts = explanation["top_concepts"]
    if concepts:
        concept_names = [c[0] for c in concepts]
        concept_scores = [c[1] for c in concepts]

        # Sort by score
        sorted_idx = np.argsort(concept_scores)
        concept_names = [concept_names[i] for i in sorted_idx]
        concept_scores = [concept_scores[i] for i in sorted_idx]

        # Create horizontal bar chart
        ax2.barh(concept_names, concept_scores, color='skyblue')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Concept Score')
        ax2.set_title(f"Top Concepts for Prediction: {class_names.get(explanation['prediction'], f'Class {explanation['prediction']}')}", fontsize=14)

    else:
        ax2.text(0.5, 0.5, "No active concepts found",
                 ha='center', va='center', fontsize=12)

    # Add prediction information
    fig.text(0.5, 0.01,
             f"Prediction: {class_names.get(explanation['prediction'], f'Class {explanation['prediction']}')} (Confidence: {explanation['confidence']:.2f})",
             ha='center', fontsize=12, bbox=dict(facecolor='#ddfcdd', alpha=0.8))

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def train_with_early_stopping(model, tokenizer, datasets, config, metrics_tracker, patience=2, min_delta=0.001):
    """
    Train model with early stopping

    Args:
        model: Model to train
        tokenizer: Tokenizer for text processing
        datasets: Dictionary with train, validation, and test datasets
        config: Model configuration
        metrics_tracker: Metrics tracker
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement

    Returns:
        Trained model and best model path
    """
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = torch.utils.data.DataLoader(
        datasets['validation'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = torch.utils.data.DataLoader(
        datasets['test'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    # Set up model checkpoint directory
    checkpoint_dir = os.path.join(metrics_tracker.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_epoch = 0
    no_improvement_count = 0

    # Move model to device
    model = model.to(device)

    # Prepare optimizer and scheduler
    # Different learning rates for base model and other components
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.base_model_lr},
        {'params': model.rationale_extractor.parameters()},
        {'params': model.concept_mapper.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=config.learning_rate, weight_decay=config.weight_decay)

    # Learning rate scheduler with linear warmup and decay
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")

    # Training loop
    print("Starting training with early stopping...")
    global_step = 0

    # Train for each epoch
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        # Train for one epoch
        model.train()
        train_loss = 0.0
        train_steps = 0

        # Use tqdm for progress bar
        from tqdm import tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward and backward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Update weights and learning rate
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                # Standard forward and backward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs['loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Update weights and learning rate
                optimizer.step()
                scheduler.step()

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Track metrics
            train_loss += loss.item()
            train_steps += 1
            global_step += 1

            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

            # Log batch metrics
            if global_step % 100 == 0:
                # Get predictions for accuracy
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == batch['labels']).float().mean().item()

                # Log metrics
                metrics_tracker.update_batch_metrics(
                    {
                        'loss': loss.item(),
                        'accuracy': accuracy,
                        'lr': scheduler.get_last_lr()[0]
                    },
                    global_step,
                    epoch
                )

        # Calculate epoch metrics
        train_loss = train_loss / train_steps if train_steps > 0 else 0

        # Evaluate on validation set
        val_results = evaluate_model(model, val_loader)
        val_loss = val_results['loss']

        # Log metrics
        metrics_tracker.update_epoch_metrics(
            {'loss': train_loss / max(1, train_steps)},
            epoch,
            'train'
        )
        metrics_tracker.update_epoch_metrics(
            val_results,
            epoch,
            'val'
        )

        # Print progress
        print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_results['accuracy']:.4f}")

        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_val_epoch = epoch
            no_improvement_count = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")
        else:
            no_improvement_count += 1
            print(f"  No improvement in validation loss for {no_improvement_count} epochs")

            # Check if we should stop early
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_epoch+1}")
                break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate on test set
    test_results = evaluate_model(model, test_loader)

    # Log test metrics
    metrics_tracker.update_epoch_metrics(
        test_results,
        config.num_epochs,
        'test'
    )

    print(f"\nTest results:")
    print(f"  Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")

    return model, best_model_path

def run_experiment(dataset_name, args):
    """Run experiment on a specific dataset"""
    print(f"\n{'='*80}")
    print(f"Running experiment on {dataset_name} dataset")
    print(f"{'='*80}")

    # Set random seed
    set_seed(args.seed)

    # Create model configuration
    config = create_config(args, dataset_name)

    # Print basic information
    print(f"Training rationale-concept bottleneck model on {dataset_name}")
    print(f"Using device: {device}")
    print(f"Model: {config.base_model_name}")
    print(f"Number of concepts: {config.num_concepts}")
    print(f"Using LoRA: {config.use_lora}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Initialize output directory for visualizations
    timestamp = time.strftime("%Y%m%d-%H%M%S") if not args.inference_only else "inference"
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Dataset-specific class names and examples
    class_names, examples = get_dataset_examples(dataset_name)

    # Load model for inference only or train new model
    if args.inference_only and args.model_path:
        print(f"\nLoading model from {args.model_path} for inference...")
        model = RationaleConceptBottleneckModel(config)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        # Load and preprocess dataset
        datasets = load_and_process_dataset(dataset_name, tokenizer, config)

        # Limit dataset sizes if specified
        if args.max_train_samples is not None:
            print(f"Limiting training set to {args.max_train_samples} samples")
            datasets['train'] = datasets['train'].select(range(min(args.max_train_samples, len(datasets['train']))))

        if args.max_val_samples is not None:
            print(f"Limiting validation set to {args.max_val_samples} samples")
            datasets['validation'] = datasets['validation'].select(range(min(args.max_val_samples, len(datasets['validation']))))

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(config)

        # Save configuration explicitly
        config_path = os.path.join(metrics_tracker.output_dir, 'config.json')
        config.save(config_path)
        print(f"Configuration saved to: {config_path}")

        # Create model
        model = RationaleConceptBottleneckModel(config)

        # Train model
        print("\nStarting training...")

        if args.early_stopping:
            print(f"Early stopping enabled (patience={args.patience}, min_delta={args.min_delta})")
            # Use our custom training function with early stopping
            model, best_model_path = train_with_early_stopping(
                model=model,
                tokenizer=tokenizer,
                datasets=datasets,
                config=config,
                metrics_tracker=metrics_tracker,
                patience=args.patience,
                min_delta=args.min_delta
            )
        else:
            # Use the original training function
            model, best_model_path = train_model(
                model=model,
                tokenizer=tokenizer,
                train_dataset=datasets['train'],
                val_dataset=datasets['validation'],
                test_dataset=datasets['test'],
                config=config,
                metrics_tracker=metrics_tracker
            )

        # Print results
        print(f"\nTraining complete! Best model saved to: {best_model_path}")
        print(f"All outputs saved to: {metrics_tracker.output_dir}")

        # Print config path and ablation command
        config_path = os.path.join(metrics_tracker.output_dir, 'config.json')
        print(f"Configuration saved to: {config_path}")
        print("\nTo run ablation analysis on this model, use:")
        print(f"python ablation_analysis.py --dataset {dataset_name} \\")
        print(f"  --model_path {best_model_path} \\")
        print(f"  --config_path {config_path}")

        # Update output directory for visualizations
        output_dir = metrics_tracker.output_dir

    # Generate explanations and visualizations
    if args.visualize or args.inference_only:
        # Create visualization directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Generate and visualize explanations
        print("\nGenerating explanations and visualizations...")
        explanations = []

        for i, text in enumerate(examples):
            # Generate explanation
            explanation = model.explain_prediction(tokenizer, text)
            explanations.append({
                'text': text,
                'explanation': explanation,
                'class_name': class_names.get(explanation['prediction'], f'Class {explanation["prediction"]}')
            })

            # Visualize explanation
            save_path = os.path.join(viz_dir, f"example_{i+1}_explanation.png")
            visualize_explanation(text, explanation, class_names, save_path)
            predicted_class = explanation['prediction']
            print(f"  Explanation {i+1}: {class_names.get(predicted_class, f'Class {predicted_class}')} (saved to {save_path})")

        # Save explanations to JSON file
        explanations_file = os.path.join(viz_dir, "example_explanations.json")
        with open(explanations_file, 'w') as f:
            json.dump(explanations, f, indent=2, cls=NumpyEncoder)
        print(f"  Explanations saved to: {explanations_file}")

        # Create a summary visualization
        summary_path = os.path.join(viz_dir, "explanations_summary.png")
        create_explanations_summary(explanations, summary_path)
        print(f"  Summary visualization saved to: {summary_path}")

        print(f"\nAll visualizations saved to: {viz_dir}")

        # Try to open the summary visualization
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(summary_path)}")
        except:
            pass

    print(f"\nExperiment on {dataset_name} complete!")
    return model

def create_explanations_summary(explanations, save_path):
    """Create a summary visualization of multiple explanations"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # Define grid layout
    num_examples = min(len(explanations), 3)  # Show up to 3 examples
    gs = plt.GridSpec(num_examples, 2, figure=fig)

    # Process each explanation
    for i, ex in enumerate(explanations[:num_examples]):
        text = ex['text']
        explanation = ex['explanation']
        class_name = ex['class_name']

        # 1. Visualize rationale for this example
        ax1 = fig.add_subplot(gs[i, 0])

        # Extract rationale tokens
        words = text.split()
        rationale_words = explanation['rationale'].split()

        # Create a binary mask for highlighting
        mask = []
        for word in words:
            if any(r.lower().startswith(word.lower()) for r in rationale_words):
                mask.append(1)
            else:
                mask.append(0)

        # Create text visualization with highlighted rationale
        highlight_color = "#ffcccc"  # Light red
        normal_color = "#f2f2f2"    # Light gray

        # Plot text
        for j, (word, is_rationale) in enumerate(zip(words, mask)):
            color = highlight_color if is_rationale else normal_color
            ax1.text(j, 0, word,
                    bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                    ha='center', va='center', fontsize=8)

        # Hide axes but keep frame
        ax1.set_xlim(-1, len(words))
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"Example {i+1}: {class_name} (Rationale: {explanation['rationale_percentage']:.1%} of text)", fontsize=12)

        # 2. Visualize top concepts for this example
        ax2 = fig.add_subplot(gs[i, 1])

        # Get top concepts
        concepts = explanation["top_concepts"]
        if concepts:
            concept_names = [c[0] for c in concepts]
            concept_scores = [c[1] for c in concepts]

            # Sort by score
            sorted_idx = np.argsort(concept_scores)
            concept_names = [concept_names[i] for i in sorted_idx]
            concept_scores = [concept_scores[i] for i in sorted_idx]

            # Create horizontal bar chart
            ax2.barh(concept_names, concept_scores, color='skyblue')
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Concept Score')
            ax2.set_title(f"Top Concepts for {class_name}", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "No active concepts found",
                     ha='center', va='center', fontsize=12)

    # Add overall title
    plt.suptitle("Explanation Summary", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def get_dataset_examples(dataset_name):
    """Get class names and example texts for a dataset"""
    if dataset_name == "yelp_polarity":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        examples = [
            "This restaurant was terrible. The food was cold and the service was slow.",
            "I absolutely loved this place! The staff was friendly and the food was amazing.",
            "Worst experience ever. Would not recommend to anyone.",
            "Great atmosphere and delicious food. Will definitely come back!"
        ]
    elif dataset_name == "dbpedia":
        class_names = {
            0: "Company", 1: "Educational Institution", 2: "Artist",
            3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
            6: "Building", 7: "Natural Place", 8: "Village",
            9: "Animal", 10: "Plant", 11: "Album",
            12: "Film", 13: "Written Work"
        }
        examples = [
            "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
            "Harvard University is a private Ivy League research university in Cambridge, Massachusetts.",
            "Leonardo da Vinci was an Italian polymath of the Renaissance whose areas of interest included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, literature, anatomy, geology, astronomy, botany, paleontology, and cartography.",
            "Michael Jordan is an American former professional basketball player and businessman."
        ]
    elif dataset_name == "sst2":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        examples = [
            "The movie was a complete waste of time with terrible acting and a predictable plot.",
            "This film is a masterpiece of storytelling with amazing performances and stunning visuals.",
            "I couldn't stand the slow pacing and convoluted storyline of this boring movie.",
            "The director's vision shines through in every scene, creating a truly memorable cinematic experience."
        ]
    elif dataset_name == "agnews":
        class_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        examples = [
            "Oil prices fell more than $1 a barrel on Friday on concerns that Hurricane Ivan will miss most oil and gas production in the Gulf of Mexico.",
            "Yankees catcher Jorge Posada was suspended for three games and Angels shortstop David Eckstein was penalized one game for their actions in a game this week.",
            "Microsoft Corp. is accelerating the schedule for its next Windows operating system update, code-named Longhorn, and scaling back some features to meet its new timetable.",
            "Scientists in the United States say they have developed a new type of drone aircraft that flies by flapping its wings rather than using an engine."
        ]
    else:
        class_names = {}
        examples = []

    return class_names, examples

def main():
    """Main function for training and evaluating the model"""
    args = parse_arguments()

    if args.dataset == "all":
        # Run experiments on all datasets
        print("\nRunning experiments on all datasets...")
        datasets = ["yelp_polarity", "dbpedia", "sst2", "agnews"]
        for dataset in datasets:
            try:
                run_experiment(dataset, args)
            except Exception as e:
                print(f"\nError running experiment on {dataset}: {str(e)}")
                print("Continuing with next dataset...")
    else:
        # Run experiment on selected dataset
        model = run_experiment(args.dataset, args)

    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()
