#!/usr/bin/env python
"""
Train Rationale-Concept Model on CEBAB Dataset

This script trains the Rationale-Concept Bottleneck Model on the CEBAB dataset
to improve explanation quality and confidence scores.
"""

import os
import argparse
import json
import torch
import numpy as np
import logging
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset
from optimized_rationale_concept_model import (
    RationaleConceptBottleneckModel,
    ModelConfig as BaseModelConfig,
    MetricsTracker
)

# Extended ModelConfig with our additional parameters
class ModelConfig(BaseModelConfig):
    def __init__(
        self,
        base_model_name="distilbert-base-uncased",
        num_labels=2,
        num_concepts=50,
        hidden_size=768,
        dropout_rate=0.1,
        min_span_size=3,
        max_span_size=20,
        length_bonus_factor=0.01,
        concept_sparsity_weight=0.03,
        concept_diversity_weight=0.01,
        rationale_sparsity_weight=0.03,
        rationale_continuity_weight=0.1,
        classification_weight=1.0,
        target_rationale_percentage=0.2,
        enable_concept_interactions=False,
        use_skip_connection=True,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        batch_size=32,
        max_seq_length=128,
        learning_rate=2e-5,
        base_model_lr=1e-5,
        weight_decay=0.01,
        num_epochs=5,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=42,
        output_dir="models",
        # Additional parameters
        gradient_accumulation_steps=1
    ):
        super().__init__(
            base_model_name=base_model_name,
            num_labels=num_labels,
            num_concepts=num_concepts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            min_span_size=min_span_size,
            max_span_size=max_span_size,
            length_bonus_factor=length_bonus_factor,
            concept_sparsity_weight=concept_sparsity_weight,
            concept_diversity_weight=concept_diversity_weight,
            rationale_sparsity_weight=rationale_sparsity_weight,
            rationale_continuity_weight=rationale_continuity_weight,
            classification_weight=classification_weight,
            target_rationale_percentage=target_rationale_percentage,
            enable_concept_interactions=enable_concept_interactions,
            use_skip_connection=use_skip_connection,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            learning_rate=learning_rate,
            base_model_lr=base_model_lr,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            seed=seed,
            output_dir=output_dir
        )
        # Add our additional parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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

class CEBABDataset(Dataset):
    """Dataset wrapper for CEBAB"""
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get text and label
        text = item['description']  # CEBaB uses 'description' key for the review text

        # Use the review majority as the label
        # CEBaB uses numeric values or string values
        # Convert to binary classification (positive/negative)
        review_majority = item['review_majority']

        # Handle different label formats
        if isinstance(review_majority, str):
            if review_majority == 'no majority':
                # For 'no majority', we'll default to positive (1)
                label = 1
            else:
                # Try to convert string numbers to integers
                try:
                    label_value = int(review_majority)
                    # Map ratings to binary: 1-2 as negative (0), 3-5 as positive (1)
                    label = 1 if label_value >= 3 else 0
                except ValueError:
                    # If conversion fails, default to positive
                    label = 1
        else:
            # If it's already a number
            try:
                label_value = int(review_majority)
                # Map ratings to binary: 1-2 as negative (0), 3-5 as positive (1)
                label = 1 if label_value >= 3 else 0
            except (ValueError, TypeError):
                # If conversion fails, default to positive
                label = 1

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

        # Add label
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding

def evaluate_model(model, dataloader):
    """
    Evaluate model on a data loader with detailed metrics

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data

    Returns:
        Dictionary with evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()

    all_labels = []
    all_preds = []
    all_losses = []

    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Get predictions
            logits = outputs['logits']
            loss = outputs['loss']

            preds = torch.argmax(logits, dim=1)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_losses.append(loss.item())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # For binary classification, use binary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate class distribution
    label_counts = {}
    for label in set(all_labels):
        label_counts[f"class_{label}"] = all_labels.count(label)

    # Calculate average loss
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0

    # Log detailed metrics
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Total samples: {len(all_labels)}")
    logger.info(f"  Label distribution: {label_counts}")
    logger.info(f"  Confusion matrix:\n{cm}")

    # Return all metrics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'label_counts': label_counts,
        'confusion_matrix': cm.tolist()
    }

def load_cebab_dataset(tokenizer, max_length=128, max_train_samples=None, max_val_samples=None, max_test_samples=None, balance_method='none'):
    """Load and preprocess CEBAB dataset

    Args:
        tokenizer: Tokenizer to use for preprocessing
        max_length: Maximum sequence length for tokenization
        max_train_samples: Maximum number of training samples to use (None for all)
        max_val_samples: Maximum number of validation samples to use (None for all)
        max_test_samples: Maximum number of test samples to use (None for all)
    """
    logger.info("Loading CEBAB dataset...")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("CEBaB/CEBaB")
        logger.info("Successfully loaded CEBAB from Hugging Face")

        # CEBAB has different split names
        # Use 'train_inclusive' as the training set
        train_split = 'train_inclusive'
        val_split = 'validation'
        test_split = 'test'

        logger.info(f"Available splits: {list(dataset.keys())}")
    except Exception as e:
        logger.warning(f"Error loading from Hugging Face: {e}")
        logger.info("Trying to load from local files...")

        # Try loading from local files
        dataset = {
            'train_inclusive': load_dataset('json', data_files='train_inclusive.json', split='train'),
            'validation': load_dataset('json', data_files='dev.json', split='train'),
            'test': load_dataset('json', data_files='test.json', split='train')
        }
        logger.info("Successfully loaded CEBAB from local files")

        # Set split names for local files
        train_split = 'train_inclusive'
        val_split = 'validation'
        test_split = 'test'

    # Check class balance
    def get_class_distribution(dataset_split):
        """Get class distribution for a dataset split"""
        labels = [item['review_majority'] for item in dataset_split]
        unique_labels = set(labels)
        distribution = {label: labels.count(label) for label in unique_labels}
        return distribution

    # Log class distribution
    train_dist = get_class_distribution(dataset[train_split])
    logger.info(f"Original training set class distribution: {train_dist}")

    # Balance dataset if requested
    def balance_dataset(dataset_split, balance_method='undersample'):
        """Balance a dataset by undersampling or oversampling"""
        # Get class counts
        labels = [item['review_majority'] for item in dataset_split]
        class_counts = {}
        for label in set(labels):
            class_counts[label] = labels.count(label)

        # Find minority and majority class
        min_class = min(class_counts, key=class_counts.get)
        min_count = class_counts[min_class]

        # Create balanced dataset
        balanced_indices = []

        if balance_method == 'undersample':
            # Undersample: Take all minority class samples and randomly sample from majority class
            for label in class_counts:
                # Get indices for this class
                class_indices = [i for i, item in enumerate(dataset_split) if item['review_majority'] == label]

                # If this is majority class, sample down to minority class size
                if len(class_indices) > min_count:
                    import random
                    random.seed(42)  # For reproducibility
                    class_indices = random.sample(class_indices, min_count)

                balanced_indices.extend(class_indices)
        else:  # oversample
            # Oversample: Take all majority class samples and oversample minority class
            max_class = max(class_counts, key=class_counts.get)
            max_count = class_counts[max_class]

            for label in class_counts:
                # Get indices for this class
                class_indices = [i for i, item in enumerate(dataset_split) if item['review_majority'] == label]

                # If this is minority class, oversample to majority class size
                if len(class_indices) < max_count:
                    import random
                    random.seed(42)  # For reproducibility
                    # Oversample with replacement
                    oversampled_indices = random.choices(class_indices, k=max_count - len(class_indices))
                    class_indices.extend(oversampled_indices)

                balanced_indices.extend(class_indices)

        # Create new balanced dataset
        balanced_dataset = dataset_split.select(balanced_indices)
        return balanced_dataset

    # Balance dataset if requested
    if balance_method != 'none':
        # Balance all splits
        for split_name in [train_split, val_split, test_split]:
            logger.info(f"Balancing {split_name} dataset using method: {balance_method}")
            dataset[split_name] = balance_dataset(dataset[split_name], balance_method)
            balanced_dist = get_class_distribution(dataset[split_name])
            logger.info(f"Balanced {split_name} class distribution: {balanced_dist}")

    # Limit dataset sizes if specified
    if max_train_samples is not None:
        logger.info(f"Limiting training set to {max_train_samples} samples")
        dataset[train_split] = dataset[train_split].select(range(min(max_train_samples, len(dataset[train_split]))))

    if max_val_samples is not None:
        logger.info(f"Limiting validation set to {max_val_samples} samples")
        dataset[val_split] = dataset[val_split].select(range(min(max_val_samples, len(dataset[val_split]))))

    if max_test_samples is not None:
        logger.info(f"Limiting test set to {max_test_samples} samples")
        dataset[test_split] = dataset[test_split].select(range(min(max_test_samples, len(dataset[test_split]))))

    # Create dataset wrappers
    train_dataset = CEBABDataset(dataset[train_split], tokenizer, max_length)
    val_dataset = CEBABDataset(dataset[val_split], tokenizer, max_length)
    test_dataset = CEBABDataset(dataset[test_split], tokenizer, max_length)

    # Log dataset sizes
    logger.info(f"Final dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Validation: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def train_model(model, train_dataset, val_dataset, test_dataset, config, metrics_tracker):
    """Train the model on CEBAB dataset"""
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 4) if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else None,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch size for evaluation
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4) if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else None,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,  # Larger batch size for evaluation
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4) if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else None,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    # Move model to device
    model = model.to(device)

    # Set up mixed precision training if available
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using automatic mixed precision training")

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.base_model_lr},
        {'params': model.rationale_extractor.parameters()},
        {'params': model.concept_mapper.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=config.learning_rate, weight_decay=config.weight_decay)

    # Learning rate scheduler with linear warmup and decay
    # Adjust for gradient accumulation
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Set up model checkpoint directory
    checkpoint_dir = os.path.join(metrics_tracker.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {effective_batch_size} (batch_size={config.batch_size} × accumulation_steps={config.gradient_accumulation_steps})")
    global_step = 0
    best_val_accuracy = 0.0

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        optimizer.zero_grad(set_to_none=True)  # Zero gradients at the start of epoch

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for step, batch in enumerate(train_pbar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward and backward pass with mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss'] / config.gradient_accumulation_steps  # Normalize loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                # Standard forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs['loss'] / config.gradient_accumulation_steps  # Normalize loss
                loss.backward()

            # Track metrics (use the original loss value for logging)
            train_loss += outputs['loss'].item()  # Add the full loss, not the normalized one
            train_steps += 1

            # Update weights after accumulating gradients
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                    # Update weights and learning rate with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': outputs['loss'].item(),
                    'lr': scheduler.get_last_lr()[0]
                })

                # Log batch metrics periodically
                if global_step % 20 == 0:
                    # Get predictions for accuracy
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == batch['labels']).float().mean().item()

                    # Log metrics
                    metrics_tracker.update_batch_metrics(
                        {
                            'loss': outputs['loss'].item(),
                            'accuracy': accuracy,
                            'lr': scheduler.get_last_lr()[0]
                        },
                        global_step,
                        epoch
                    )

        # Calculate epoch metrics
        train_loss = train_loss / train_steps if train_steps > 0 else 0

        # Validation
        val_results = evaluate_model(model, val_loader)

        # Log epoch metrics
        metrics_tracker.update_epoch_metrics(
            {'loss': train_loss},
            epoch,
            'train'
        )
        metrics_tracker.update_epoch_metrics(
            val_results,
            epoch,
            'val'
        )

        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")

        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model saved! Accuracy: {best_val_accuracy:.4f}")

        # Generate training plots
        metrics_tracker.generate_training_plots()

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader)

    # Log test metrics
    metrics_tracker.update_epoch_metrics(
        test_results,
        config.num_epochs,
        'test'
    )

    logger.info(f"Test results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")
    if 'f1' in test_results:
        logger.info(f"  F1 Score: {test_results['f1']:.4f}")

    # Save final summary report
    report_path = metrics_tracker.export_summary_report()
    logger.info(f"Training summary saved to: {report_path}")

    return model, best_model_path

def main():
    parser = argparse.ArgumentParser(description="Train Rationale-Concept Model on CEBAB")

    # Model arguments
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Base model name")
    parser.add_argument("--num_concepts", type=int, default=50,
                        help="Number of concepts in the bottleneck")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--base_model_lr", type=float, default=1e-5,
                        help="Learning rate for base model")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay to apply to parameters")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping")

    # Dataset arguments
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training examples to use (for faster training)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Maximum number of validation examples to use")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum number of test examples to use")
    parser.add_argument("--balance_dataset", choices=['none', 'undersample', 'oversample'], default='none',
                        help="Balance dataset classes: none, undersample (reduce majority), or oversample (increase minority)")

    # Fast development mode
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run a quick development run with limited data and epochs")

    # Architecture arguments
    parser.add_argument("--enable_concept_interactions", action="store_true",
                        help="Enable concept interaction matrix")
    parser.add_argument("--disable_skip_connection", action="store_true",
                        help="Disable skip connection from encoder to classifier")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="cebab_models",
                        help="Directory to save model and outputs")

    args = parser.parse_args()

    # Apply fast development mode settings if enabled
    if args.fast_dev_run:
        logger.info("Fast development run enabled - using optimized settings for quick iteration")
        args.max_train_samples = args.max_train_samples or 5000  # Using 5000 samples as requested
        args.max_val_samples = args.max_val_samples or 500       # Increased validation samples
        args.max_test_samples = args.max_test_samples or 500     # Increased test samples

        # Only override epochs if not explicitly set by user
        if args.num_epochs == 5:  # Default value
            args.num_epochs = 20  # Using 20 epochs as requested

        args.max_seq_length = min(args.max_seq_length, 128)      # Increased sequence length
        args.batch_size = max(args.batch_size, 32)               # Larger batch size for faster iteration

    logger.info(f"Training with {args.num_epochs} epochs on {args.max_train_samples or 'all'} training samples")

    # Set random seeds
    set_seed(args.seed)

    # Create model configuration
    config = ModelConfig(
        base_model_name=args.model,
        num_labels=2,  # CEBAB has 2 classes (Positive/Negative)
        num_concepts=args.num_concepts,
        enable_concept_interactions=args.enable_concept_interactions,
        use_skip_connection=not args.disable_skip_connection,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        base_model_lr=args.base_model_lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        # Add new parameters
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Load CEBAB dataset with sample limits for faster training
    train_dataset, val_dataset, test_dataset = load_cebab_dataset(
        tokenizer,
        max_length=config.max_seq_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        balance_method=args.balance_dataset
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)

    # Save configuration
    config_path = os.path.join(metrics_tracker.output_dir, 'config.json')
    config.save(config_path)

    # Initialize model
    model = RationaleConceptBottleneckModel(config)

    # Train model
    model, best_model_path = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
        metrics_tracker=metrics_tracker
    )

    logger.info(f"Training complete! Best model saved to: {best_model_path}")
    logger.info(f"All outputs saved to: {metrics_tracker.output_dir}")

if __name__ == "__main__":
    main()
