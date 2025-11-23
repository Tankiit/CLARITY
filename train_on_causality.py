#!/usr/bin/env python
"""
Train Rationale-Concept Model on Causality Datasets

This script trains the Rationale-Concept Bottleneck Model on causal reasoning datasets
including COPA, e-CARE, and Balanced COPA. These datasets test causal reasoning abilities
by requiring the model to identify cause-effect relationships.

Supported datasets:
- COPA (Choice of Plausible Alternatives): Binary choice between two possible causes/effects
- e-CARE (Explainable Causal Reasoning): Causal reasoning with natural language explanations
- Balanced COPA: COPA without superficial cues that models might exploit

References:
- COPA: https://huggingface.co/datasets/super_glue (copa subset)
- e-CARE: https://huggingface.co/datasets/12ml/e-CARE
- Balanced COPA: https://huggingface.co/datasets/pkavumba/balanced-copa
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from optimized_rationale_concept_model import (
    RationaleConceptBottleneckModel,
    ModelConfig as BaseModelConfig,
    MetricsTracker,
    evaluate_model
)

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


class CausalityModelConfig(BaseModelConfig):
    """Extended ModelConfig with causality-specific parameters"""
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
        # Causality-specific parameters
        causality_dataset="copa",
        question_type="cause",  # 'cause' or 'effect' for COPA
        include_question_context=True,
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
        self.causality_dataset = causality_dataset
        self.question_type = question_type
        self.include_question_context = include_question_context
        self.gradient_accumulation_steps = gradient_accumulation_steps


class COPADataset(Dataset):
    """
    Dataset wrapper for COPA (Choice of Plausible Alternatives)

    COPA format:
    - premise: The situation or event
    - choice1: First possible cause/effect
    - choice2: Second possible cause/effect
    - question: "cause" or "effect" indicating what to find
    - label: 0 or 1 indicating correct choice

    We convert this to binary classification by creating separate examples
    for each choice concatenated with the premise.
    """
    def __init__(self, dataset, tokenizer, max_length=128, include_question_context=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_question_context = include_question_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        premise = item['premise']
        choice1 = item['choice1']
        choice2 = item['choice2']
        question = item['question']  # 'cause' or 'effect'
        label = item['label']  # 0 or 1

        # Create combined text input
        # Format: [CLS] premise [SEP] Question: What is the {question}? [SEP]
        #         A: choice1 [SEP] B: choice2 [SEP]
        if self.include_question_context:
            if question == 'cause':
                question_text = "What was the cause of this?"
            else:
                question_text = "What happened as a result?"

            text = f"{premise} {question_text} A: {choice1} B: {choice2}"
        else:
            text = f"{premise} A: {choice1} B: {choice2}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ECareDataset(Dataset):
    """
    Dataset wrapper for e-CARE (Explainable Causal Reasoning)

    e-CARE format:
    - premise: The causal premise
    - hypothesis1: First candidate hypothesis
    - hypothesis2: Second candidate hypothesis
    - label: 0 or 1 indicating which hypothesis is correct
    - (optional) conceptual_explanation: Natural language explanation
    """
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        premise = item.get('premise', item.get('context', ''))

        # Handle different column naming conventions
        if 'hypothesis1' in item:
            choice1 = item['hypothesis1']
            choice2 = item['hypothesis2']
        elif 'ask-for' in item:
            # Some versions have different format
            choice1 = item.get('choice1', item.get('answer1', ''))
            choice2 = item.get('choice2', item.get('answer2', ''))
        else:
            choice1 = item.get('choice1', '')
            choice2 = item.get('choice2', '')

        label = item.get('label', 0)

        # Create combined text
        text = f"Premise: {premise} Hypothesis A: {choice1} Hypothesis B: {choice2}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BalancedCOPADataset(Dataset):
    """
    Dataset wrapper for Balanced COPA

    Same format as COPA but with balanced examples to prevent
    models from exploiting superficial cues.
    """
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        premise = item['premise']
        choice1 = item['choice1']
        choice2 = item['choice2']
        question = item.get('question', 'effect')
        label = item['label']

        # Format the input
        if question == 'cause':
            context = "What was the cause?"
        else:
            context = "What was the effect?"

        text = f"{premise} {context} Choice A: {choice1} Choice B: {choice2}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_causality_dataset(dataset_name, tokenizer, config):
    """
    Load and preprocess a causality dataset

    Args:
        dataset_name: Name of the causality dataset ('copa', 'e_care', 'balanced_copa')
        tokenizer: Tokenizer for text processing
        config: Model configuration

    Returns:
        Dictionary with processed train, validation, and test datasets
    """
    logger.info(f"Loading causality dataset: {dataset_name}")

    cache_dir = os.path.join("cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    if dataset_name == 'copa':
        # Load COPA from SuperGLUE
        dataset = load_dataset('super_glue', 'copa', cache_dir=cache_dir)

        train_dataset = COPADataset(
            dataset['train'], tokenizer, config.max_seq_length,
            include_question_context=config.include_question_context
        )
        val_dataset = COPADataset(
            dataset['validation'], tokenizer, config.max_seq_length,
            include_question_context=config.include_question_context
        )
        # COPA doesn't have labeled test set, use validation for testing
        test_dataset = val_dataset

    elif dataset_name == 'e_care':
        # Load e-CARE dataset
        try:
            dataset = load_dataset('12ml/e-CARE', cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Could not load e-CARE from HuggingFace: {e}")
            logger.info("Attempting to load from alternative source...")
            # Try alternative loading
            dataset = load_dataset('csv', data_files={
                'train': 'data/e_care_train.csv',
                'validation': 'data/e_care_dev.csv',
                'test': 'data/e_care_test.csv'
            }, cache_dir=cache_dir)

        train_dataset = ECareDataset(dataset['train'], tokenizer, config.max_seq_length)
        val_dataset = ECareDataset(dataset['validation'], tokenizer, config.max_seq_length)
        test_dataset = ECareDataset(dataset['test'], tokenizer, config.max_seq_length)

    elif dataset_name == 'balanced_copa':
        # Load Balanced COPA
        dataset = load_dataset('pkavumba/balanced-copa', cache_dir=cache_dir)

        train_dataset = BalancedCOPADataset(dataset['train'], tokenizer, config.max_seq_length)
        val_dataset = BalancedCOPADataset(dataset['validation'], tokenizer, config.max_seq_length)
        test_dataset = BalancedCOPADataset(dataset['test'], tokenizer, config.max_seq_length)

    else:
        raise ValueError(f"Unknown causality dataset: {dataset_name}. "
                        f"Supported: copa, e_care, balanced_copa")

    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }


def train_causality_model(model, tokenizer, datasets, config, metrics_tracker):
    """
    Train and evaluate the model on causality data

    Args:
        model: Model to train
        tokenizer: Tokenizer for text processing
        datasets: Dictionary with train/validation/test datasets
        config: Model configuration
        metrics_tracker: Metrics tracker

    Returns:
        Trained model and best model path
    """
    # Create data loaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        datasets['validation'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    # Move model to device
    model = model.to(device)

    # Prepare optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.base_model_lr},
        {'params': model.rationale_extractor.parameters(), 'lr': config.learning_rate},
        {'params': model.concept_mapper.parameters(), 'lr': config.learning_rate},
        {'params': model.classifier.parameters(), 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)

    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Setup checkpointing
    checkpoint_dir = os.path.join(metrics_tracker.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_causality_model.pt')

    logger.info("Starting causality model training...")
    best_val_accuracy = 0.0
    global_step = 0

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        for batch_idx, batch in enumerate(train_pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']

                # Backward pass with gradient scaling
                scaler.scale(loss / config.gradient_accumulation_steps).backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs['loss']

                (loss / config.gradient_accumulation_steps).backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Track metrics
            train_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            train_correct += (preds == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)
            global_step += 1

            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        val_results = evaluate_causality_model(model, val_loader)

        # Log metrics
        metrics_tracker.update_epoch_metrics({'loss': train_loss, 'accuracy': train_accuracy}, epoch, 'train')
        metrics_tracker.update_epoch_metrics(val_results, epoch, 'val')

        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Val   - Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")

        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model! Accuracy: {best_val_accuracy:.4f}")

        # Generate plots
        metrics_tracker.generate_training_plots()

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))

    # Final evaluation
    logger.info("Evaluating on test set...")
    test_results = evaluate_causality_model(model, test_loader)
    metrics_tracker.update_epoch_metrics(test_results, config.num_epochs, 'test')

    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}")
    logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"  F1 Score: {test_results['f1']:.4f}")

    # Save summary
    report_path = metrics_tracker.export_summary_report()
    logger.info(f"Training summary saved to: {report_path}")

    return model, best_model_path


def evaluate_causality_model(model, dataloader):
    """
    Evaluate model on causality data

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            preds = torch.argmax(outputs['logits'], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_losses.append(outputs['loss'].item())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    avg_loss = sum(all_losses) / len(all_losses)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def analyze_causal_concepts(model, tokenizer, datasets, config, output_dir):
    """
    Analyze what causal concepts the model has learned

    This function examines the concept activations for cause vs effect
    questions to understand what the model has learned about causality.
    """
    logger.info("Analyzing causal concepts...")

    model.eval()

    # Collect concept activations by question type (for COPA)
    cause_concepts = []
    effect_concepts = []

    # Sample from validation set
    val_loader = DataLoader(datasets['validation'], batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100:  # Analyze first 100 examples
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            concept_probs = outputs['concept_probs'].cpu().numpy()[0]

            # Get the original item to check question type
            if hasattr(datasets['validation'], 'dataset'):
                orig_item = datasets['validation'].dataset[i]
                question_type = orig_item.get('question', 'unknown')

                if question_type == 'cause':
                    cause_concepts.append(concept_probs)
                else:
                    effect_concepts.append(concept_probs)

    # Analyze concept differences
    analysis_results = {}

    if cause_concepts and effect_concepts:
        cause_mean = np.mean(cause_concepts, axis=0)
        effect_mean = np.mean(effect_concepts, axis=0)

        # Find concepts that differ most between cause/effect
        diff = np.abs(cause_mean - effect_mean)
        top_diff_concepts = np.argsort(diff)[::-1][:10]

        analysis_results['cause_effect_concept_differences'] = {
            f'concept_{i}': {
                'cause_activation': float(cause_mean[i]),
                'effect_activation': float(effect_mean[i]),
                'difference': float(diff[i])
            }
            for i in top_diff_concepts
        }

        logger.info("Top concepts distinguishing cause vs effect:")
        for i in top_diff_concepts[:5]:
            logger.info(f"  Concept {i}: cause={cause_mean[i]:.3f}, effect={effect_mean[i]:.3f}")

    # Save analysis
    analysis_path = os.path.join(output_dir, 'causal_concept_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    logger.info(f"Causal concept analysis saved to: {analysis_path}")

    return analysis_results


def main():
    """Main entry point for causality training"""
    parser = argparse.ArgumentParser(
        description="Train a rationale-concept bottleneck model on causality datasets"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="copa",
                        choices=["copa", "e_care", "balanced_copa"],
                        help="Causality dataset to train on")

    # Model arguments
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Base model name")
    parser.add_argument("--num_concepts", type=int, default=50,
                        help="Number of concepts in the bottleneck")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (smaller for causality due to longer sequences)")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length (longer for premise + choices)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--base_model_lr", type=float, default=1e-5,
                        help="Learning rate for base model")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")

    # Architecture arguments
    parser.add_argument("--enable_concept_interactions", action="store_true",
                        help="Enable concept interaction matrix")
    parser.add_argument("--disable_skip_connection", action="store_true",
                        help="Disable skip connection")
    parser.add_argument("--no_question_context", action="store_true",
                        help="Don't include question type in input (COPA only)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="causality_models",
                        help="Directory to save model and outputs")

    # Analysis options
    parser.add_argument("--analyze_concepts", action="store_true",
                        help="Analyze causal concept activations after training")

    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create configuration
    config = CausalityModelConfig(
        base_model_name=args.model,
        num_labels=2,  # All causality tasks are binary
        num_concepts=args.num_concepts,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        base_model_lr=args.base_model_lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        enable_concept_interactions=args.enable_concept_interactions,
        use_skip_connection=not args.disable_skip_connection,
        causality_dataset=args.dataset,
        include_question_context=not args.no_question_context,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Load causality dataset
    datasets = load_causality_dataset(args.dataset, tokenizer, config)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)

    # Save configuration
    config_path = os.path.join(metrics_tracker.output_dir, 'causality_config.json')
    config.save(config_path)

    # Initialize model
    model = RationaleConceptBottleneckModel(config)

    # Train model
    model, best_model_path = train_causality_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        config=config,
        metrics_tracker=metrics_tracker
    )

    # Analyze causal concepts if requested
    if args.analyze_concepts:
        analyze_causal_concepts(
            model, tokenizer, datasets, config,
            metrics_tracker.output_dir
        )

    logger.info(f"Training complete! Best model: {best_model_path}")
    logger.info(f"All outputs: {metrics_tracker.output_dir}")


if __name__ == "__main__":
    main()
