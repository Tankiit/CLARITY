#!/usr/bin/env python
# CEBaB 50-Epoch Training Script
# Standalone script for focused CEBaB training with CLARITY Credal CBM

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the CLARITY Credal CBM model
from clarity_credal_cbm import ClarityCredalCBM

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
print(f"Using device: {device}")


class CEBABDataset(Dataset):
    # Dataset wrapper for CEBAB
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get text and label
        text = item['description']  # CEBaB uses 'description' key

        # Use the review majority as the label
        review_majority = item['review_majority']

        # Handle different label formats
        if isinstance(review_majority, str):
            if review_majority == 'no majority':
                label = 1  # Default to positive
            else:
                try:
                    label_value = int(review_majority)
                    # Map ratings to binary: 1-2 as negative (0), 3-5 as positive (1)
                    label = 1 if label_value >= 3 else 0
                except ValueError:
                    label = 1
        else:
            try:
                label_value = int(review_majority)
                label = 1 if label_value >= 3 else 0
            except (ValueError, TypeError):
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


def load_cebab_dataset(tokenizer, max_length=128):
    # Load CEBAB dataset
    print("Loading CEBaB dataset...")

    try:
        dataset = load_dataset("CEBaB/CEBaB")
        print(f"Successfully loaded CEBaB from Hugging Face")
        train_split = 'train_inclusive'
        val_split = 'validation'
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        raise

    # Create dataset wrappers
    train_dataset = CEBABDataset(dataset[train_split], tokenizer, max_length)
    val_dataset = CEBABDataset(dataset[val_split], tokenizer, max_length)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    return train_dataset, val_dataset


def train_model():
    # Configuration
    base_model_name = 'distilbert-base-uncased'
    num_concepts = 15
    n_credal_points = 5
    num_classes = 2
    batch_size = 16
    epochs = 50
    learning_rate = 2e-5
    base_model_lr = 1e-5
    max_seq_length = 128

    print("\n" + "="*80)
    print(" "*15 + "CEBaB 50-Epoch Training with CLARITY Credal CBM")
    print("="*80 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load dataset
    train_dataset, val_dataset = load_cebab_dataset(tokenizer, max_seq_length)

    # Create data loaders
    is_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda
    )

    # Initialize model
    print("\nInitializing CLARITY Credal CBM model...")
    concept_names = [f"concept_{i}" for i in range(num_concepts)]

    model = ClarityCredalCBM(
        base_model_name=base_model_name,
        num_concepts=num_concepts,
        num_classes=num_classes,
        n_credal_points=n_credal_points,
        concept_names=concept_names
    ).to(device)

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup mixed precision training for CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using automatic mixed precision (AMP) for faster CUDA training")

    # Optimizer
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': base_model_lr},
        {'params': model.rationale_extractor.parameters(), 'lr': learning_rate},
        {'params': model.concept_mapper.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ], weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print("\n" + "="*80)
    print(f"Training for {epochs} epochs")
    print("="*80 + "\n")

    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{train_correct/train_total:.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_epistemic = []
        all_aleatoric = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, labels)

                val_loss += outputs['loss'].item()
                preds = outputs['logits'].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Collect uncertainty metrics
                epistemic = outputs['uncertainty_metrics']['epistemic'].mean(1)
                aleatoric = outputs['uncertainty_metrics']['aleatoric'].mean(1)
                all_epistemic.extend(epistemic.tolist())
                all_aleatoric.extend(aleatoric.tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        avg_epistemic = np.mean(all_epistemic)
        avg_aleatoric = np.mean(all_aleatoric)
        avg_total = avg_epistemic + avg_aleatoric
        epistemic_ratio = avg_epistemic / avg_total if avg_total > 0 else 0

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Uncertainty:")
        print(f"    Epistemic:   {avg_epistemic:.4f} ({epistemic_ratio:.1%})")
        print(f"    Aleatoric:   {avg_aleatoric:.4f}")
        print(f"    Total:       {avg_total:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'epistemic': avg_epistemic,
                'aleatoric': avg_aleatoric
            }, 'checkpoints/best_model_cebab_50epochs.pt')
            print(f"  [BEST] New best model saved! Accuracy: {val_acc:.4f}")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_model()
