# Complete Training Pipeline for CLARITY Credal CBM
# Demonstrates training on text classification with uncertainty quantification

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Assuming ClarityCredalCBM is imported from previous artifact
from clarity_credal_cbm import ClarityCredalCBM


class TextClassificationDataset(torch.utils.data.Dataset):
    # Dataset wrapper for text classification

    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label']),
            'text': item['text']
        }


def train_clarity_credal_cbm(
    model,
    train_loader,
    val_loader,
    tokenizer,
    epochs=10,
    lr=2e-5,
    device='cpu'
):
    # Training loop for CLARITY Credal CBM
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},  # Lower LR for encoder
        {'params': model.rationale_extractor.parameters(), 'lr': lr},
        {'params': model.concept_mapper.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_epistemic': [], 'val_aleatoric': []
    }

    best_val_acc = 0.0

    print("=" * 80)
    print(" " * 20 + "TRAINING CLARITY CREDAL CBM")
    print("=" * 80)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

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

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_epistemic = []
        val_aleatoric = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, labels)

                val_loss += outputs['loss'].item()
                preds = outputs['logits'].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Collect uncertainty metrics
                val_epistemic.extend(
                    outputs['uncertainty_metrics']['epistemic'].mean(1).tolist()
                )
                val_aleatoric.extend(
                    outputs['uncertainty_metrics']['aleatoric'].mean(1).tolist()
                )

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        avg_epistemic = np.mean(val_epistemic)
        avg_aleatoric = np.mean(val_aleatoric)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_epistemic'].append(avg_epistemic)
        history['val_aleatoric'].append(avg_aleatoric)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Uncertainty:")
        print(f"    Epistemic: {avg_epistemic:.4f}")
        print(f"    Aleatoric: {avg_aleatoric:.4f}")
        print(f"    Total:     {avg_epistemic + avg_aleatoric:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_clarity_credal_cbm.pt')
            print(f"  [BEST] New best model saved (acc: {val_acc:.4f})")

        print("-" * 80)

    return model, history


def visualize_training_history(history: Dict):
    # Visualize training metrics including uncertainty

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', marker='o')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Uncertainty decomposition
    axes[1, 0].plot(epochs, history['val_epistemic'],
                   label='Epistemic', marker='o', color='#3498db')
    axes[1, 0].plot(epochs, history['val_aleatoric'],
                   label='Aleatoric', marker='s', color='#e74c3c')
    total_unc = [e + a for e, a in zip(history['val_epistemic'], history['val_aleatoric'])]
    axes[1, 0].plot(epochs, total_unc,
                   label='Total', marker='^', color='purple', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Uncertainty')
    axes[1, 0].set_title('Validation Uncertainty Decomposition')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Uncertainty ratio over time
    axes[1, 1].plot(epochs,
                   [e/(e+a) for e, a in zip(history['val_epistemic'], history['val_aleatoric'])],
                   marker='o', color='#9b59b6')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Epistemic / Total Uncertainty')
    axes[1, 1].set_title('Epistemic Uncertainty Ratio (Should Decrease)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('clarity_credal_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_explanation(model, tokenizer, text: str, true_label: int = None, device='cpu'):
    # Demonstrate explanation generation for a text sample
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get explanation
    explanation = model.explain_prediction(input_ids, attention_mask, tokenizer)

    print("=" * 80)
    print(" " * 25 + "EXPLANATION REPORT")
    print("=" * 80)
    print(f"\nText: {text}")
    print(f"\nPredicted Class: {explanation['predicted_class']}")
    print(f"Confidence: {explanation['confidence']:.4f}")

    if true_label is not None:
        print(f"True Label: {true_label}")
        correct_symbol = '[CORRECT]' if explanation['predicted_class'] == true_label else '[INCORRECT]'
        print(f"Result: {correct_symbol}")

    print(f"\n{'─' * 80}")
    print("RATIONALES (Important Text Spans):")
    print(f"{'─' * 80}")
    for token, importance in explanation['rationale_tokens'][:15]:
        bar = '█' * int(importance * 30)
        print(f"  {token:20s} {bar} {importance:.4f}")

    print(f"\n{'─' * 80}")
    print("TOP 5 UNCERTAIN CONCEPTS:")
    print(f"{'─' * 80}")
    for i, (concept_name, unc_decomp) in enumerate(explanation['top_uncertain_concepts'][:5], 1):
        print(f"\n{i}. {concept_name}")
        print(f"   Epistemic: {unc_decomp['epistemic']:.4f} ({unc_decomp['epistemic_ratio']:.1%})")
        print(f"   Aleatoric: {unc_decomp['aleatoric']:.4f} ({unc_decomp['aleatoric_ratio']:.1%})")
        print(f"   Total:     {unc_decomp['total']:.4f}")

        if unc_decomp['epistemic_ratio'] > 0.7:
            print(f"   [HIGH EPISTEMIC] Good candidate for intervention")
        elif unc_decomp['aleatoric_ratio'] > 0.7:
            print(f"   [HIGH ALEATORIC] Limited intervention benefit")

    print(f"\n{'─' * 80}")
    print("OVERALL UNCERTAINTY:")
    print(f"{'─' * 80}")
    print(f"Mean Epistemic: {explanation['mean_epistemic_uncertainty']:.4f}")
    print(f"Mean Aleatoric: {explanation['mean_aleatoric_uncertainty']:.4f}")
    print(f"Total:          {explanation['mean_epistemic_uncertainty'] + explanation['mean_aleatoric_uncertainty']:.4f}")
    print("=" * 80)


def main_example():
    # Complete example: Train and evaluate CLARITY Credal CBM
    print("\n" + "=" * 80)
    print(" " * 15 + "CLARITY CREDAL CBM - COMPLETE EXAMPLE")
    print("=" * 80)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load dataset (e.g., SST-2)
    print("Loading dataset...")
    dataset = load_dataset('glue', 'sst2')

    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create model
    model = ClarityCredalCBM(
        base_model_name=model_name,
        num_concepts=30,  # Number of interpretable concepts
        num_classes=2,    # Binary classification
        n_credal_points=5,
        concept_names=[f"sentiment_concept_{i}" for i in range(30)]
    ).to(device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = TextClassificationDataset(
        dataset['train'].select(range(5000)),  # Subset for demo
        tokenizer
    )
    val_dataset = TextClassificationDataset(
        dataset['validation'].select(range(500)),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Train model
    print("\nTraining model...")
    model, history = train_clarity_credal_cbm(
        model, train_loader, val_loader, tokenizer,
        epochs=5, device=device
    )

    # Visualize training
    print("\nGenerating training visualizations...")
    visualize_training_history(history)

    # Demonstrate explanations
    print("\n" + "=" * 80)
    print(" " * 15 + "DEMONSTRATION: EXPLANATION GENERATION")
    print("=" * 80)

    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money.",
        "It was okay, nothing special but not bad either."
    ]

    for text in test_texts:
        demonstrate_explanation(model, tokenizer, text, device=device)
        print("\n")

    print("Training and evaluation complete!")
    print("\nKey Advantages of CLARITY Credal CBM:")
    print("  1. Interpretable rationales (which text spans matter)")
    print("  2. Uncertainty-aware concepts (epistemic vs aleatoric)")
    print("  3. Active learning ready (target high epistemic uncertainty)")
    print("  4. Intervention-friendly (human feedback on uncertain concepts)")
    print("  5. Calibrated predictions (uncertainty reflects accuracy)")


if __name__ == "__main__":
    main_example()
