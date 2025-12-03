"""
Step 1 (Improved): Train Baseline CBM with Better Metrics
Now tracks: Accuracy, Balanced Accuracy, F1 Score, Per-Class Accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import our new CBM framework
from cbm_model import BaselineCBM

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TASK_ATTR_IDX = 36  # Wearing_Lipstick
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Data loading
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.CelebA(
    root='./data',
    split='train',
    target_type='attr',
    transform=transform,
    download=True
)

val_dataset = datasets.CelebA(
    root='./data',
    split='valid',
    target_type='attr',
    transform=transform,
    download=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Check class distribution
all_labels = []
for i in range(min(5000, len(val_dataset))):  # Sample first 5000
    _, attrs = val_dataset[i]
    all_labels.append(attrs[TASK_ATTR_IDX].item())

positive_ratio = sum(all_labels) / len(all_labels)

# Initialize model
model = BaselineCBM(num_concepts=40, num_classes=2).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# Training metrics storage
history = {
    'train_loss': [],
    'val_accuracy': [],
    'val_balanced_accuracy': [],
    'val_f1': [],
    'val_class0_acc': [],
    'val_class1_acc': []
}

def compute_metrics(all_preds, all_labels):
    """Compute comprehensive metrics"""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    class0_mask = (all_labels == 0)
    class1_mask = (all_labels == 1)

    class0_acc = (all_preds[class0_mask] == 0).mean() * 100 if class0_mask.sum() > 0 else 0
    class1_acc = (all_preds[class1_mask] == 1).mean() * 100 if class1_mask.sum() > 0 else 0

    # Balanced accuracy
    balanced_acc = (class0_acc + class1_acc) / 2

    # Precision, Recall, F1 for class 1
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'class0_acc': class0_acc,
        'class1_acc': class1_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training loop
best_balanced_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, attributes in progress_bar:
        images = images.to(DEVICE)
        concept_labels = attributes.float().to(DEVICE)
        task_labels = attributes[:, TASK_ATTR_IDX].to(DEVICE)

        # Forward pass
        output = model(images)

        # Combined loss: task + concepts
        task_loss = criterion(output['task_logits'], task_labels)
        concept_loss = nn.BCELoss()(output['concepts'], concept_labels)
        loss = task_loss + 0.5 * concept_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = train_loss / train_batches
    history['train_loss'].append(avg_train_loss)

    # Validation with detailed metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, attributes in val_loader:
            images = images.to(DEVICE)
            task_labels = attributes[:, TASK_ATTR_IDX]

            output = model(images)
            predictions = output['task_predictions'].argmax(dim=1).cpu()

            all_preds.extend(predictions.numpy())
            all_labels.extend(task_labels.numpy())

    # Compute all metrics
    metrics = compute_metrics(all_preds, all_labels)

    # Store metrics
    history['val_accuracy'].append(metrics['accuracy'])
    history['val_balanced_accuracy'].append(metrics['balanced_accuracy'])
    history['val_f1'].append(metrics['f1'])
    history['val_class0_acc'].append(metrics['class0_acc'])
    history['val_class1_acc'].append(metrics['class1_acc'])

    # Learning rate scheduling based on balanced accuracy
    scheduler.step(metrics['balanced_accuracy'])

    # Save best model (based on balanced accuracy)
    if metrics['balanced_accuracy'] > best_balanced_acc:
        best_balanced_acc = metrics['balanced_accuracy']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'balanced_accuracy': metrics['balanced_accuracy'],
        }, 'outputs/best_baseline_cbm_improved.pth')

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, f'outputs/baseline_cbm_improved_epoch{epoch+1}.pth')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy vs Balanced Accuracy
axes[0, 1].plot(history['val_accuracy'], label='Accuracy', marker='o')
axes[0, 1].plot(history['val_balanced_accuracy'], label='Balanced Accuracy', marker='s')
axes[0, 1].axhline(y=max(positive_ratio, 1-positive_ratio)*100,
                   color='r', linestyle='--', label='Majority Baseline')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Accuracy Metrics (Use Balanced!)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Per-class accuracy
axes[1, 0].plot(history['val_class0_acc'], label='Class 0 (No lipstick)', marker='o')
axes[1, 0].plot(history['val_class1_acc'], label='Class 1 (Has lipstick)', marker='s')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Per-Class Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# F1 Score
axes[1, 1].plot(history['val_f1'], label='F1 Score', marker='o', color='green')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score (Precision-Recall Balance)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('outputs/step1_training_curves_improved.png', dpi=300, bbox_inches='tight')

# Final evaluation
final_metrics = compute_metrics(all_preds, all_labels)

# Success criteria
majority_baseline = max(positive_ratio, 1-positive_ratio) * 100
improvement = final_metrics['balanced_accuracy'] - majority_baseline

success_criteria = [
    (final_metrics['balanced_accuracy'] > 70, "Balanced Accuracy > 70%"),
    (improvement > 15, "Improvement > 15% over baseline"),
    (final_metrics['f1'] > 0.65, "F1 Score > 0.65"),
    (abs(final_metrics['class0_acc'] - final_metrics['class1_acc']) < 15,
     "Class accuracies within 15% of each other"),
]

all_passed = True
for passed, criterion in success_criteria:
    if not passed:
        all_passed = False

# Additional: Test concept predictions
model.eval()
with torch.no_grad():
    sample_images, sample_attrs = next(iter(val_loader))
    sample_images = sample_images[:4].to(DEVICE)
    sample_attrs = sample_attrs[:4].to(DEVICE)

    concepts = model.predict_concepts(sample_images)
    task_from_concepts = model.predict_task_from_concepts(concepts)