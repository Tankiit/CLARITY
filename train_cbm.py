"""
Training Script for Baseline CBM - Step 1
Simple training loop for CelebA using the new CBM framework
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

# Import our new CBM framework
from cbm_model import BaselineCBM, SimpleCBM

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TASK_ATTR_IDX = 36  # Wearing_Lipstick

print("="*60)
print("STEP 1: Training Baseline CBM with New Framework")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Task: Wearing_Lipstick (attribute {TASK_ATTR_IDX})")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Try different dataset loading approaches
print("\nLoading CelebA dataset...")
dataset_loaded = False
train_dataset = None
val_dataset = None

# Method 1: Try standard torchvision CelebA
try:
    print("Trying torchvision CelebA...")
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
        download=True
    )

    print(f"torchvision CelebA loaded successfully")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    dataset_loaded = True

except Exception as e:
    print(f"torchvision CelebA failed: {e}")

# Method 2: Try custom CelebA wrapper if torchvision fails
if not dataset_loaded:
    try:
        print("Trying custom CelebA wrapper...")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from disentangled_vae import CelebAWrapper

        train_dataset = CelebAWrapper(split='train', size=64)
        val_dataset = CelebAWrapper(split='valid', size=64)

        print(f"Custom CelebA wrapper loaded successfully")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Val: {len(val_dataset)} samples")
        dataset_loaded = True

    except Exception as e:
        print(f"Custom CelebA wrapper failed: {e}")

# Method 3: Try using existing data directory structure
if not dataset_loaded:
    try:
        print("Trying local CelebA data...")
        # Check if we have the data directory structure
        if os.path.exists('./data/celeba'):
            train_dataset = datasets.ImageFolder(
                root='./data/celeba/train',
                transform=transform
            )
            val_dataset = datasets.ImageFolder(
                root='./data/celeba/val',
                transform=transform
            )

            print(f"Local ImageFolder loaded successfully")
            print(f"Train: {len(train_dataset)} samples")
            print(f"Val: {len(val_dataset)} samples")
            dataset_loaded = True
        else:
            print("Local data directory not found")

    except Exception as e:
        print(f"Local data loading failed: {e}")

if not dataset_loaded:
    print("\nERROR: Could not load CelebA dataset with any method!")
    print("\nTo set up CelebA dataset:")
    print("1. Run: python setup_celeba.py")
    print("2. Or download manually from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("3. Or run: python test_celeba.py to create a mock dataset")
    sys.exit(1)

# Create dataloaders (using standard collate since torchvision CelebA works)
print("\nCreating dataloaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False
)

# Initialize model
print("\nInitializing model...")
model = BaselineCBM(num_concepts=40, num_classes=2).to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Training history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

print("\n" + "="*60)
print("Starting training...")
print("="*60)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # ==== TRAINING ====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, attributes in pbar:
        images = images.to(DEVICE)
        attributes = attributes.to(DEVICE)

        task_labels = attributes[:, TASK_ATTR_IDX].long()
        concept_labels = attributes.float()

        # Forward
        losses = model.compute_loss(images, task_labels, concept_labels)

        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Stats
        train_loss += losses['total_loss'].item()
        output = model(images)
        preds = output['task_predictions'].argmax(dim=1)
        train_correct += (preds == task_labels).sum().item()
        train_total += len(task_labels)

        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.3f}",
            'acc': f"{100*train_correct/train_total:.1f}%"
        })

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # ==== VALIDATION ====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, attributes in pbar:
            images = images.to(DEVICE)
            attributes = attributes.to(DEVICE)

            task_labels = attributes[:, TASK_ATTR_IDX].long()
            concept_labels = attributes.float()

            losses = model.compute_loss(images, task_labels, concept_labels)
            output = model(images)

            val_loss += losses['total_loss'].item()
            preds = output['task_predictions'].argmax(dim=1)
            val_correct += (preds == task_labels).sum().item()
            val_total += len(task_labels)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    # Update scheduler
    scheduler.step(avg_val_loss)

    # Save history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)

    # Print summary
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Save checkpoint
    if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, f'outputs/baseline_cbm_epoch{epoch+1}.pth')
        print(f"  Checkpoint saved")

# Training complete
print("\n" + "="*60)
print("Training complete!")
print("="*60)

# Plot curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('outputs/step1_training_curves.png', dpi=150)
print("Saved training curves to outputs/step1_training_curves.png")

# Final results
print(f"\nFinal Results:")
print(f"   Training Accuracy:   {history['train_acc'][-1]:.2f}%")
print(f"   Validation Accuracy: {history['val_acc'][-1]:.2f}%")

if history['val_acc'][-1] > 75:
    print("\nSUCCESS! Validation accuracy > 75%")
    print("   Step 1 complete! Ready for Step 2.")
else:
    print(f"\nValidation accuracy below 75%")
    print("   Consider training for more epochs or adjusting hyperparameters")

print("\n" + "="*60)

# Additional: Test concept predictions
print("\nTesting concept predictions...")
model.eval()
with torch.no_grad():
    # Get a small batch
    sample_images, sample_attrs = next(iter(val_loader))
    sample_images = sample_images[:4].to(DEVICE)
    sample_attrs = sample_attrs[:4].to(DEVICE)

    # Get concept predictions
    concepts = model.predict_concepts(sample_images)
    print(f"Concept predictions shape: {concepts.shape}")
    print(f"Sample concept values: {concepts[0][:5]}")

    # Test task from concepts
    task_from_concepts = model.predict_task_from_concepts(concepts)
    print(f"Task from concepts shape: {task_from_concepts['task_logits'].shape}")

print("\nAll tests completed successfully!")