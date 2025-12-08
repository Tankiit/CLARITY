import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_ATTR_IDX = 36  # Wearing_Lipstick
DATA_DIR = "/home/cril/Meher/Projet-cril/Projects/cbm-clarity/CLARITY/data/celeba"
# upload locally the dataset from kaggle

# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

## Create a class to load the dataset by Meher


class CelebADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load split info
        partition_file = os.path.join(root_dir, "list_eval_partition.txt")
        self.split_dict = {}
        with open(partition_file, "r") as f:
            for line in f:
                img, part = line.strip().split()
                self.split_dict[img] = int(part)

        # Load attribute labels
        attr_file = os.path.join(root_dir, "list_attr_celeba.txt")
        self.attr_names = []
        self.attr_labels = {}
        with open(attr_file, "r") as f:
            lines = f.readlines()
            self.attr_names = lines[1].strip().split()
            for line in lines[2:]:
                parts = line.strip().split()
                img = parts[0]
                labels = [int(x) for x in parts[1:]]
                # convert -1/+1 -> 0/1
                labels = [(x + 1) // 2 for x in labels]
                self.attr_labels[img] = labels

        # Select images for this split
        split_map = {"train": 0, "valid": 1, "test": 2}
        self.img_list = [
            img for img in self.split_dict if self.split_dict[img] == split_map[split]
        ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, "img_align_celeba", img_name)
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(self.attr_labels[img_name], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels


# ----------------------------
# Create datasets and loaders
# ----------------------------
train_dataset = CelebADataset(DATA_DIR, split="train", transform=transform)
val_dataset = CelebADataset(DATA_DIR, split="valid", transform=transform)


train_dataset = CelebADataset(DATA_DIR, split="train", transform=transform)
val_dataset = CelebADataset(DATA_DIR, split="valid", transform=transform)

## the rest of the original code
# Create dataloaders (using standard collate since torchvision CelebA works)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if DEVICE == "cuda" else False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE == "cuda" else False,
)


# Initialize model
model = BaselineCBM(num_concepts=40, num_classes=2).to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Training history
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# Training loop
for epoch in range(NUM_EPOCHS):
    # ==== TRAINING ====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, attributes in pbar:
        images = images.to(DEVICE)
        attributes = attributes.to(DEVICE)

        task_labels = attributes[:, TASK_ATTR_IDX].long()
        concept_labels = attributes.float()

        # Forward
        losses = model.compute_loss(images, task_labels, concept_labels)

        # Backward
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        # Stats
        train_loss += losses["total_loss"].item()
        output = model(images)
        preds = output["task_predictions"].argmax(dim=1)
        train_correct += (preds == task_labels).sum().item()
        train_total += len(task_labels)

        pbar.set_postfix(
            {
                "loss": f"{losses['total_loss'].item():.3f}",
                "acc": f"{100 * train_correct / train_total:.1f}%",
            }
        )

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # ==== VALIDATION ====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, attributes in pbar:
            images = images.to(DEVICE)
            attributes = attributes.to(DEVICE)

            task_labels = attributes[:, TASK_ATTR_IDX].long()
            concept_labels = attributes.float()

            losses = model.compute_loss(images, task_labels, concept_labels)
            output = model(images)

            val_loss += losses["total_loss"].item()
            preds = output["task_predictions"].argmax(dim=1)
            val_correct += (preds == task_labels).sum().item()
            val_total += len(task_labels)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    # Update scheduler
    scheduler.step(avg_val_loss)

    # Save history
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)

    # Save checkpoint
    if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            f"outputs/baseline_cbm_epoch{epoch + 1}.pth",
        )

# Plot curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"], label="Val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curve")
ax1.legend()
ax1.grid(True)

ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"], label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Curve")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("outputs/step1_training_curves.png", dpi=150)

# Additional: Test concept predictions
model.eval()
with torch.no_grad():
    sample_images, sample_attrs = next(iter(val_loader))
    sample_images = sample_images[:4].to(DEVICE)
    sample_attrs = sample_attrs[:4].to(DEVICE)

    concepts = model.predict_concepts(sample_images)
    task_from_concepts = model.predict_task_from_concepts(concepts)

# RQ2 Setup (EXPANDED): Boolean Structure Discovery
# Prepare CelebA data for concept prediction experiments

# EXPANDED TO 10 TASKS (from original 4) for:
# - Stronger statistical validation
# - Diverse Boolean patterns (AND, OR, XOR)
# - Task difficulty analysis
# - Better paper section

# Research Question: What's the best method to learn Boolean rules over concepts?
# - Method 1: Decision Trees (sklearn)
# - Method 2: Differentiable Logic (UniquePolynomialLayer)

import numpy as np
from collections import defaultdict
import json

# Configuration
DATA_DIR = "./data"
OUTPUT_DIR = "./data/rq2_boolean_discovery_expanded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CelebA attributes (40 total)
ATTRIBUTE_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

# TASK 1 IMPLEMENTATION: Wearing_Lipstick (Medium Difficulty)
# This is the main task we'll implement in this file


# Helper function to get attribute index
def get_attr_idx(attr_name):
    try:
        return ATTRIBUTE_NAMES.index(attr_name)
    except ValueError:
        raise ValueError(f"Attribute {attr_name} not found in CelebA attributes")


# Extract attributes for the main task
def extract_lipstick_task_data():
    # Get indices
    target_idx = get_attr_idx("Wearing_Lipstick")
    concept_indices = [
        get_attr_idx(c)
        for c in ["Male", "Young", "Attractive", "Smiling", "Heavy_Makeup"]
    ]

    # Create task directory
    task_dir = os.path.join(OUTPUT_DIR, "task1_lipstick")
    os.makedirs(task_dir, exist_ok=True)

    # Extract from datasets if available
    if train_dataset and val_dataset:
        # Process training data (sample first 10k for speed)
        n_samples = min(10000, len(train_dataset))
        X_train = []
        y_train = []

        for i in range(n_samples):
            _, attrs = train_dataset[i]
            # Extract base concepts and target
            X_train.append(attrs[concept_indices].numpy())
            y_train.append(attrs[target_idx].item())

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        # Process validation data
        n_val_samples = min(2000, len(val_dataset))
        X_val = []
        y_val = []

        for i in range(n_val_samples):
            _, attrs = val_dataset[i]
            X_val.append(attrs[concept_indices].numpy())
            y_val.append(attrs[target_idx].item())

        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int64)

        # Convert to 0/1 from -1/1 (CelebA uses -1/1)
        X_train = (X_train + 1) / 2
        y_train = (y_train + 1) // 2
        X_val = (X_val + 1) / 2
        y_val = (y_val + 1) // 2

        # Save as numpy (for sklearn)
        np.save(os.path.join(task_dir, "X_train.npy"), X_train)
        np.save(os.path.join(task_dir, "y_train.npy"), y_train)
        np.save(os.path.join(task_dir, "X_val.npy"), X_val)
        np.save(os.path.join(task_dir, "y_val.npy"), y_val)

        # Save as PyTorch (for differentiable logic)
        torch.save(
            {
                "X": torch.from_numpy(X_train),
                "y": torch.from_numpy(y_train),
                "concept_names": [
                    "Male",
                    "Young",
                    "Attractive",
                    "Smiling",
                    "Heavy_Makeup",
                ],
                "target_name": "Wearing_Lipstick",
            },
            os.path.join(task_dir, "train.pt"),
        )

        torch.save(
            {
                "X": torch.from_numpy(X_val),
                "y": torch.from_numpy(y_val),
                "concept_names": [
                    "Male",
                    "Young",
                    "Attractive",
                    "Smiling",
                    "Heavy_Makeup",
                ],
                "target_name": "Wearing_Lipstick",
            },
            os.path.join(task_dir, "val.pt"),
        )

        # Compute statistics
        class_dist = np.bincount(y_train) / len(y_train)

        # Save task metadata
        metadata = {
            "task_id": "task1_lipstick",
            "task_name": "Wearing_Lipstick",
            "difficulty": "medium",
            "base_concepts": ["Male", "Young", "Attractive", "Smiling", "Heavy_Makeup"],
            "target_concept": "Wearing_Lipstick",
            "expected_pattern": "¬Male ∧ (Attractive ∨ Heavy_Makeup)",
            "description": "Gender/age/makeup → lipstick",
            "concept_indices": concept_indices,
            "target_idx": target_idx,
            "statistics": {
                "n_samples": len(y_train),
                "class_balance": {
                    "negative": float(class_dist[0]),
                    "positive": float(class_dist[1]),
                },
            },
        }

        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return True
    else:
        return False


# ============================================================================
# TASK 2: ATTRACTIVE
# ============================================================================


def extract_attractive_task_data():
    """Create Task 2: Attractive"""

    # ← CHANGE 1: Target concept
    target_idx = get_attr_idx("Attractive")

    # ← CHANGE 2: Base concepts
    concept_indices = [
        get_attr_idx(c)
        for c in ["High_Cheekbones", "Oval_Face", "Smiling", "Young", "Heavy_Makeup"]
    ]

    # ← CHANGE 3: Task directory
    task_dir = os.path.join(OUTPUT_DIR, "task2_attractive")
    os.makedirs(task_dir, exist_ok=True)

    # Extract from datasets if available
    if train_dataset and val_dataset:
        # Process training data (sample first 10k for speed)
        n_samples = min(10000, len(train_dataset))
        X_train = []
        y_train = []

        for i in range(n_samples):
            _, attrs = train_dataset[i]
            # Extract base concepts and target
            X_train.append(attrs[concept_indices].numpy())
            y_train.append(attrs[target_idx].item())

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        # Process validation data
        n_val_samples = min(2000, len(val_dataset))
        X_val = []
        y_val = []

        for i in range(n_val_samples):
            _, attrs = val_dataset[i]
            X_val.append(attrs[concept_indices].numpy())
            y_val.append(attrs[target_idx].item())

        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int64)

        # Convert to 0/1 from -1/1 (CelebA uses -1/1)
        X_train = (X_train + 1) / 2
        y_train = (y_train + 1) // 2
        X_val = (X_val + 1) / 2
        y_val = (y_val + 1) // 2

        # Save as numpy (for sklearn)
        np.save(os.path.join(task_dir, "X_train.npy"), X_train)
        np.save(os.path.join(task_dir, "y_train.npy"), y_train)
        np.save(os.path.join(task_dir, "X_val.npy"), X_val)
        np.save(os.path.join(task_dir, "y_val.npy"), y_val)

        # Save as PyTorch (for differentiable logic)
        torch.save(
            {
                "X": torch.from_numpy(X_train),
                "y": torch.from_numpy(y_train),
                "concept_names": [
                    "High_Cheekbones",
                    "Oval_Face",
                    "Smiling",
                    "Young",
                    "Heavy_Makeup",
                ],
                "target_name": "Attractive",
            },
            os.path.join(task_dir, "train.pt"),
        )

        torch.save(
            {
                "X": torch.from_numpy(X_val),
                "y": torch.from_numpy(y_val),
                "concept_names": [
                    "High_Cheekbones",
                    "Oval_Face",
                    "Smiling",
                    "Young",
                    "Heavy_Makeup",
                ],
                "target_name": "Attractive",
            },
            os.path.join(task_dir, "val.pt"),
        )

        # Compute statistics
        class_dist = np.bincount(y_train) / len(y_train)

    # ← CHANGE 4: Metadata
    metadata = {
        "task_id": "task2_attractive",
        "task_name": "Attractive",
        "difficulty": "hard",
        "base_concepts": [
            "High_Cheekbones",
            "Oval_Face",
            "Smling",
            "Young",
            "Heavy_Makeup",
        ],
        "target_concept": "Attractive",
        "expected_pattern": "High_Cheekbones ∧ (Oval_Face ∨ Smiling) ∧ Young",
        "description": "Facial features → attractiveness",
        "concept_indices": concept_indices,
        "target_idx": target_idx,
        "statistics": {
            "n_samples": len(y_train),
            "class_balance": {
                "negative": float(class_dist[0]),
                "positive": float(class_dist[1]),
            },
        },
    }

    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return True


# ============================================================================
# TASK 3: HEAVY_MAKEUP
# ============================================================================


def extract_makeup_task_data():
    """Create Task 3: Heavy_Makeup"""

    # ← CHANGE 1: Target concept
    target_idx = get_attr_idx("Heavy_Makeup")

    # ← CHANGE 2: Base concepts
    concept_indices = [
        get_attr_idx(c) for c in ["Male", "Young", "Wearing_Lipstick", "Rosy_Cheeks"]
    ]

    # ← CHANGE 3: Task directory
    task_dir = os.path.join(OUTPUT_DIR, "task3_makeup")
    os.makedirs(task_dir, exist_ok=True)

    # Extract from datasets if available
    if train_dataset and val_dataset:
        # Process training data (sample first 10k for speed)
        n_samples = min(10000, len(train_dataset))
        X_train = []
        y_train = []

        for i in range(n_samples):
            _, attrs = train_dataset[i]
            # Extract base concepts and target
            X_train.append(attrs[concept_indices].numpy())
            y_train.append(attrs[target_idx].item())

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        # Process validation data
        n_val_samples = min(2000, len(val_dataset))
        X_val = []
        y_val = []

        for i in range(n_val_samples):
            _, attrs = val_dataset[i]
            X_val.append(attrs[concept_indices].numpy())
            y_val.append(attrs[target_idx].item())

        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int64)

        # Convert to 0/1 from -1/1 (CelebA uses -1/1)
        X_train = (X_train + 1) / 2
        y_train = (y_train + 1) // 2
        X_val = (X_val + 1) / 2
        y_val = (y_val + 1) // 2

        # Save as numpy (for sklearn)
        np.save(os.path.join(task_dir, "X_train.npy"), X_train)
        np.save(os.path.join(task_dir, "y_train.npy"), y_train)
        np.save(os.path.join(task_dir, "X_val.npy"), X_val)
        np.save(os.path.join(task_dir, "y_val.npy"), y_val)

        # Save as PyTorch (for differentiable logic)
        torch.save(
            {
                "X": torch.from_numpy(X_train),
                "y": torch.from_numpy(y_train),
                "concept_names": ["Male", "Young", "Wearing_Lipstick", "Rosy_Cheeks"],
                "target_name": "Heavy_Makeup",
            },
            os.path.join(task_dir, "train.pt"),
        )

        torch.save(
            {
                "X": torch.from_numpy(X_val),
                "y": torch.from_numpy(y_val),
                "concept_names": ["Male", "Young", "Wearing_Lipstick", "Rosy_Cheeks"],
                "target_name": "Heavy_Makeup",
            },
            os.path.join(task_dir, "val.pt"),
        )

        # Compute statistics
        class_dist = np.bincount(y_train) / len(y_train)

    # ← CHANGE 4: Metadata
    metadata = {
        "task_id": "task3_makeup",
        "task_name": "Heavy_Makeup",
        "difficulty": "medium",
        "base_concepts": ["Male", "Young", "Wearing_Lipstick", "Rosy_Cheeks"],
        "target_concept": "Heavy_Makeup",
        "expected_pattern": "¬Male ∧ (Wearing_Lipstick ∨ Rosy_Cheeks)",
        "description": "Individual makeup features → heavy makeup",
        "concept_indices": concept_indices,
        "target_idx": target_idx,
        "statistics": {
            "n_samples": len(y_train),
            "class_balance": {
                "negative": float(class_dist[0]),
                "positive": float(class_dist[1]),
            },
        },
    }

    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return True


# ============================================================================
# TASK 4: Young
# ============================================================================


def extract_young_data():
    """Create Task 3: Young"""

    target_idx = get_attr_idx("Young")

    concept_indices = [
        get_attr_idx(c)
        for c in ["Gray_Hair", "Bald", "Receding_Hairline", "Bags_Under_Eyes"]
    ]

    task_dir = os.path.join(OUTPUT_DIR, "task4_young")
    os.makedirs(task_dir, exist_ok=True)

    # Extract from datasets if available
    if train_dataset and val_dataset:
        # Process training data (sample first 10k for speed)
        n_samples = min(10000, len(train_dataset))
        X_train = []
        y_train = []

        for i in range(n_samples):
            _, attrs = train_dataset[i]
            # Extract base concepts and target
            X_train.append(attrs[concept_indices].numpy())
            y_train.append(attrs[target_idx].item())

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        # Process validation data
        n_val_samples = min(2000, len(val_dataset))
        X_val = []
        y_val = []

        for i in range(n_val_samples):
            _, attrs = val_dataset[i]
            X_val.append(attrs[concept_indices].numpy())
            y_val.append(attrs[target_idx].item())

        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int64)

        # Convert to 0/1 from -1/1 (CelebA uses -1/1)
        X_train = (X_train + 1) / 2
        y_train = (y_train + 1) // 2
        X_val = (X_val + 1) / 2
        y_val = (y_val + 1) // 2

        # Save as numpy (for sklearn)
        np.save(os.path.join(task_dir, "X_train.npy"), X_train)
        np.save(os.path.join(task_dir, "y_train.npy"), y_train)
        np.save(os.path.join(task_dir, "X_val.npy"), X_val)
        np.save(os.path.join(task_dir, "y_val.npy"), y_val)

        # Save as PyTorch (for differentiable logic)
        torch.save(
            {
                "X": torch.from_numpy(X_train),
                "y": torch.from_numpy(y_train),
                "concept_names": [
                    "Gray_Hair",
                    "Bald",
                    "Receding_Hairline",
                    "Bags_Under_Eyes",
                ],
                "target_name": "Young",
            },
            os.path.join(task_dir, "train.pt"),
        )

        torch.save(
            {
                "X": torch.from_numpy(X_val),
                "y": torch.from_numpy(y_val),
                "concept_names": [
                    "Gray_Hair",
                    "Bald",
                    "Receding_Hairline",
                    "Bags_Under_Eyes",
                ],
                "target_name": "Young",
            },
            os.path.join(task_dir, "val.pt"),
        )

        # Compute statistics
        class_dist = np.bincount(y_train) / len(y_train)

    # ← CHANGE: Metadata
    metadata = {
        "task_id": "task4_young",
        "task_name": "Young",
        "difficulty": "easy",
        "base_concepts": ["Gray_Hair", "Bald", "Receding_Hairline", "Bags_Under_Eyes"],
        "target_concept": "Young",
        "expected_pattern": "¬Gray_Hair ∧ ¬Bald ∧ ¬Receding_Hairline",
        "description": "Facial features → Young",
        "concept_indices": concept_indices,
        "target_idx": target_idx,
        "statistics": {
            "n_samples": len(y_train),
            "class_balance": {
                "negative": float(class_dist[0]),
                "positive": float(class_dist[1]),
            },
        },
    }

    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return True


# Execute the task creation
task_created = extract_lipstick_task_data()

task2_created = extract_attractive_task_data()

task3_created = extract_makeup_task_data()

task4_created = extract_young_data()
