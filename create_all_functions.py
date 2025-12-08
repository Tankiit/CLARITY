# ============================================================================
# RQ2: MULTIPLE TASKS TEMPLATE
# ============================================================================

import torch
import numpy as np
import os
import json
from collections import defaultdict

# Configuration
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


def get_attr_idx(attr_name):
    return ATTRIBUTE_NAMES.index(attr_name)


# ============================================================================
# TASK 1: WEARING LIPSTICK (ORIGINAL)
# ============================================================================
def extract_lipstick_task_data():
    """Create Task 1: Wearing_Lipstick"""
    target_idx = get_attr_idx("Wearing_Lipstick")
    concept_indices = [
        get_attr_idx(c)
        for c in ["Male", "Young", "Attractive", "Smiling", "Heavy_Makeup"]
    ]
    task_dir = os.path.join(OUTPUT_DIR, "task1_lipstick")

    # ... [Copy extraction code from your working Task 1] ...
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
# TASK 2: ATTRACTIVE (COPY & MODIFY)
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

    # ... [Copy extraction code from Task 1] ...

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
# TASK 3: HEAVY_MAKEUP (COPY & MODIFY)
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


## Exercice of adding Young Task :


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


# ============================================================================
# EXECUTE ALL TASKS
# ============================================================================
if __name__ == "__main__":
    # Task 1 (existing)
    task1_created = extract_lipstick_task_data()

    # Task 2 (new)
    task2_created = extract_attractive_task_data()

    # Task 3 (new)
    task3_created = extract_makeup_task_data()

    # Task 4+ (add more if needed)

    task4_created = extract_young_data()
