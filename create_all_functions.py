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
# EXECUTE ALL TASKS
# ============================================================================
if __name__ == "__main__":

