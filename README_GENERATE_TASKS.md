# CLARITY – CelebA Multi-Task Boolean Dataset Generator

## Overview
This project provides a complete pipeline to generate **multiple concept-based binary tasks from the CelebA dataset**.
The generated datasets are designed for **Concept Bottleneck Models (CBMs)**, symbolic reasoning, and classical machine learning.

---

## Repository Structure

```
.
├── create_all_functions.py
├── cbm_model.py
├── training_cbm.py
├── data/
│   └── rq2_boolean_discovery_expanded/
│       ├── task1_lipstick/
│       ├── task2_attractive/
│       ├── task3_makeup/
│       └── task4_young/
└── README.md
```

---

## Dependencies

Required libraries:

- torch
- torchvision
- numpy
- tqdm
- matplotlib

Install all dependencies with:

```bash
pip install torch torchvision numpy tqdm matplotlib
```

---

## CelebA Attributes

CelebA contains 40 binary attributes:

- Male
- Young
- Attractive
- Smiling
- Heavy_Makeup
- Wearing_Lipstick
- Gray_Hair
- Bald
- Receding_Hairline
- Bags_Under_Eyes
- ... (40 total)

All attributes are converted from {-1, +1} to {0, 1}.

---

## Implemented Tasks

### Task 1 – Wearing Lipstick
**Target:** Wearing_Lipstick  
**Base Concepts:** Male, Young, Attractive, Smiling, Heavy_Makeup  
**Logic:** ¬Male ∧ (Attractive ∨ Heavy_Makeup)  
**Difficulty:** Medium  

---

### Task 2 – Attractive
**Target:** Attractive  
**Base Concepts:** High_Cheekbones, Oval_Face, Smiling, Young, Heavy_Makeup  
**Logic:** High_Cheekbones ∧ (Oval_Face ∨ Smiling) ∧ Young  
**Difficulty:** Hard  

---

### Task 3 – Heavy Makeup
**Target:** Heavy_Makeup  
**Base Concepts:** Male, Young, Wearing_Lipstick, Rosy_Cheeks  
**Logic:** ¬Male ∧ (Wearing_Lipstick ∨ Rosy_Cheeks)  
**Difficulty:** Medium  

---

### Task 4 – Young
**Target:** Young  
**Base Concepts:** Gray_Hair, Bald, Receding_Hairline, Bags_Under_Eyes  
**Logic:** ¬Gray_Hair ∧ ¬Bald ∧ ¬Receding_Hairline  
**Difficulty:** Easy  

---

## Generated Files (Per Task)

Each task folder contains:

- X_train.npy
- y_train.npy
- X_val.npy
- y_val.npy
- train.pt
- val.pt
- metadata.json

---

## Metadata Example

```json
{
  "task_id": "task1_lipstick",
  "task_name": "Wearing_Lipstick",
  "difficulty": "medium",
  "base_concepts": ["Male", "Young", "Attractive", "Smiling", "Heavy_Makeup"],
  "target_concept": "Wearing_Lipstick",
  "expected_pattern": "¬Male ∧ (Attractive ∨ Heavy_Makeup)"
}
```

---

## How to Run

- Ensure CelebA is available and loaded
- Copy all functions to train_cbm.py
- then run:

```bash
python train_cbm.py
```

Generated tasks will be stored in:

```
data/rq2_boolean_discovery_expanded/
```

---

Outputs:
- Model checkpoints
- Training curves
- Validation accuracy

---

## Extending with New Tasks

To add a new task:
1. Choose a target attribute
2. Define base concepts
3. Specify expected logic
4. Save NumPy, PyTorch, and metadata files

---
