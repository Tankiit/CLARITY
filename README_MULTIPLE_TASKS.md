# Creating Multiple RQ2 Tasks - Quick Start Guide

## Overview

You have **Task 1 (Wearing_Lipstick)** working. Now you want to create **multiple tasks** for your Boolean Structure Discovery experiments.

This guide shows you the **simplest way** to copy, modify, and run multiple tasks.

---

## The Fast Method: Copy & Modify

**Strategy**: Copy the entire `extract_lipstick_task_data()` function, then change just 5 key things.

### What Works (Current Setup)
```python
# Task 1: Wearing_Lipstick
def extract_lipstick_task_data():
    target_idx = get_attr_idx('Wearing_Lipstick')
    concept_indices = [get_attr_idx(c) for c in ['Male', 'Task_Attributes', 'Here']]
    task_dir = os.path.join(OUTPUT_DIR, 'task1_lipstick')

    # ... extraction code ...

    metadata = {
        'task_id': 'task1_lipstick',
        'task_name': 'Wearing_Lipstick',
        'difficulty': 'medium',
        'base_concepts': ['Male', 'Young', 'Attractive', 'Smiling', 'Heavy_Makeup'],
        # ... rest of metadata
    }
```

### Step 1: Copy Function

**Copy the ENTIRE function** and rename it:

```python
# COPY AND PASTE
def extract_attractive_task_data():  # ← Rename function
    # ... copy ALL code from extract_lipstick_task_data()

def extract_makeup_task_data():      # ← Rename function
    # ... copy ALL code from extract_lipstick_task_data()
```

### Step 2: Change 5 Things (Only!)

For each new task, change **EXACTLY** these 5 things:

| What to Change | Task 1 (Template) | Task 2 (Attractive) | Task 3 (Makeup) |
|----------------|-------------------|---------------------|----------------|
| **Target concept** | `'Wearing_Lipstick'` | `'Attractive'` | `'Heavy_Makeup'` |
| **Base concepts** | `['Male', 'Young', 'Attractive', 'Smiling', 'Heavy_Makeup']` | `['High_Cheekbones', 'Oval_Face', 'Smiling', 'Young', 'Heavy_Makeup']` | `['Male', 'Young', 'Wearing_Lipstick', 'Rosy_Cheeks']` |
| **Task directory** | `'task1_lipstick'` | `'task2_attractive'` | `'task3_makeup'` |
| **Task ID** | `'task1_lipstick'` | `'task2_attractive'` | `'task3_makeup'` |
| **Expected pattern** | `¬Male ∧ (Attractive ∨ Heavy_Makeup)` | `High_Cheekbones ∧ (Oval_Face ∨ Smiling) ∧ Young` | `¬Male ∧ (Wearing_Lipstick ∨ Rosy_Cheeks)` |

### Step 3: Execute All Tasks

At the end of your file:

```python
# Execute all tasks
task1_created = extract_lipstick_task_data()
task2_created = extract_attractive_task_data()
task3_created = extract_makeup_task_data()
```

---

## Complete Code Template

Copy this template and fill in your task details:

```python
# ============================================================================
# RQ2: MULTIPLE TASKS TEMPLATE
# ============================================================================

import torch
import numpy as np
import os
import json
from collections import defaultdict

# Configuration
OUTPUT_DIR = './data/rq2_boolean_discovery_expanded'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CelebA attributes (40 total)
ATTRIBUTE_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

def get_attr_idx(attr_name):
    return ATTRIBUTE_NAMES.index(attr_name)

# ============================================================================
# TASK 1: WEARING LIPSTICK (ORIGINAL)
# ============================================================================
def extract_lipstick_task_data():
    """Create Task 1: Wearing_Lipstick"""
    target_idx = get_attr_idx('Wearing_Lipstick')
    concept_indices = [get_attr_idx(c) for c in ['Male', 'Young', 'Attractive', 'Smiling', 'Heavy_Makeup']]
    task_dir = os.path.join(OUTPUT_DIR, 'task1_lipstick')

    # ... [Copy extraction code from your working Task 1] ...
    return True

# ============================================================================
# TASK 2: ATTRACTIVE (COPY & MODIFY)
# ============================================================================
def extract_attractive_task_data():
    """Create Task 2: Attractive"""

    # ← CHANGE 1: Target concept
    target_idx = get_attr_idx('Attractive')

    # ← CHANGE 2: Base concepts
    concept_indices = [get_attr_idx(c) for c in [
        'High_Cheekbones', 'Oval_Face', 'Smiling', 'Young', 'Heavy_Makeup'
    ]]

    # ← CHANGE 3: Task directory
    task_dir = os.path.join(OUTPUT_DIR, 'task2_attractive')

    # ... [Copy extraction code from Task 1] ...

    # ← CHANGE 4: Metadata
    metadata = {
        'task_id': 'task2_attractive',
        'task_name': 'Attractive',
        'difficulty': 'hard',
        'base_concepts': ['High_Cheekbones', 'Oval_Face', 'Smling', 'Young', 'Heavy_Makeup'],
        'target_concept': 'Attractive',
        'expected_pattern': 'High_Cheekbones ∧ (Oval_Face ∨ Smiling) ∧ Young',
        'description': 'Facial features → attractiveness',
        'concept_indices': concept_indices,
        'target_idx': target_idx,
        'statistics': {
            'n_samples': len(y_train),
            'class_balance': {
                'negative': float(class_dist[0]),
                'positive': float(class_dist[1])
            }
        }
    }

    with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return True

# ============================================================================
# TASK 3: HEAVY_MAKEUP (COPY & MODIFY)
# ============================================================================
def extract_makeup_task_data():
    """Create Task 3: Heavy_Makeup"""

    # ← CHANGE 1: Target concept
    target_idx = get_attr_idx('Heavy_Makeup')

    # ← CHANGE 2: Base concepts
    concept_indices = [get_attr_idx(c) for c in [
        'Male', 'Young', 'Wearing_Lipstick', 'Rosy_Cheeks'
    ]]

    # ← CHANGE 3: Task directory
    task_dir = os.path.join(OUTPUT_DIR, 'task3_makeup')

    # ... [Copy extraction code from Task 1] ...

    # ← CHANGE 4: Metadata
    metadata = {
        'task_id': 'task3_makeup',
        'task_name': 'Heavy_Makeup',
        'difficulty': 'medium',
        'base_concepts': ['Male', 'Young', 'Wearing_Lipstick', 'Rosy_Cheeks'],
        'target_concept': 'Heavy_Makeup',
        'expected_pattern': '¬Male ∧ (Wearing_Lipstick ∨ Rosy_Cheeks)',
        'description': 'Individual makeup features → heavy makeup',
        'concept_indices': concept_indices,
        'target_idx': target_idx,
        'statistics': {
            'n_samples': len(y_train),
            'class_balance': {
                'negative': float(class_dist[0]),
                'positive': float(class_dist[1])
            }
        }
    }

    with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return True

# ============================================================================
# TASK 4+: ADD MORE TASKS HERE (COPY & MODIFY)
# ============================================================================
def extract_your_task_data():
    """Create Your Task X"""
    # ← CHANGE: Target concept
    target_idx = get_attr_idx('YourTargetConcept')

    # ← CHANGE: Base concepts
    concept_indices = [get_attr_idx(c) for c in ['Concept1', 'Concept2', 'Concept3']]

    # ← CHANGE: Task directory
    task_dir = os.path.join(OUTPUT_DIR, 'taskX_yourtask')

    # ... [Copy extraction code from Task 1] ...

    # ← CHANGE: Metadata
    metadata = {
        'task_id': 'taskX_yourtask',
        'task_name': 'YourTaskName',
        'difficulty': 'easy/medium/hard',
        'base_concepts': ['Concept1', 'Concept2', 'Concept3'],
        'target_concept': 'YourTargetConcept',
        'expected_pattern': 'Concept1 ∧ Concept2',
        'description': 'Your task description',
        'concept_indices': concept_indices,
        'target_idx': target_idx,
        'statistics': {
            'n_samples': len(y_train),
            'class_balance': {
                'negative': float(class_dist[0]),
                'positive': float(class_dist[1])
            }
        }
    }

    with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
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
    # task4_created = extract_your_task_data()
```

---

## Quick Reference: Available Concepts

Here are all the concepts you can use:

### Easy Tasks (Simple Boolean)
- `'Male'`, `'No_Beard'`, `'Young'`
- Pattern: Simple OR/AND logic

### Medium Tasks (Moderate Complexity)
- `'Heavy_Makeup'`, `'Wearing_Lipstick'`, `'Rosy_Cheeks'`
- Pattern: Mixed OR/AND logic

### Hard Tasks (Complex Interactions)
- `'Attractive'`, `'High_Cheekbones'`, `'Blond_Hair'`
- Pattern: Complex interactions, XOR patterns

### Hair Color (XOR patterns)
- `'Black_Hair'`, `'Brown_Hair'`, `'Blond_Hair'`, `'Gray_Hair'`
- Pattern: Mutually exclusive (only one can be true)

### Facial Features
- `'Oval_Face'`, `'Pointy_Nose'`, `'Rosy_Cheeks'`
- Pattern: Complex feature interactions

### Accessories
- `'Wearing_Earrings'`, `'Wearing_Hat'`, `'Wearing_Necktie'`
- Pattern: Correlation-based patterns

---

## Testing Your Tasks

After running the code, verify:

```bash
# Check task directories
ls -la data/rq2_boolean_discovery_expanded/

# Check metadata
cat data/rq2_boolean_discovery_expanded/task2_attractive/metadata.json

# Quick verification
python -c "
import numpy as np
X = np.load('data/rq2_boolean_discovery_expanded/task2_attractive/X_train.npy')
y = np.load('data/rq2_boolean_discovery_expanded/task2_attractive/y_train.npy')
print(f'Task 2: X={X.shape}, y={y.shape}, balance={y.mean():.1%}')
"
```

---

## Common Mistakes & Fixes

### Mistake 1: Attribute Name Typos
```python
# WRONG (typo)
target_idx = get_attr_idx('Atractive')

# CORRECT
target_idx = get_attr_idx('Attractive')
```

### Mistake 2: Inconsistent Metadata
```python
# WRONG (concept list doesn't match)
concept_indices = [get_attr_idx(c) for c in ['Male', 'Young']]
metadata = {
    'base_concepts': ['Male', 'Young', 'Attractive']  # 3 concepts!
}

# CORRECT (must match exactly)
concept_indices = [get_attr_idx(c) for c in ['Male', 'Young', 'Attractive']]
metadata = {
    'base_concepts': ['Male', 'Young', 'Attractive']  # 3 concepts
}
```

### Mistake 3: Directory Conflicts
```python
# WRONG (overwrites Task 1)
task_dir = os.path.join(OUTPUT_DIR, 'task1_lipstick')

# CORRECT (unique directory)
task_dir = os.path.join(OUTPUT_DIR, 'task2_attractive')
```

---

## Next Steps

### After Creating Tasks:

1. Run Method 1 (Decision Trees):
   ```bash
   python rq2_method1_decision_tree.py
   ```

2. Run Method 2 (Differentiable Logic):
   ```bash
   python rq2_method2_differentiable_logic.py
   ```

3. Compare Results:
   ```bash
   python rq2_compare_methods.py
   ```

4. Analyze Results:
   ```bash
   python analyze_rq2_results.py
   ```

### Expected Output Structure:

```
data/rq2_boolean_discovery_expanded/
├── task1_lipstick/
│   ├── X_train.npy, y_train.npy
│   ├── X_val.npy, y_val.npy
│   ├── X_test.npy, y_test.npy
│   ├── train.pt, val.pt, test.pt
│   └── metadata.json
│
├── task2_attractive/
│   ├── X_train.npy, y_train.npy
│   ├── X_val.npy, y_val.npy
│   ├── X_test.npy, y_test.npy
│   ├── train.pt, val.pt, test.pt
│   └── metadata.json
│
└── task3_makeup/
    ├── X_train.npy, y_train.npy
    ├── X_val.npy, y_val.npy
    ├── X_test.npy, y_test.npy
    ├── train.pt, val.pt, test.pt
    └── metadata.json
```

---

## Pro Tips

### Tip 1: Test One at a Time
Create Task 2, test it works, then create Task 3.

### Tip 2: Use the Task Table
Keep the task table handy while creating tasks to reference concept names.

### Tip 3: Check Class Balance
Good tasks have roughly 20-80% positive class balance.

### Tip 4: Start Simple
Begin with easy tasks, then move to harder ones.

### Tip 5: Document Patterns
Write clear expected patterns in metadata for easier analysis.

---

## Learning Exercise

Try creating Task 4 (Young) yourself!

**Hint:**
- Target: `'Young'`
- Concepts: `['Gray_Hair', 'Bald', 'Receding_Hairline', 'Bags_Under_Eyes']`
- Pattern: `'¬Gray_Hair ∧ ¬Bald ∧ ¬Receding_Hairline'`
- Difficulty: `'easy'`

---

## Troubleshooting

If you get errors:

1. **Attribute not found**: Check spelling in `ATTRIBUTE_NAMES`
2. **Directory exists**: Use unique task IDs
3. **Shape mismatches**: Ensure concept list length matches indices
4. **Permission errors**: Check directory permissions

**Debug Command:**
```bash
python -c "
import sys
sys.path.append('.')
from your_file import *

# Test one task
try:
    result = extract_attractive_task_data()
    print(f'Task 2 created: {result}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"
```

---

## Success Checklist

When done, you should have:

- [ ] 3+ tasks created with unique directories
- [ ] All metadata files with correct task information
- [ ] Balanced datasets (check with verification script)
- [ ] No errors during creation
- [ ] Ready for method comparison

Then you're ready for Boolean Structure Discovery!

---

**Need help?** The detailed guide above covers common scenarios. For specific issues, check the troubleshooting section or ask for help!