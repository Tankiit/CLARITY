# Generating Explanations from Rationale-Concept Model Checkpoints

This guide explains how to use a trained Rationale-Concept Bottleneck Model checkpoint to generate various types of explanations for text classification.

## Overview

The Rationale-Concept Bottleneck Model provides several types of explanations:

1. **Basic Explanations**: Shows the rationale (important text spans) and concepts used for classification
2. **Concept Interventions**: Modifies concept values to see how they affect predictions
3. **Counterfactual Explanations**: Finds concepts that would change the prediction to a target class
4. **Contrastive Explanations**: Compares explanations between two different texts
5. **Concept Importance Analysis**: Analyzes which concepts are most important across examples

## Prerequisites

Before using these scripts, you need:

1. A trained model checkpoint (`.pt` file)
2. The corresponding model configuration (`.json` file)
3. The `optimized_rationale_concept_model.py` file in your working directory

## Scripts

This directory contains several scripts for generating explanations:

- `generate_explanations.py`: Main script for generating various explanations
- `example_generate_explanations.py`: Simple example of loading a checkpoint and generating explanations
- `explore_concept_space.py`: Advanced script for exploring the concept space

## Basic Usage

### 1. Generate Basic Explanations

To generate basic explanations for a text:

```bash
python generate_explanations.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --text "Your text to explain" \
  --mode basic
```

This will show:
- The predicted class and confidence
- The rationale (important spans of text)
- The top concepts used for classification

### 2. Generate Concept Interventions

To see how modifying concepts affects predictions:

```bash
python generate_explanations.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --text "Your text to explain" \
  --mode intervention
```

This will:
- Identify top concepts for the text
- Show what happens when each concept is turned on (1.0) or off (0.0)
- Highlight when the prediction changes

### 3. Generate Counterfactual Explanations

To find concepts that would change the prediction to a target class:

```bash
python generate_explanations.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --text "Your text to explain" \
  --mode counterfactual \
  --target_class 2
```

This will:
- Try to find concepts that, when modified, change the prediction to the target class
- Show which concepts need to be changed and how

### 4. Save Explanations to File

To save explanations to a JSON file:

```bash
python generate_explanations.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --text "Your text to explain" \
  --mode all \
  --output_file explanations.json
```

## Advanced Usage

### 1. Analyze Concept Importance

To analyze which concepts are most important across multiple examples:

```bash
python explore_concept_space.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --mode importance \
  --output_dir concept_analysis
```

This will:
- Test each concept's effect on multiple examples
- Rank concepts by their importance (how often they change predictions)
- Generate visualizations of concept importance

### 2. Generate Contrastive Explanations

To compare explanations between two different texts:

```bash
python explore_concept_space.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --mode contrastive \
  --text "First text to compare" \
  --text2 "Second text to compare"
```

This will:
- Show rationales for both texts
- Identify concepts that differ significantly between the texts
- Highlight key differences in the explanations

### 3. Find Minimal Concept Sets

To find the smallest set of concepts that change a prediction:

```bash
python explore_concept_space.py \
  --checkpoint_path models/YOUR_CHECKPOINT_DIR/checkpoints/best_model.pt \
  --config_path models/YOUR_CHECKPOINT_DIR/config.json \
  --mode minimal_set \
  --text "Your text to explain" \
  --target_class 2
```

This will:
- Try to find the smallest set of concepts (up to 3) that change the prediction
- Show which combinations of concepts are most effective

## Example Workflow

Here's a typical workflow for exploring explanations:

1. Train your model and save the checkpoint
2. Generate basic explanations for some test examples
3. Identify interesting cases where the model makes correct/incorrect predictions
4. For those cases, explore concept interventions to understand what drives the predictions
5. Generate counterfactual explanations to see what would change the predictions
6. Analyze concept importance across your dataset to identify key concepts

## Tips for Better Explanations

1. **Dataset-specific class names**: Use the `--dataset` parameter to get human-readable class names
2. **Concept naming**: The model uses generic concept names (concept_0, concept_1, etc.). You can analyze patterns in the data to give meaningful names to these concepts.
3. **Rationale quality**: The quality of rationales depends on the model's training. If rationales seem random, try adjusting the rationale sparsity and continuity weights during training.
4. **Concept interpretability**: If concepts don't seem interpretable, try training with higher concept sparsity and diversity weights.

## Troubleshooting

- **CUDA out of memory**: Try reducing batch sizes or using CPU inference
- **No concepts change predictions**: Try increasing the number of concepts during training
- **Poor explanations**: The model might need more training or different hyperparameters

## Further Customization

You can modify these scripts to:
- Add support for your own datasets
- Implement new types of explanations
- Visualize explanations in different ways
- Integrate with other analysis tools

For more details, see the documentation in each script and the main `optimized_rationale_concept_model.py` file.
