# CLARITY Open Source Knowledge Base

> **CLARITY** - Text Classification with Concept Learning and Rationale Extraction
>
> Repository: `github.com/Tankiit/CLARITY`
> License: MIT
> Author: mukher74 (mukher74@imec.be)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Open Source Dependencies](#3-open-source-dependencies)
4. [Core Modules Reference](#4-core-modules-reference)
5. [Supported Datasets](#5-supported-datasets)
6. [Configuration Reference](#6-configuration-reference)
7. [Scripts & Tools Inventory](#7-scripts--tools-inventory)
8. [Usage Patterns](#8-usage-patterns)
9. [Explanation Types](#9-explanation-types)
10. [Extending CLARITY](#10-extending-clarity)

---

## 1. Project Overview

CLARITY is an open source interpretable machine learning system for text classification. It builds transparent, explainable predictions by combining three key ideas:

- **Rationale Extraction** - Automatically identifies the most important text spans that drive a prediction.
- **Concept Learning** - Maps rationales to high-level, human-interpretable concept representations.
- **Concept Bottleneck** - Forces all predictions through an interpretable concept layer, enabling inspection and intervention.

The system is designed for researchers and practitioners who need to understand *why* a model makes a particular classification decision, not just *what* the decision is.

### Key Capabilities

| Capability | Description |
|---|---|
| Interpretable Classification | Three-stage pipeline producing human-readable explanations |
| Concept Interventions | Modify concept values at inference time to observe effects |
| Counterfactual Explanations | Find which concepts would change a prediction |
| Contrastive Explanations | Compare explanations between two different inputs |
| Causal Discovery | PC-algorithm-based analysis of concept relationships |
| Ablation Studies | Systematic component contribution analysis |
| Multi-Dataset Support | CEBaB, AG News, SST-2, Yelp Polarity, DBpedia |

---

## 2. Architecture

### 2.1 Three-Stage Pipeline

```
Input Text
    |
    v
[Transformer Encoder] ---- (DistilBERT / RoBERTa)
    |
    v
[Rationale Extractor] ---- Attention-based span selection
    |                       (RationaleExtractor)
    v
[Concept Mapper] --------- Maps rationale embeddings to concept probabilities
    |                       (ConceptMapper)
    v
[Classifier] ------------- Linear layer over concept probabilities
    |                       (optional skip connection from encoder CLS token)
    v
Prediction + Explanation
```

### 2.2 Core Components

#### RationaleExtractor
- Uses a learned attention mechanism (query/key projections) to score token importance.
- Extracts contiguous text spans via vectorized span scoring.
- Configurable span sizes (`min_span_size=3`, `max_span_size=20`).
- Produces: `token_scores`, `rationale_mask`, `token_probs`, `pooled_attended`.

#### ConceptMapper
- Encodes rationale-weighted embeddings into concept probabilities via a feed-forward network.
- Optional concept interaction matrix (symmetric, sigmoid-gated) for modeling inter-concept dependencies.
- Produces: `concept_scores`, `concept_probs`, `interaction_matrix`.

#### RationaleConceptBottleneckModel
- Full pipeline model integrating encoder, rationale extractor, concept mapper, and classifier.
- Supports skip connections (CLS embedding concatenated with concepts before classification).
- Multi-objective loss: classification + concept sparsity + concept diversity + rationale sparsity + rationale continuity.
- Built-in `explain_prediction()` method for generating human-readable explanations.
- Optional LoRA (Low-Rank Adaptation) via Hugging Face PEFT for parameter-efficient fine-tuning.

### 2.3 Loss Function

The training loss is a weighted sum of five components:

| Loss Component | Weight Config Key | Purpose |
|---|---|---|
| Cross-entropy classification | `classification_weight` (1.0) | Primary task objective |
| Concept sparsity | `concept_sparsity_weight` (0.03) | Encourage fewer active concepts |
| Concept diversity | `concept_diversity_weight` (0.01) | Encourage diverse concept usage |
| Rationale sparsity | `rationale_sparsity_weight` (0.03) | Encourage concise rationales |
| Rationale continuity | `rationale_continuity_weight` (0.1) | Encourage contiguous spans |

---

## 3. Open Source Dependencies

### 3.1 Core ML Stack

| Package | License | Role in CLARITY | Files Using |
|---|---|---|---|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Deep learning framework, model training, GPU acceleration | 34 |
| [Hugging Face Transformers](https://huggingface.co/transformers/) | Apache-2.0 | Pre-trained transformer models (DistilBERT, RoBERTa), tokenizers, optimizers | 32 |
| [Hugging Face Datasets](https://huggingface.co/docs/datasets/) | Apache-2.0 | Dataset loading and preprocessing (AG News, SST-2, CEBaB, etc.) | 14 |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause | Evaluation metrics (accuracy, F1, precision, recall), PCA, t-SNE, cosine similarity | 6 |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Numerical computing, array operations | 34 |

### 3.2 Visualization & Data

| Package | License | Role in CLARITY | Files Using |
|---|---|---|---|
| [Matplotlib](https://matplotlib.org/) | PSF-based | Training plots, concept visualizations, rationale heatmaps | 26 |
| [Seaborn](https://seaborn.pydata.org/) | BSD-3-Clause | Statistical data visualizations, concept correlation plots | 12 |
| [Pandas](https://pandas.pydata.org/) | BSD-3-Clause | Data manipulation, result tables, analysis DataFrames | 18 |

### 3.3 Utilities

| Package | License | Role in CLARITY | Files Using |
|---|---|---|---|
| [tqdm](https://tqdm.github.io/) | MPL-2.0 / MIT | Progress bars for training loops and data processing | 12 |
| [colorama](https://github.com/tartley/colorama) | BSD-3-Clause | Colored terminal output for explanations and analysis | 6 |
| [Pillow (PIL)](https://python-pillow.org/) | HPND | Image handling for visualization exports | 3 |
| [PyPDF2](https://pypdf2.readthedocs.io/) | BSD-3-Clause | Combining PDF visualizations | 1 |

### 3.4 Optional Dependencies

| Package | License | Role in CLARITY |
|---|---|---|
| [causal-learn](https://causal-learn.readthedocs.io/) | MIT | PC algorithm for causal discovery between learned concepts |
| [PEFT (LoRA)](https://huggingface.co/docs/peft/) | Apache-2.0 | Parameter-efficient fine-tuning via Low-Rank Adaptation |

### 3.5 Pre-trained Models Used

| Model | Source | License | Usage |
|---|---|---|---|
| `distilbert-base-uncased` | Hugging Face Hub | Apache-2.0 | Default base encoder |
| `roberta-base` | Hugging Face Hub | MIT | Alternative base encoder |

### 3.6 Installation

```bash
# Required
pip install torch transformers datasets matplotlib seaborn pandas numpy tqdm

# Optional
pip install causal-learn    # For causal discovery analysis
pip install peft            # For LoRA fine-tuning
pip install colorama        # For colored terminal output
pip install Pillow PyPDF2   # For image/PDF handling
```

---

## 4. Core Modules Reference

### 4.1 `optimized_rationale_concept_model.py` (Core)

The foundational module. Contains all model classes and training utilities.

| Class / Function | Type | Description |
|---|---|---|
| `ModelConfig` | Class | Configuration dataclass with serialization (`save`/`load`) |
| `RationaleExtractor` | `nn.Module` | Attention-based rationale span extractor |
| `ConceptMapper` | `nn.Module` | Rationale-to-concept probability mapper |
| `RationaleConceptBottleneckModel` | `nn.Module` | Full pipeline model with explain/intervene capabilities |
| `MetricsTracker` | Class | Training metrics tracking and history |
| `load_and_process_dataset()` | Function | Dataset loading and tokenization pipeline |
| `train_model()` | Function | Full training loop with validation |
| `evaluate_model()` | Function | Evaluation with metrics computation |

### 4.2 `main.py` (AG News Training)

Entry point for AG News experiments. Supports `--small`, `--fast`, `--visualize`, `--causal_discovery`, and `--inference_only` modes.

### 4.3 `train_on_cebab.py` (CEBaB Training)

Specialized training for the CEBaB restaurant review dataset. Extends `ModelConfig` with gradient accumulation support.

### 4.4 `cebab_mapper.py` (Concept-Attribute Mapping)

Maps learned concepts to CEBaB's ground-truth aspects (food, service, ambiance, noise) using cosine similarity and keyword analysis.

---

## 5. Supported Datasets

| Dataset | Task | Classes | Source | Key Aspects |
|---|---|---|---|---|
| **CEBaB** | Sentiment | 2 (pos/neg) | Hugging Face | Restaurant reviews with aspect annotations (food, service, ambiance, noise) |
| **AG News** | Topic | 4 (World, Sports, Business, Sci/Tech) | Hugging Face | News article categorization |
| **SST-2** | Sentiment | 2 (pos/neg) | Hugging Face (GLUE) | Movie review sentiment |
| **Yelp Polarity** | Sentiment | 2 (pos/neg) | Hugging Face | Business review sentiment |
| **DBpedia** | Ontology | 14 | Hugging Face | Wikipedia article classification |

### Test Examples (CEBaB)

Eight curated restaurant review examples in `test_examples/`:

| File | Aspect | Sentiment |
|---|---|---|
| `food_positive.txt` | Food quality | Positive |
| `food_negative.txt` | Food quality | Negative |
| `service_positive.txt` | Service quality | Positive |
| `service_negative.txt` | Service quality | Negative |
| `ambiance_positive.txt` | Ambiance | Positive |
| `ambiance_negative.txt` | Ambiance | Negative |
| `noise_positive.txt` | Noise level | Positive |
| `noise_negative.txt` | Noise level | Negative |

---

## 6. Configuration Reference

All configuration is managed through the `ModelConfig` class.

### 6.1 Model Architecture Parameters

| Parameter | Default | Description |
|---|---|---|
| `base_model_name` | `"distilbert-base-uncased"` | Hugging Face model identifier |
| `num_labels` | `4` | Number of output classes |
| `num_concepts` | `50` | Number of learned concepts |
| `hidden_size` | `768` | Hidden dimension (auto-set from base model) |
| `dropout_rate` | `0.1` | Dropout probability |
| `enable_concept_interactions` | `False` | Enable inter-concept interaction matrix |
| `use_skip_connection` | `True` | Add CLS embedding to classifier input |
| `use_lora` | `False` | Use LoRA for parameter-efficient training |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor |

### 6.2 Rationale Extraction Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_span_size` | `3` | Minimum rationale span length (tokens) |
| `max_span_size` | `20` | Maximum rationale span length (tokens) |
| `length_bonus_factor` | `0.01` | Bonus for longer spans in scoring |
| `target_rationale_percentage` | `0.2` | Target fraction of input to select as rationale |

### 6.3 Loss Weights

| Parameter | Default | Description |
|---|---|---|
| `classification_weight` | `1.0` | Weight for cross-entropy loss |
| `concept_sparsity_weight` | `0.03` | Weight for concept sparsity regularization |
| `concept_diversity_weight` | `0.01` | Weight for concept diversity regularization |
| `rationale_sparsity_weight` | `0.03` | Weight for rationale sparsity regularization |
| `rationale_continuity_weight` | `0.1` | Weight for rationale continuity regularization |

### 6.4 Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `32` | Training batch size |
| `max_seq_length` | `128` | Maximum input sequence length |
| `learning_rate` | `2e-5` | Learning rate for model heads |
| `base_model_lr` | `1e-5` | Learning rate for base encoder |
| `weight_decay` | `0.01` | AdamW weight decay |
| `num_epochs` | `5` | Number of training epochs |
| `warmup_ratio` | `0.1` | Fraction of training for LR warmup |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `seed` | `42` | Random seed |
| `output_dir` | `"models"` | Output directory for checkpoints |

### 6.5 Preset Configurations

| Preset | Flags | Key Changes |
|---|---|---|
| **Default** | (none) | 50 concepts, batch 32, 50 epochs |
| **Small** | `--small` | 20 concepts, batch 16, seq_len 64, 2 epochs |
| **Fast** | `--fast` | LoRA enabled (r=8), no concept interactions |
| **Small+Fast** | `--small --fast` | Combined optimizations for quick experiments |

---

## 7. Scripts & Tools Inventory

### 7.1 Training Scripts (3)

| Script | Dataset | Description |
|---|---|---|
| `main.py` | AG News | Primary training pipeline with visualization and causal discovery |
| `train_on_cebab.py` | CEBaB | CEBaB-specific training with gradient accumulation |
| `yelp_dbpedia_experiment.py` | Yelp/DBpedia | Multi-dataset experiment runner |

### 7.2 Visualization Scripts (7)

| Script | Description |
|---|---|
| `visualize_cebab_model.py` | Comprehensive CEBaB model visualization (HTML output) |
| `visualize_sst2_model.py` | SST-2 model visualization |
| `visualize_examples.py` | General example visualization |
| `create_latex_visualization.py` | LaTeX-compatible visualization exports |
| `create_visualization_summary.py` | Summary visualization creation |
| `combine_visualizations.py` | Combine multiple visualization outputs |
| `combine_pdfs.py` | Merge PDF visualization files |

### 7.3 Explanation Generation Scripts (5)

| Script | Description |
|---|---|
| `generate_explanations.py` | Main explanation generator (basic, intervention, counterfactual) |
| `generate_cebab_explanations.py` | CEBaB-specific explanation generation |
| `generate_explanations_from_checkpoint.py` | Load checkpoint and generate explanations |
| `simple_explanations.py` | Simplified explanation interface |
| `explore_concept_space.py` | Advanced concept space exploration and contrastive explanations |

### 7.4 Analysis Scripts (10)

| Script | Description |
|---|---|
| `ablation_analysis.py` | Systematic ablation study implementation |
| `ablation_study.py` | Additional ablation analysis tools |
| `generate_ablation_report.py` | Generate formatted ablation reports |
| `analyze_concepts.py` | Concept analysis and interpretation |
| `analyze_concepts_and_attributes.py` | Joint concept-attribute analysis |
| `analyze_concept_attributes.py` | Concept-attribute alignment scoring |
| `analyze_aspect_rationales.py` | Aspect-specific rationale analysis |
| `analyze_rationale_concept_relationship.py` | Rationale-concept relationship mapping |
| `compare_attribute_rationales.py` | Cross-attribute rationale comparison |
| `analyze_rationales_by_aspect.py` | Aspect-grouped rationale analysis |

### 7.5 Extraction Scripts (2)

| Script | Description |
|---|---|
| `extract_concept_rationales.py` | Extract rationale spans per concept |
| `extract_rationales_concepts.py` | Extract rationale-concept mappings |

### 7.6 CEBaB-Specific Tools (4)

| Script | Description |
|---|---|
| `cebab_mapper.py` | Concept-to-rationale mapping with cosine similarity |
| `map_cebab_concepts.py` | Map concepts to CEBaB aspect annotations |
| `cebab_test_examples.py` | Run test examples through CEBaB model |
| `check_cebab_attributes.py` | Validate CEBaB attribute annotations |
| `analyze_cebab_concepts.py` | Analyze concept alignment with CEBaB aspects |

### 7.7 Utility Scripts (4)

| Script | Description |
|---|---|
| `plot_history.py` | Plot training history (loss, accuracy curves) |
| `examine_checkpoint.py` | Inspect saved model checkpoints |
| `direct_predict.py` | Direct prediction on raw text |
| `simple_predict.py` | Simplified prediction interface |

### 7.8 Shell Scripts (12)

| Script | Description |
|---|---|
| `visualize_cebab_model.sh` | Automate CEBaB model visualization |
| `visualize_sst2_model.sh` | Automate SST-2 visualization |
| `visualize_trained_model.sh` | Visualize any trained model |
| `visualize_existing_model.sh` | Visualize from existing checkpoint |
| `visualize_legacy_model.sh` | Handle legacy model formats |
| `visualize_examples.sh` | Batch example visualization |
| `run_cebab_test.sh` | Run CEBaB test suite |
| `run_ablation_analysis.sh` | Run full ablation study |
| `run_ablation_on_existing_model.sh` | Ablation on pre-trained model |
| `analyze_concept_attributes.sh` | Concept-attribute analysis pipeline |
| `analyze_rationale_concept.sh` | Rationale-concept analysis pipeline |
| `cebab_map.sh` | CEBaB concept mapping pipeline |

---

## 8. Usage Patterns

### 8.1 Train a Model

```bash
# Default AG News training
python main.py

# Quick experiment
python main.py --small --fast --visualize

# CEBaB training
python train_on_cebab.py --model_name distilbert-base-uncased --num_concepts 50

# With causal discovery
python main.py --causal_discovery --visualize
```

### 8.2 Generate Explanations

```bash
# Basic explanation
python generate_explanations.py \
  --checkpoint_path models/DIR/checkpoints/best_model.pt \
  --config_path models/DIR/config.json \
  --text "Your text here" \
  --mode basic

# Concept interventions
python generate_explanations.py ... --mode intervention

# Counterfactual explanations
python generate_explanations.py ... --mode counterfactual --target_class 2

# All explanation types to JSON
python generate_explanations.py ... --mode all --output_file explanations.json
```

### 8.3 Visualize Results

```bash
# CEBaB visualization
python visualize_cebab_model.py \
  --model_dir /path/to/model --attribute food --num_examples 5

# LaTeX export
python create_latex_visualization.py --model_dir /path/to/model
```

### 8.4 Run Analysis

```bash
# Ablation study
python ablation_analysis.py --model_path /path/to/model --dataset cebab

# Concept-attribute analysis
python analyze_concept_attributes.py --model_dir /path/to/model

# Contrastive analysis
python explore_concept_space.py ... --mode contrastive \
  --text "First text" --text2 "Second text"
```

### 8.5 Inference Only

```bash
python main.py --inference_only --model_path /path/to/checkpoint
python direct_predict.py --model_path /path/to/model --text "Classify this"
```

---

## 9. Explanation Types

CLARITY supports five categories of explanation:

### 9.1 Basic Explanations
Shows the rationale (highlighted text spans) and top activated concepts for a prediction. Provides predicted class, confidence, and rationale percentage.

### 9.2 Concept Interventions
Modifies individual concept values (set to 0.0 or 1.0) at inference time to observe how the prediction changes. Reveals which concepts are decision-critical.

### 9.3 Counterfactual Explanations
Finds the minimal set of concept changes that would flip the prediction to a target class. Answers "what would need to change for a different outcome?"

### 9.4 Contrastive Explanations
Compares explanations between two inputs side-by-side. Identifies which concepts differ significantly and highlights divergent rationale patterns.

### 9.5 Causal Discovery
Uses the PC algorithm (constraint-based causal discovery) to build a causal graph over learned concepts. Reveals directional relationships and independence structures.

---

## 10. Extending CLARITY

### 10.1 Adding a New Dataset

1. Implement a data loading function using `datasets.load_dataset()` or a custom loader.
2. Set `num_labels` in `ModelConfig` to match the new dataset's class count.
3. Map the dataset columns to `text` and `label` fields expected by the model.
4. Add dataset-specific class names for human-readable explanations.

### 10.2 Using a Different Base Encoder

Change `base_model_name` in `ModelConfig` to any Hugging Face model identifier:
```python
config = ModelConfig(base_model_name="roberta-base", ...)
```
The hidden size is auto-detected from the loaded model.

### 10.3 Custom Concept Analysis

- Inspect `concept_probs` output tensor to see which concepts activated for a given input.
- Use `explain_prediction()` to get structured explanations.
- Use `explore_concept_space.py` for systematic concept importance analysis.

### 10.4 Checkpoints

Models save to `{output_dir}/{timestamp}/`:
- `checkpoints/best_model.pt` - Model weights
- `config.json` - Full configuration for reproducibility
- `plots/` - Training curves
- `visualizations/` - Generated visualizations

---

## License

CLARITY is released under the **MIT License**.

All open source dependencies are under permissive licenses (MIT, BSD-3-Clause, Apache-2.0, PSF, MPL-2.0) that are compatible with MIT-licensed projects and commercial use.

---

*Knowledge Base generated for CLARITY v1.0 - Last updated: 2026-04-08*
