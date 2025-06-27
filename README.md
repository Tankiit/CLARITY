# CLARITY: Text Classification with Concept Learning and Rationale Extraction

This repository contains experiments for text classification using concept learning and rationale extraction methods. The focus is on interpretable machine learning models that can provide explanations for their predictions.

## Overview

The project implements various text classification models with emphasis on:
- **Rationale Extraction**: Identifying important text spans that contribute to predictions
- **Concept Learning**: Learning interpretable concept representations
- **Visualization**: Creating comprehensive visualizations of model decisions
- **Ablation Studies**: Analyzing the contribution of different model components

## Key Features

### Models
- Rationale-Concept Models for interpretable text classification
- Support for multiple datasets: CEBaB, AG News, SST-2, Yelp, DBpedia
- Transformer-based architectures (DistilBERT, RoBERTa)

### Visualization Tools
- Interactive HTML visualizations of model predictions
- Concept activation analysis
- Rationale highlighting and explanation generation
- LaTeX-compatible visualization exports

### Analysis Tools
- Ablation studies for understanding model components
- Concept-attribute mapping analysis
- Aspect-based rationale comparison
- Comprehensive reporting tools

## Main Scripts

### Training
- `main.py`: Main training script for text classification models
- `train_on_cebab.py`: Specialized training for CEBaB dataset
- `optimized_rationale_concept_model.py`: Optimized model implementation

### Visualization
- `visualize_cebab_model.py`: Comprehensive CEBaB model visualization
- `visualize_examples.py`: General example visualization
- `create_visualization_summary.py`: Summary visualization creation
- `create_latex_visualization.py`: LaTeX-compatible visualizations

### Analysis
- `ablation_analysis.py`: Ablation study implementation
- `ablation_study.py`: Additional ablation analysis tools
- `compare_attribute_rationales.py`: Attribute-based rationale comparison
- `analyze_aspect_rationales.py`: Aspect rationale analysis

### Explanation Generation
- `generate_explanations.py`: General explanation generation
- `generate_cebab_explanations.py`: CEBaB-specific explanations
- `extract_concept_rationales.py`: Concept rationale extraction
- `simple_explanations.py`: Simplified explanation interface

## Datasets

The repository supports multiple text classification datasets:

1. **CEBaB (Consumer Reviews)**: Restaurant review dataset with aspect-based annotations
2. **AG News**: News categorization dataset
3. **SST-2**: Stanford Sentiment Treebank
4. **Yelp Polarity**: Yelp review sentiment classification
5. **DBpedia**: Ontology classification

## Usage

### Basic Training
```bash
python main.py --dataset ag_news --model distilbert-base-uncased
```

### CEBaB Training
```bash
python train_on_cebab.py --model_name distilbert-base-uncased --num_concepts 50
```

### Visualization
```bash
python visualize_cebab_model.py --model_dir /path/to/model --attribute food --num_examples 5
```

### Ablation Analysis
```bash
python ablation_analysis.py --model_path /path/to/model --dataset cebab
```

## Model Architecture

The core model implements a three-stage pipeline:
1. **Input → Rationale**: Extract important text spans
2. **Rationale → Concepts**: Map rationales to interpretable concepts
3. **Concepts → Prediction**: Make final classification

## Installation

```bash
# Clone the repository
git clone https://github.com/Tankiit/CLARITY.git
cd CLARITY

# Install dependencies
pip install torch transformers datasets matplotlib seaborn pandas numpy tqdm
```

## Directory Structure

```
├── *.py                    # Main Python scripts
├── *.sh                    # Shell scripts for automation
├── test_examples/          # Example texts for testing
├── latex_cebab_visualizations/  # LaTeX visualization exports
├── README.md              # This file
├── README_concept_rationales.md  # Detailed concept rationale documentation
└── .gitignore            # Git ignore file
```

## Key Concepts

### Rationales
Text spans that are most important for the model's decision. The model learns to identify these automatically during training.

### Concepts
High-level, interpretable features that the model learns to associate with different aspects of the input (e.g., "food quality", "service quality" for restaurant reviews).

### Ablation Studies
Systematic removal or modification of model components to understand their individual contributions to performance.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{clarity2025,
  title={CLARITY: Text Classification with Concept Learning and Rationale Extraction},
  author={[Your Name]},
  year={2025},
  url={https://github.com/Tankiit/CLARITY}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 