#!/usr/bin/env python
"""
Example script to generate explanations from a checkpoint

This is a simplified example showing how to load a checkpoint and
generate explanations for a specific text.
"""

import torch
import json
import logging
from transformers import AutoTokenizer
from optimized_rationale_concept_model import RationaleConceptBottleneckModel, ModelConfig

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Path to your checkpoint and config
CHECKPOINT_PATH = "/Users/tanmoy/research/Concept_Learning/text_classification_expts/results/distilbert-base-uncased_sst2_1/sst2_best_model.pt"
CONFIG_PATH = "/Users/tanmoy/research/Concept_Learning/text_classification_expts/results/distilbert-base-uncased_sst2_1/sst2_config.json"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
print(f"Using device: {device}")

def load_config_from_json(config_path):
    """
    Load configuration from a JSON file and convert it to ModelConfig

    This function handles different config formats and converts them to
    the format expected by ModelConfig
    """
    # Load the JSON file
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Check if it's the new format with 'model' key
    if 'model' in config_data:
        logger.info("Detected new config format, converting to ModelConfig format")

        # Extract model name
        base_model_name = config_data['model']['model_name']
        if base_model_name.startswith('models/'):
            base_model_name = base_model_name[7:]  # Remove 'models/' prefix

        # Extract number of classes for the dataset
        dataset_name = config_data.get('dataset_name', 'sst2')
        num_classes_str = config_data['model']['num_classes']
        # Convert string representation to dict if needed
        if isinstance(num_classes_str, str) and '{' in num_classes_str:
            import ast
            num_classes_dict = ast.literal_eval(num_classes_str)
            num_labels = num_classes_dict.get(dataset_name, 2)
        else:
            num_labels = int(num_classes_str)

        # Extract concept count - handle string representation of concept config
        concept_config = config_data['model']['concept']
        if isinstance(concept_config, str):
            # Parse the string representation
            import re
            concept_counts_match = re.search(r"concept_counts=({[^}]+})", concept_config)
            if concept_counts_match:
                import ast
                concept_counts_str = concept_counts_match.group(1)
                concept_counts = ast.literal_eval(concept_counts_str)
                num_concepts = concept_counts.get(dataset_name, 100)
            else:
                num_concepts = 100  # Default

            # Check for concept interactions in the string
            enable_concept_interactions = 'enable_concept_interactions=True' in concept_config
        else:
            # It's a dictionary
            concept_counts_str = concept_config.get('concept_counts', '{"sst2": 100}')
            if isinstance(concept_counts_str, str) and '{' in concept_counts_str:
                import ast
                concept_counts = ast.literal_eval(concept_counts_str)
                num_concepts = concept_counts.get(dataset_name, 100)
            else:
                num_concepts = 100  # Default

            enable_concept_interactions = concept_config.get('enable_concept_interactions', 'True') == 'True'

        # Extract other parameters with defaults
        use_skip_connection = config_data['model'].get('use_skip_connection', 'True') == 'True'

        # Create ModelConfig
        return ModelConfig(
            base_model_name=base_model_name,
            num_labels=num_labels,
            num_concepts=num_concepts,
            use_skip_connection=use_skip_connection,
            enable_concept_interactions=enable_concept_interactions
        )
    else:
        # If it's already in the expected format, use ModelConfig.load
        return ModelConfig.load(config_path)

def main():
    # Load configuration
    config = load_config_from_json(CONFIG_PATH)
    logger.info(f"Loaded configuration: {config.base_model_name}, {config.num_labels} classes, {config.num_concepts} concepts")

    # Initialize model with config
    model = RationaleConceptBottleneckModel(config)

    # Load checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Example text to explain
    text = "The company announced a new product that will revolutionize the market."

    print(f"\nGenerating explanation for: \"{text}\"")

    # 1. Basic explanation
    explanation = model.explain_prediction(tokenizer, text)

    # Print explanation
    print("\n" + "="*80)
    print(f"PREDICTION: {explanation['prediction']} (Confidence: {explanation['confidence']:.4f})")
    print("-"*80)
    print(f"RATIONALE: \"{explanation['rationale']}\"")
    print(f"Rationale length: {explanation['rationale_percentage']*100:.1f}% of text")
    print("-"*80)
    print("TOP CONCEPTS:")
    for concept, prob in explanation['top_concepts']:
        print(f"  {concept}: {prob:.4f}")
    print("="*80 + "\n")

    # 2. Concept intervention
    # Choose a concept to intervene on (e.g., the first concept from top concepts)
    if explanation['top_concepts']:
        concept_idx = int(explanation['top_concepts'][0][0].split('_')[1])

        # Try setting the concept to 0 (off)
        print(f"Intervening on {explanation['top_concepts'][0][0]}:")

        # Original value
        original_value = explanation['top_concepts'][0][1]
        print(f"  Original value: {original_value:.4f}")

        # Set to 0
        intervention_0 = model.intervene_on_concepts(tokenizer, text, concept_idx, 0.0)
        print(f"  Setting to 0.0:")
        print(f"    Original prediction: {intervention_0['original_prediction']}")
        print(f"    New prediction: {intervention_0['intervened_prediction']}")

        # Set to 1
        intervention_1 = model.intervene_on_concepts(tokenizer, text, concept_idx, 1.0)
        print(f"  Setting to 1.0:")
        print(f"    Original prediction: {intervention_1['original_prediction']}")
        print(f"    New prediction: {intervention_1['intervened_prediction']}")

    # 3. Try multiple texts
    print("\n" + "="*80)
    print("TESTING MULTIPLE TEXTS")
    print("="*80)

    test_texts = [
        "The team won the championship after an incredible comeback.",
        "The stock market crashed yesterday, causing widespread panic.",
        "Scientists discovered a new species of frog in the Amazon rainforest."
    ]

    for test_text in test_texts:
        # Generate explanation
        test_explanation = model.explain_prediction(tokenizer, test_text)

        # Print simplified explanation
        print(f"\nText: \"{test_text}\"")
        print(f"Prediction: {test_explanation['prediction']}")
        print(f"Rationale: \"{test_explanation['rationale']}\"")
        print("Top concepts: " + ", ".join([f"{c[0]}:{c[1]:.2f}" for c in test_explanation['top_concepts'][:3]]))
        print("-"*80)

if __name__ == "__main__":
    main()
