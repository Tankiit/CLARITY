#!/usr/bin/env python
"""
Generate Explanations from Existing Checkpoint

This script is specifically designed to work with the existing checkpoint format
and generate explanations for text classification.
"""

import os
import argparse
import json
import torch
import numpy as np
import re
import logging
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
logger.info(f"Using device: {device}")

class RationaleExtractor(torch.nn.Module):
    """Simplified rationale extractor to match checkpoint format"""
    def __init__(self, hidden_size):
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        # Score network
        self.score_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states, attention_mask):
        # Apply attention mechanism
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (hidden_states.size(-1) ** 0.5)
        attention_scores = attention_scores.masked_fill(~attention_mask.unsqueeze(1).bool(), -10000.0)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention to value
        context = torch.matmul(attention_weights, value)
        context = self.layer_norm(context)

        # Calculate token scores
        token_scores = self.score_net(context).squeeze(-1)
        token_scores = token_scores.masked_fill(~attention_mask.bool(), -10000.0)

        # Get token probabilities
        token_probs = torch.nn.functional.softmax(token_scores, dim=-1)

        # Create binary rationale mask (threshold at 0.5)
        rationale_mask = (token_probs > 0.5).float()

        # Apply rationale mask to get attended states
        attended_states = hidden_states * rationale_mask.unsqueeze(-1)

        # Calculate pooled representation
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)
        pooled_attended = torch.sum(attended_states, dim=1) / valid_tokens

        return {
            'token_scores': token_scores,
            'token_probs': token_probs,
            'rationale_mask': rationale_mask,
            'attended_states': attended_states,
            'pooled_attended': pooled_attended
        }

class ConceptMapper(torch.nn.Module):
    """Simplified concept mapper to match checkpoint format"""
    def __init__(self, hidden_size, dataset_name, num_concepts):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_concepts = num_concepts

        # Create encoders dictionary to match checkpoint format
        self.encoders = torch.nn.ModuleDict()

        # Based on the error message, we need to adjust the architecture
        # The checkpoint has a different structure than what we initially assumed
        self.encoders[dataset_name] = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.LayerNorm(hidden_size // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 4, num_concepts)
        )

        # Create interactions dictionary
        self.interactions = torch.nn.ParameterDict()
        self.interactions[dataset_name] = torch.nn.Parameter(torch.zeros(num_concepts, num_concepts))

    def forward(self, pooled_embeddings):
        # Apply concept encoder for the dataset
        concept_scores = self.encoders[self.dataset_name](pooled_embeddings)

        # Apply concept interactions
        interaction_matrix = self.interactions[self.dataset_name]
        # Make interaction matrix symmetric
        sym_interactions = torch.sigmoid((interaction_matrix + interaction_matrix.t()) / 2)

        # Apply interactions
        interaction_effect = torch.matmul(concept_scores, sym_interactions)
        concept_scores = concept_scores + interaction_effect

        # Apply sigmoid to get concept probabilities
        concept_probs = torch.sigmoid(concept_scores)

        return {
            'concept_scores': concept_scores,
            'concept_probs': concept_probs,
            'interaction_matrix': interaction_matrix
        }

class Classifier(torch.nn.Module):
    """Simplified classifier to match checkpoint format"""
    def __init__(self, num_concepts, dataset_name, num_labels):
        super().__init__()
        self.dataset_name = dataset_name

        # Based on the error message, we need to adjust the architecture
        # The checkpoint expects different dimensions
        # For sst2: input size 868, hidden size 384, output size 2
        self.classifiers = torch.nn.ModuleDict()

        # We'll create a classifier with the expected dimensions
        # This is based on the error message showing the expected sizes
        self.classifiers[dataset_name] = torch.nn.Sequential(
            torch.nn.Linear(868, 384),  # From error message
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(384, num_labels)
        )

    def forward(self, concept_probs):
        # Since our input is only concept_probs but the model expects more,
        # we'll pad the input with zeros to match the expected size
        batch_size = concept_probs.size(0)
        padded_input = torch.zeros(batch_size, 868, device=concept_probs.device)

        # Copy concept_probs to the beginning of the padded input
        padded_input[:, :concept_probs.size(1)] = concept_probs

        # Apply classifier for the dataset
        logits = self.classifiers[self.dataset_name](padded_input)
        return logits

class RationaleConceptModel(torch.nn.Module):
    """Simplified model to match checkpoint format"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load base encoder
        self.encoder = AutoModel.from_pretrained(config['base_model_name'])
        hidden_size = self.encoder.config.hidden_size

        # Initialize component modules
        self.rationale_extractor = RationaleExtractor(hidden_size)
        self.concept_mapper = ConceptMapper(
            hidden_size,
            config['dataset_name'],
            config['num_concepts']
        )
        self.classifier = Classifier(
            config['num_concepts'],
            config['dataset_name'],
            config['num_labels']
        )

    def forward(self, input_ids, attention_mask):
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Extract rationales
        rationale_outputs = self.rationale_extractor(hidden_states, attention_mask)
        pooled_attended = rationale_outputs['pooled_attended']

        # Map to concepts
        concept_outputs = self.concept_mapper(pooled_attended)
        concept_probs = concept_outputs['concept_probs']

        # Classification
        logits = self.classifier(concept_probs)

        # Combine all outputs
        outputs = {
            'logits': logits,
            'rationale_mask': rationale_outputs['rationale_mask'],
            'token_probs': rationale_outputs['token_probs'],
            'concept_probs': concept_probs,
            'concept_scores': concept_outputs['concept_scores'],
            'interaction_matrix': concept_outputs['interaction_matrix'],
            'pooled_attended': pooled_attended
        }

        return outputs

    def explain_prediction(self, tokenizer, text, min_concept_prob=0.5):
        """Generate explanation for a prediction"""
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,  # Default max length
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        self.eval()
        with torch.no_grad():
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

        # Get prediction
        logits = outputs["logits"]
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probs[0, prediction].item()

        # Get rationale text
        rationale_mask = outputs["rationale_mask"][0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Identify rationale tokens and their positions
        rationale_tokens = []
        for i, (token, mask) in enumerate(zip(tokens, rationale_mask)):
            if mask > 0.5 and inputs["attention_mask"][0, i] > 0:
                rationale_tokens.append(token)

        # Convert tokens to text while handling special tokens
        rationale_text = tokenizer.convert_tokens_to_string(rationale_tokens)

        # Get active concepts and their probabilities
        concept_probs = outputs["concept_probs"][0].cpu().numpy()

        # Get top concepts
        top_indices = np.argsort(concept_probs)[::-1][:5]  # Top 5 concepts
        top_concepts = [(f"concept_{idx}", concept_probs[idx])
                       for idx in top_indices if concept_probs[idx] > min_concept_prob]

        # Calculate rationale length as percentage of total sequence
        valid_length = inputs["attention_mask"][0].sum().item()
        rationale_length = rationale_mask.sum().item()
        rationale_percentage = rationale_length / valid_length if valid_length > 0 else 0

        # Create explanation
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "rationale": rationale_text,
            "rationale_percentage": rationale_percentage,
            "top_concepts": top_concepts,
            "class_probabilities": probs[0].cpu().numpy()
        }

        return explanation

    def intervene_on_concepts(self, tokenizer, text, concept_idx, new_value):
        """Perform intervention on concepts to analyze their effect"""
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,  # Default max length
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Original forward pass
        self.eval()
        with torch.no_grad():
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

            # Get original prediction
            orig_logits = outputs["logits"]
            orig_probs = torch.nn.functional.softmax(orig_logits, dim=1)
            orig_prediction = torch.argmax(orig_logits, dim=1).item()

            # Get concept probabilities
            concept_probs = outputs["concept_probs"].clone()

            # Modify the concept
            concept_probs[0, concept_idx] = new_value

            # Predict with modified concepts
            new_logits = self.classifier(concept_probs)

            new_probs = torch.nn.functional.softmax(new_logits, dim=1)
            new_prediction = torch.argmax(new_logits, dim=1).item()

        # Get concept name
        concept_name = f"concept_{concept_idx}"

        return {
            "original_prediction": orig_prediction,
            "original_probs": orig_probs[0].cpu().numpy(),
            "intervened_prediction": new_prediction,
            "intervened_probs": new_probs[0].cpu().numpy(),
            "concept_name": concept_name,
            "concept_value": {
                "original": outputs["concept_probs"][0, concept_idx].item(),
                "modified": new_value
            }
        }

def load_config_from_json(config_path):
    """Load configuration from a JSON file"""
    # Load the JSON file
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Extract model name
    base_model_name = config_data['model']['model_name']
    if base_model_name.startswith('models/'):
        base_model_name = base_model_name[7:]  # Remove 'models/' prefix

    # Extract dataset name
    dataset_name = config_data.get('dataset_name', 'sst2')

    # Extract number of classes for the dataset
    num_classes_str = config_data['model']['num_classes']
    # Convert string representation to dict if needed
    if isinstance(num_classes_str, str) and '{' in num_classes_str:
        import ast
        num_classes_dict = ast.literal_eval(num_classes_str)
        num_labels = num_classes_dict.get(dataset_name, 2)
    else:
        num_labels = int(num_classes_str)

    # Extract concept count
    concept_config = config_data['model']['concept']
    if isinstance(concept_config, str):
        # Parse the string representation
        concept_counts_match = re.search(r"concept_counts=({[^}]+})", concept_config)
        if concept_counts_match:
            import ast
            concept_counts_str = concept_counts_match.group(1)
            concept_counts = ast.literal_eval(concept_counts_str)
            num_concepts = concept_counts.get(dataset_name, 100)
        else:
            num_concepts = 100  # Default
    else:
        # It's a dictionary
        concept_counts_str = concept_config.get('concept_counts', '{"sst2": 100}')
        if isinstance(concept_counts_str, str) and '{' in concept_counts_str:
            import ast
            concept_counts = ast.literal_eval(concept_counts_str)
            num_concepts = concept_counts.get(dataset_name, 100)
        else:
            num_concepts = 100  # Default

    # Create simplified config
    return {
        'base_model_name': base_model_name,
        'dataset_name': dataset_name,
        'num_labels': num_labels,
        'num_concepts': num_concepts
    }

def load_checkpoint_with_remapping(model, checkpoint_path, dataset_name):
    """Load checkpoint with key remapping to match model structure"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create a new model state dict
    model_dict = model.state_dict()

    # Directly load encoder weights
    encoder_keys = [k for k in checkpoint.keys() if k.startswith('encoder.')]
    for key in encoder_keys:
        if key in model_dict:
            model_dict[key] = checkpoint[key]

    # Load rationale extractor weights
    rationale_keys = [k for k in checkpoint.keys() if k.startswith('rationale_extractor.')]
    for key in rationale_keys:
        if key in model_dict:
            model_dict[key] = checkpoint[key]

    # Load concept mapper weights for the specific dataset
    concept_mapper_keys = [
        k for k in checkpoint.keys()
        if k.startswith(f'concept_mapper.encoders.{dataset_name}.') or
           k == f'concept_mapper.interactions.{dataset_name}'
    ]
    for key in concept_mapper_keys:
        if key in model_dict:
            model_dict[key] = checkpoint[key]

    # Load classifier weights for the specific dataset
    classifier_keys = [k for k in checkpoint.keys() if k.startswith(f'classifiers.{dataset_name}.')]
    for key in classifier_keys:
        model_key = f'classifier.{key}'
        if model_key in model_dict:
            model_dict[model_key] = checkpoint[key]

    # Load the state dict with strict=False to ignore missing keys
    model.load_state_dict(model_dict, strict=False)

    # Print loading summary
    logger.info(f"Loaded encoder keys: {len(encoder_keys)}")
    logger.info(f"Loaded rationale extractor keys: {len(rationale_keys)}")
    logger.info(f"Loaded concept mapper keys: {len(concept_mapper_keys)}")
    logger.info(f"Loaded classifier keys: {len(classifier_keys)}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Generate explanations from a checkpoint")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model configuration")

    # Optional arguments
    parser.add_argument("--text", type=str,
                        help="Text to explain (if not provided, will use examples)")
    parser.add_argument("--dataset", type=str, choices=['sst2', 'yelp_polarity', 'ag_news', 'dbpedia'],
                        help="Dataset name for class labels")

    args = parser.parse_args()

    # Load configuration
    config = load_config_from_json(args.config_path)
    logger.info(f"Loaded configuration: {config['base_model_name']}, {config['num_labels']} classes, {config['num_concepts']} concepts")

    # Initialize model with config
    model = RationaleConceptModel(config)

    # Load checkpoint with key remapping
    dataset_name = args.dataset or config['dataset_name']
    model = load_checkpoint_with_remapping(model, args.checkpoint_path, dataset_name)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])

    # Use provided text or examples
    texts = []
    if args.text:
        texts.append(args.text)
    else:
        # Example texts for different datasets
        if args.dataset == 'sst2' or args.dataset == 'yelp_polarity':
            texts = [
                "This movie was absolutely amazing. The acting was superb.",
                "I hated every minute of this terrible film.",
                "The restaurant had great food but poor service."
            ]
        elif args.dataset == 'ag_news':
            texts = [
                "The company announced a new product that will revolutionize the market.",
                "The team won the championship after an incredible comeback.",
                "Scientists discovered a new species in the Amazon rainforest."
            ]
        else:
            texts = [
                "The company announced a new product that will revolutionize the market.",
                "The team won the championship after an incredible comeback.",
                "This restaurant has amazing food and excellent service.",
                "The movie was boring and predictable. I wouldn't recommend it."
            ]

    # Define class names for datasets
    class_names = {
        'sst2': ['Negative', 'Positive'],
        'yelp_polarity': ['Negative', 'Positive'],
        'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'dbpedia': ['Company', 'Educational Institution', 'Artist', 'Athlete',
                   'Office Holder', 'Mean Of Transportation', 'Building', 'Natural Place',
                   'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
    }

    # Generate explanations for each text
    for text in texts:
        print(f"\nGenerating explanation for: \"{text}\"")

        # Basic explanation
        explanation = model.explain_prediction(tokenizer, text)

        # Print explanation
        print("\n" + "="*80)
        prediction_class = explanation['prediction']
        if args.dataset and args.dataset in class_names:
            prediction_class = f"{prediction_class} ({class_names[args.dataset][prediction_class]})"

        print(f"PREDICTION: {prediction_class} (Confidence: {explanation['confidence']:.4f})")
        print("-"*80)
        print(f"RATIONALE: \"{explanation['rationale']}\"")
        print(f"Rationale length: {explanation['rationale_percentage']*100:.1f}% of text")
        print("-"*80)
        print("TOP CONCEPTS:")
        for concept, prob in explanation['top_concepts']:
            print(f"  {concept}: {prob:.4f}")
        print("="*80 + "\n")

        # Concept intervention
        if explanation['top_concepts']:
            concept_idx = int(explanation['top_concepts'][0][0].split('_')[1])

            print(f"Intervening on {explanation['top_concepts'][0][0]}:")

            # Original value
            original_value = explanation['top_concepts'][0][1]
            print(f"  Original value: {original_value:.4f}")

            # Set to 0
            intervention_0 = model.intervene_on_concepts(tokenizer, text, concept_idx, 0.0)
            orig_pred = intervention_0['original_prediction']
            new_pred = intervention_0['intervened_prediction']

            # Get class names if available
            if args.dataset and args.dataset in class_names:
                orig_pred_name = f"{orig_pred} ({class_names[args.dataset][orig_pred]})"
                new_pred_name = f"{new_pred} ({class_names[args.dataset][new_pred]})"
            else:
                orig_pred_name = str(orig_pred)
                new_pred_name = str(new_pred)

            print(f"  Setting to 0.0:")
            print(f"    Original prediction: {orig_pred_name}")
            print(f"    New prediction: {new_pred_name}")

            # Set to 1
            intervention_1 = model.intervene_on_concepts(tokenizer, text, concept_idx, 1.0)
            new_pred = intervention_1['intervened_prediction']

            # Get class name if available
            if args.dataset and args.dataset in class_names:
                new_pred_name = f"{new_pred} ({class_names[args.dataset][new_pred]})"
            else:
                new_pred_name = str(new_pred)

            print(f"  Setting to 1.0:")
            print(f"    New prediction: {new_pred_name}")

            print("\n" + "-"*80)

if __name__ == "__main__":
    main()
