"""
Visualization script for CEBaB dataset models.

This script loads a pretrained model for the CEBaB dataset and visualizes:
1. Input to rationale mapping
2. Rationale to concept mapping
3. Concept to prediction mapping
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import time
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

# Custom JSON encoder for NumPy and PyTorch types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
print(f"Using device: {device}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize CEBaB model predictions")

    # Model checkpoint
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing the model files")

    # Model filename
    parser.add_argument("--model_filename", type=str, default="cebab_best_model.pt",
                       help="Filename of the model checkpoint")

    # Config filename
    parser.add_argument("--config_filename", type=str, default="cebab_config.json",
                       help="Filename of the model configuration")

    # Attribute to analyze
    parser.add_argument("--attribute", type=str, default="food",
                       choices=["food", "ambiance", "service", "noise", "price"],
                       help="Attribute to analyze")

    # Number of examples
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to visualize")

    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations (default: model_dir/visualizations_timestamp)")

    # Rationale threshold
    parser.add_argument("--rationale_threshold", type=float, default=0.2,
                       help="Threshold for rationale extraction")

    # Concept threshold
    parser.add_argument("--concept_threshold", type=float, default=0.1,
                       help="Threshold for concept activation")

    return parser.parse_args()

class RationaleConceptModel(torch.nn.Module):
    """A model for loading the CEBaB model weights"""
    def __init__(self, num_concepts=50, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        # Rationale extractor components
        self.rationale_extractor = torch.nn.ModuleDict({
            'query': torch.nn.Linear(768, 384),
            'key': torch.nn.Linear(768, 384),
            'score_ffn': torch.nn.Sequential(
                torch.nn.Linear(768, 384),
                torch.nn.LayerNorm(384),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(384, 1)
            )
        })

        # Concept mapper
        self.concept_mapper = torch.nn.ModuleDict({
            'concept_encoder': torch.nn.Sequential(
                torch.nn.Linear(768, 384),
                torch.nn.LayerNorm(384),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(384, num_concepts)
            )
        })

        # Classifier
        # The classifier takes both concepts and hidden states (skip connection)
        # 768 (hidden_size) + 50 (num_concepts) = 818
        self.classifier = torch.nn.Linear(768 + num_concepts, num_classes)

        # Configuration
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.target_rationale_percentage = 0.2
        self.use_skip_connection = True

    def forward(self, input_ids, attention_mask):
        # Encode text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Extract rationale
        query = self.rationale_extractor['query'](hidden_states)  # [batch_size, seq_len, hidden_size]
        key = self.rationale_extractor['key'](hidden_states)  # [batch_size, seq_len, hidden_size]

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (384 ** 0.5)  # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores.masked_fill(
            (1 - attention_mask.unsqueeze(1)) > 0, -1e9
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Apply attention
        context_vectors = torch.matmul(attention_weights, hidden_states)  # [batch_size, seq_len, hidden_size]

        # Compute rationale scores
        rationale_scores = self.rationale_extractor['score_ffn'](context_vectors).squeeze(-1)  # [batch_size, seq_len]
        rationale_scores = rationale_scores.masked_fill((1 - attention_mask) > 0, -1e9)

        # Apply sigmoid to get rationale mask
        rationale_mask = torch.sigmoid(rationale_scores)  # [batch_size, seq_len]

        # Apply rationale mask to hidden states
        masked_hidden_states = hidden_states * rationale_mask.unsqueeze(-1)  # [batch_size, seq_len, hidden_size]

        # Pool masked hidden states
        masked_pooled = masked_hidden_states.sum(dim=1) / (rationale_mask.sum(dim=1, keepdim=True) + 1e-10)  # [batch_size, hidden_size]

        # Map to concepts
        concept_logits = self.concept_mapper['concept_encoder'](masked_pooled)  # [batch_size, num_concepts]

        # Apply sigmoid to get concept scores
        concept_scores = torch.sigmoid(concept_logits)  # [batch_size, num_concepts]

        # Pool hidden states for skip connection
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Concatenate concept scores with pooled hidden states (skip connection)
        if self.use_skip_connection:
            combined_features = torch.cat([concept_scores, pooled_hidden], dim=1)  # [batch_size, num_concepts + hidden_size]
        else:
            combined_features = concept_scores

        # Classify
        logits = self.classifier(combined_features)  # [batch_size, num_classes]

        return {
            'logits': logits,
            'rationale_mask': rationale_mask,
            'concept_scores': concept_scores,
            'hidden_states': hidden_states,
            'masked_pooled': masked_pooled
        }

def load_cebab_dataset():
    """Load the CEBaB dataset"""
    print("Loading CEBaB dataset...")
    cebab = load_dataset("CEBaB/CEBaB")
    return cebab

def get_attribute_examples(dataset, attribute, num_examples=5):
    """Get examples for a specific attribute"""
    # Print dataset structure for debugging
    print(f"Dataset structure: {list(dataset.keys())}")
    print(f"Test set size: {len(dataset['test'])}")
    print(f"Example keys: {list(dataset['test'][0].keys())}")

    # The attribute in CEBaB is in the format 'attribute_aspect_majority'
    attribute_key = f"{attribute}_aspect_majority"

    # Filter examples with the specified attribute that have meaningful ratings
    filtered = [ex for ex in dataset['test']
                if attribute_key in ex
                and ex[attribute_key] is not None
                and ex[attribute_key] not in ['', 'unknown']
                and 'description' in ex
                and ex['description']]

    print(f"Found {len(filtered)} examples with meaningful {attribute_key} ratings")

    if len(filtered) == 0:
        print("No examples found with meaningful ratings for the specified attribute. Using random examples.")
        # Use random examples if no examples with the attribute are found
        import random
        # Convert to list if it's not already a list
        test_data = list(dataset['test'])
        filtered = random.sample(test_data, min(num_examples * 2, len(test_data)))

    # Select a diverse set of examples with different ratings
    examples = []
    ratings = set()

    for ex in filtered:
        # Get rating
        rating = ex[attribute_key]
        if rating not in ratings and len(examples) < num_examples:
            examples.append(ex)
            ratings.add(rating)

    # If we don't have enough examples with different ratings, add more
    if len(examples) < num_examples:
        remaining = [ex for ex in filtered if ex[attribute_key] not in ratings]
        examples.extend(remaining[:num_examples - len(examples)])

    # If still not enough, just use the first few examples
    if len(examples) < num_examples:
        examples.extend(filtered[:num_examples - len(examples)])

    # Print the selected examples
    print(f"Selected {len(examples)} examples:")
    for i, ex in enumerate(examples):
        description = ex['description'][:100] + '...' if len(ex['description']) > 100 else ex['description']
        print(f"  Example {i+1}: Rating: {ex[attribute_key]}, Description: {description}")

    return examples[:num_examples]

def visualize_input_to_rationale(text, rationale_mask, tokenizer, save_path=None):
    """Visualize the mapping from input text to rationale"""
    # Tokenize text
    tokens = tokenizer.tokenize(text)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot rationale mask
    plt.bar(range(len(rationale_mask)), rationale_mask, color='skyblue')

    # Add token labels
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')

    # Add title and labels
    plt.title('Input to Rationale Mapping', fontsize=14)
    plt.xlabel('Tokens')
    plt.ylabel('Rationale Score')
    plt.ylim(0, 1)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_rationale_to_concept(concept_scores, top_k=10, save_path=None):
    """Visualize the mapping from rationale to concepts"""
    # Get top concepts
    top_indices = np.argsort(concept_scores)[-top_k:]
    top_scores = concept_scores[top_indices]

    # Create concept names
    concept_names = [f'Concept {i}' for i in top_indices]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot concept scores
    plt.barh(range(len(top_scores)), top_scores, color='salmon')

    # Add concept labels
    plt.yticks(range(len(concept_names)), concept_names)

    # Add title and labels
    plt.title('Rationale to Concept Mapping', fontsize=14)
    plt.xlabel('Concept Activation Score')
    plt.ylabel('Concepts')
    plt.xlim(0, 1)

    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_concept_to_prediction(logits, class_names, save_path=None):
    """Visualize the mapping from concepts to prediction"""
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot class probabilities
    plt.bar(range(len(probs)), probs, color='lightgreen')

    # Add class labels
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')

    # Add title and labels
    plt.title('Concept to Prediction Mapping', fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_text_with_rationale(text, rationale_mask, tokenizer, threshold=0.2, save_path=None):
    """Visualize text with highlighted rationale"""
    # Tokenize text
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']

    # Create a character-level mask
    char_mask = np.zeros(len(text))

    # Map token-level rationale to character-level
    for i, (start, end) in enumerate(offset_mapping):
        if i < len(rationale_mask):
            char_mask[start:end] = rationale_mask[i]

    # Create a word-level mask
    words = text.split()
    word_starts = []
    current_pos = 0
    for word in words:
        # Find the start position of the word
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
        word_starts.append(current_pos)
        current_pos += len(word)

    # Calculate word-level rationale scores
    word_scores = []
    for i, start in enumerate(word_starts):
        end = start + len(words[i])
        word_scores.append(char_mask[start:end].mean())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Create a binary mask for highlighting
    mask = [score > threshold for score in word_scores]

    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray

    # Plot text
    for i, (word, is_rationale) in enumerate(zip(words, mask)):
        color = highlight_color if is_rationale else normal_color
        ax.text(i, 0, word,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=10)

    # Hide axes but keep frame
    ax.set_xlim(-1, len(words))
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    plt.title(f"Text with Highlighted Rationale (threshold={threshold})", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_summary_visualization(example, outputs, tokenizer, class_names, attribute,
                                rationale_threshold, concept_threshold, save_path=None):
    """Create a summary visualization of the model's predictions"""
    # Extract data
    text = example.get('description', example.get('text', ''))

    # The attribute in CEBaB is in the format 'attribute_aspect_majority'
    attribute_key = f"{attribute}_aspect_majority"
    true_rating = example.get(attribute_key, "N/A")

    rationale_mask = outputs['rationale_mask'][0].cpu().numpy()
    concept_scores = outputs['concept_scores'][0].cpu().numpy()
    logits = outputs['logits'][0]

    # Get prediction
    pred = torch.argmax(logits).item()
    pred_rating = class_names[pred]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # 1. Text with highlighted rationale
    ax1 = fig.add_subplot(gs[0])

    # Tokenize text
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']

    # Create a character-level mask
    char_mask = np.zeros(len(text))

    # Map token-level rationale to character-level
    for i, (start, end) in enumerate(offset_mapping):
        if i < len(rationale_mask):
            char_mask[start:end] = rationale_mask[i]

    # Create a word-level mask
    words = text.split()
    word_starts = []
    current_pos = 0
    for word in words:
        # Find the start position of the word
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
        word_starts.append(current_pos)
        current_pos += len(word)

    # Calculate word-level rationale scores
    word_scores = []
    for i, start in enumerate(word_starts):
        end = start + len(words[i])
        word_scores.append(char_mask[start:end].mean())

    # Create a binary mask for highlighting
    mask = [score > rationale_threshold for score in word_scores]

    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray

    # Plot text
    for i, (word, is_rationale) in enumerate(zip(words, mask)):
        color = highlight_color if is_rationale else normal_color
        ax1.text(i % 20, i // 20, word,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=10)

    # Hide axes but keep frame
    ax1.set_xlim(-1, 20)
    ax1.set_ylim(-0.5, (len(words) // 20) + 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add title
    ax1.set_title(f"Text with Highlighted Rationale (threshold={rationale_threshold})", fontsize=14)

    # 2. Rationale to concept mapping
    ax2 = fig.add_subplot(gs[1])

    # Get top concepts
    top_indices = np.argsort(concept_scores)[-10:]
    top_scores = concept_scores[top_indices]

    # Create concept names
    concept_names = [f'Concept {i}' for i in top_indices]

    # Plot concept scores
    ax2.barh(range(len(top_scores)), top_scores, color='salmon')

    # Add concept labels
    ax2.set_yticks(range(len(concept_names)))
    ax2.set_yticklabels(concept_names)

    # Add title and labels
    ax2.set_title('Rationale to Concept Mapping', fontsize=14)
    ax2.set_xlabel('Concept Activation Score')
    ax2.set_ylabel('Concepts')
    ax2.set_xlim(0, 1)

    # Add grid
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    # 3. Concept to prediction mapping
    ax3 = fig.add_subplot(gs[2])

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    # Plot class probabilities
    ax3.bar(range(len(probs)), probs, color='lightgreen')

    # Add class labels
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right')

    # Add title and labels
    ax3.set_title('Concept to Prediction Mapping', fontsize=14)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)

    # Add grid
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # Add overall title
    plt.suptitle(f"Attribute: {attribute.capitalize()}, True Rating: {true_rating}, Predicted: {pred_rating}",
                fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return save_path

def main():
    """Main function for visualization"""
    args = parse_arguments()

    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.model_dir, f"visualizations_{timestamp}_{args.attribute}")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load CEBaB dataset
    cebab = load_cebab_dataset()

    # Get examples for the specified attribute
    examples = get_attribute_examples(cebab, args.attribute, args.num_examples)

    # Define class names for CEBaB
    class_names = ["Negative", "Positive"]

    # Create model
    # The model has 50 concepts based on the config
    model = RationaleConceptModel(num_concepts=50, num_classes=2)

    # Load model weights
    model_path = os.path.join(args.model_dir, args.model_filename)
    print(f"Loading model from {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with randomly initialized weights for demonstration")

    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Process examples
    print(f"\nProcessing {len(examples)} examples for attribute '{args.attribute}'...")

    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        # Get text (handle different possible keys)
        text = example.get('description', example.get('text', ''))

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])

        # Create directory for this example
        example_dir = os.path.join(output_dir, f"example_{i+1}")
        os.makedirs(example_dir, exist_ok=True)

        # Visualize input to rationale mapping
        input_to_rationale_path = os.path.join(example_dir, "input_to_rationale.png")
        visualize_input_to_rationale(
            text,
            outputs['rationale_mask'][0].cpu().numpy(),
            tokenizer,
            input_to_rationale_path
        )

        # Visualize rationale to concept mapping
        rationale_to_concept_path = os.path.join(example_dir, "rationale_to_concept.png")
        visualize_rationale_to_concept(
            outputs['concept_scores'][0].cpu().numpy(),
            top_k=10,
            save_path=rationale_to_concept_path
        )

        # Visualize concept to prediction mapping
        concept_to_prediction_path = os.path.join(example_dir, "concept_to_prediction.png")
        visualize_concept_to_prediction(
            outputs['logits'][0],
            class_names,
            concept_to_prediction_path
        )

        # Visualize text with rationale
        text_with_rationale_path = os.path.join(example_dir, "text_with_rationale.png")
        visualize_text_with_rationale(
            text,
            outputs['rationale_mask'][0].cpu().numpy(),
            tokenizer,
            threshold=args.rationale_threshold,
            save_path=text_with_rationale_path
        )

        # Create summary visualization
        summary_path = os.path.join(example_dir, "summary.png")
        create_summary_visualization(
            example,
            outputs,
            tokenizer,
            class_names,
            args.attribute,
            args.rationale_threshold,
            args.concept_threshold,
            summary_path
        )

        # Save example data
        attribute_key = f"{args.attribute}_aspect_majority"
        example_data = {
            'text': text,
            'true_rating': example.get(attribute_key, "N/A"),
            'predicted_rating': torch.argmax(outputs['logits'][0]).item() + 1,
            'rationale_mask': outputs['rationale_mask'][0].cpu().numpy().tolist(),
            'concept_scores': outputs['concept_scores'][0].cpu().numpy().tolist(),
            'logits': outputs['logits'][0].cpu().numpy().tolist()
        }

        example_json_path = os.path.join(example_dir, "data.json")
        with open(example_json_path, 'w') as f:
            json.dump(example_data, f, indent=2, cls=NumpyEncoder)

    # Create index.html
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(f"<html><head><title>CEBaB Visualizations - {args.attribute}</title></head><body>\n")
        f.write(f"<h1>CEBaB Visualizations - {args.attribute}</h1>\n")

        for i in range(len(examples)):
            f.write(f"<h2>Example {i+1}</h2>\n")
            text = examples[i].get('description', examples[i].get('text', ''))
            f.write(f"<p>Text: {text[:100]}...</p>\n")

            attribute_key = f"{args.attribute}_aspect_majority"
            true_rating = examples[i].get(attribute_key, "N/A")
            f.write(f"<p>True Rating for {args.attribute}: {true_rating}</p>\n")

            f.write(f"<h3>Summary</h3>\n")
            f.write(f"<img src='example_{i+1}/summary.png' width='800'><br>\n")
            f.write(f"<h3>Input to Rationale</h3>\n")
            f.write(f"<img src='example_{i+1}/input_to_rationale.png' width='800'><br>\n")
            f.write(f"<h3>Text with Rationale</h3>\n")
            f.write(f"<img src='example_{i+1}/text_with_rationale.png' width='800'><br>\n")
            f.write(f"<h3>Rationale to Concept</h3>\n")
            f.write(f"<img src='example_{i+1}/rationale_to_concept.png' width='800'><br>\n")
            f.write(f"<h3>Concept to Prediction</h3>\n")
            f.write(f"<img src='example_{i+1}/concept_to_prediction.png' width='800'><br>\n")
            f.write("<hr>\n")

        f.write("</body></html>\n")

    print(f"\nVisualization complete. Results saved to {output_dir}")
    print(f"Open {index_path} in a browser to view the visualizations")

    # Try to open the index.html
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(index_path)}")
    except:
        pass

if __name__ == "__main__":
    main()
