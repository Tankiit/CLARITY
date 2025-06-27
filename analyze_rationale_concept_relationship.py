"""
Script to analyze the relationship between rationales and concepts in the CEBaB model.
This script also demonstrates how concept intervention can be used to manipulate predictions.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    parser = argparse.ArgumentParser(description="Analyze rationale-concept relationship")

    # Model checkpoint
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing the model files")

    # Model filename
    parser.add_argument("--model_filename", type=str, default="checkpoints/best_model.pt",
                       help="Filename of the model checkpoint")

    # Config filename
    parser.add_argument("--config_filename", type=str, default="config.json",
                       help="Filename of the model configuration")

    # Attribute to analyze
    parser.add_argument("--attribute", type=str, default="food",
                       choices=["food", "ambiance", "service", "noise", "price"],
                       help="Attribute to analyze")

    # Number of examples
    parser.add_argument("--num_examples", type=int, default=50,
                       help="Number of examples to analyze")

    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations (default: model_dir/concept_analysis_timestamp)")

    # Rationale threshold
    parser.add_argument("--rationale_threshold", type=float, default=0.2,
                       help="Threshold for rationale extraction")

    # Concept threshold
    parser.add_argument("--concept_threshold", type=float, default=0.1,
                       help="Threshold for concept activation")

    # Number of concepts to analyze
    parser.add_argument("--num_concepts", type=int, default=10,
                       help="Number of top concepts to analyze")

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

    def forward(self, input_ids, attention_mask, intervene_concepts=None):
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

        # Apply concept intervention if provided
        if intervene_concepts is not None:
            concept_scores = intervene_concepts

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

def get_attribute_examples(dataset, attribute, num_examples=50):
    """Get examples for a specific attribute"""
    # Print dataset structure for debugging
    print(f"Dataset structure: {list(dataset.keys())}")
    print(f"Test set size: {len(dataset['test'])}")

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
    print(f"Selected {len(examples)} examples")

    return examples[:num_examples]

def analyze_rationale_concept_relationship(model, tokenizer, examples, attribute, output_dir,
                                          rationale_threshold=0.2, concept_threshold=0.1, num_concepts=10):
    """Analyze the relationship between rationales and concepts"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process examples
    all_rationales = []
    all_concepts = []
    all_predictions = []
    all_true_ratings = []
    all_texts = []

    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        # Get text
        text = example.get('description', example.get('text', ''))
        all_texts.append(text)

        # Get true rating
        attribute_key = f"{attribute}_aspect_majority"
        true_rating = example.get(attribute_key, "N/A")
        all_true_ratings.append(true_rating)

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])

        # Extract rationale
        rationale_mask = outputs['rationale_mask'][0].cpu().numpy()
        all_rationales.append(rationale_mask)

        # Extract concepts
        concept_scores = outputs['concept_scores'][0].cpu().numpy()
        all_concepts.append(concept_scores)

        # Extract prediction
        logits = outputs['logits'][0]
        pred = torch.argmax(logits).item()
        all_predictions.append(pred)

    # Convert to numpy arrays
    # Handle variable-length rationales by padding or truncating
    max_length = max(len(r) for r in all_rationales)
    padded_rationales = []
    for r in all_rationales:
        if len(r) < max_length:
            # Pad with zeros
            padded_rationales.append(np.pad(r, (0, max_length - len(r)), 'constant'))
        else:
            # Truncate
            padded_rationales.append(r[:max_length])

    all_rationales = np.array(padded_rationales)
    all_concepts = np.array(all_concepts)
    all_predictions = np.array(all_predictions)

    # 1. Analyze concept activation patterns
    concept_activation_counts = (all_concepts > concept_threshold).sum(axis=0)
    top_concepts = np.argsort(concept_activation_counts)[-num_concepts:][::-1]

    # Visualize top concept activation counts
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_concepts), concept_activation_counts[top_concepts], color='skyblue')
    plt.xticks(range(num_concepts), [f'C{i}' for i in top_concepts], rotation=45)
    plt.title(f'Top {num_concepts} Concept Activation Counts')
    plt.xlabel('Concept')
    plt.ylabel('Number of Examples')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_activation_counts.png'), dpi=300)
    plt.close()

    # 2. Analyze concept co-occurrence
    concept_binary = (all_concepts > concept_threshold).astype(int)
    concept_cooccurrence = np.dot(concept_binary.T, concept_binary)

    # Visualize concept co-occurrence for top concepts
    plt.figure(figsize=(10, 8))
    sns.heatmap(concept_cooccurrence[top_concepts][:, top_concepts],
                annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=[f'C{i}' for i in top_concepts],
                yticklabels=[f'C{i}' for i in top_concepts])
    plt.title('Concept Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_cooccurrence.png'), dpi=300)
    plt.close()

    # 3. Analyze rationale-concept correlation
    # Flatten rationales and compute correlation with each concept
    flat_rationales = all_rationales.reshape(all_rationales.shape[0], -1)

    # Compute correlation between rationales and concepts
    rationale_concept_corr = np.zeros((model.num_concepts, flat_rationales.shape[1]))
    for i in range(model.num_concepts):
        for j in range(flat_rationales.shape[1]):
            if np.std(flat_rationales[:, j]) > 0:
                rationale_concept_corr[i, j] = np.corrcoef(all_concepts[:, i], flat_rationales[:, j])[0, 1]

    # Visualize rationale-concept correlation for top concepts
    plt.figure(figsize=(12, 8))
    for i, concept_idx in enumerate(top_concepts[:5]):  # Show top 5 concepts
        plt.plot(rationale_concept_corr[concept_idx], label=f'Concept {concept_idx}')
    plt.title('Rationale-Concept Correlation')
    plt.xlabel('Token Position')
    plt.ylabel('Correlation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rationale_concept_correlation.png'), dpi=300)
    plt.close()

    # 4. Perform concept intervention
    # For each top concept, intervene by setting it to 1 and others to 0
    intervention_results = []

    for concept_idx in top_concepts[:5]:  # Intervene on top 5 concepts
        intervention_predictions = []

        for i, example in enumerate(tqdm(examples[:10], desc=f"Intervening on concept {concept_idx}")):
            # Get text
            text = example.get('description', example.get('text', ''))

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Create intervention tensor
            intervene_concepts = torch.zeros((1, model.num_concepts), device=device)
            intervene_concepts[0, concept_idx] = 1.0

            # Forward pass with intervention
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'], intervene_concepts=intervene_concepts)

            # Extract prediction
            logits = outputs['logits'][0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            intervention_predictions.append(probs)

        intervention_results.append(np.array(intervention_predictions))

    # Visualize intervention results
    plt.figure(figsize=(12, 8))

    # For each example, show how intervention affects prediction
    for i in range(min(5, len(examples))):
        plt.subplot(1, 5, i+1)

        # Get original prediction
        original_pred = all_predictions[i]

        # Create bar data
        concept_labels = [f'C{idx}' for idx in top_concepts[:5]]
        positive_probs = [intervention_results[j][i, 1] for j in range(5)]

        # Plot bars
        bars = plt.bar(concept_labels, positive_probs, color='lightgreen')

        # Add original prediction line
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')

        # Add labels
        plt.title(f'Example {i+1}')
        plt.ylim(0, 1)
        if i == 0:
            plt.ylabel('Probability of Positive Class')
        if i == 2:
            plt.xlabel('Intervened Concept')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Effect of Concept Intervention on Prediction', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'concept_intervention.png'), dpi=300)
    plt.close()

    # 5. Visualize concept space using t-SNE
    # Reduce dimensionality of concept vectors
    # Use a smaller perplexity for small sample sizes
    perplexity = min(5, len(all_concepts) - 1)  # Perplexity must be less than n_samples
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    concept_tsne = tsne.fit_transform(all_concepts)

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': concept_tsne[:, 0],
        'y': concept_tsne[:, 1],
        'prediction': all_predictions,
        'true_rating': all_true_ratings
    })

    # Plot t-SNE visualization colored by prediction
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='prediction', palette='viridis', s=100)
    plt.title('t-SNE Visualization of Concept Space (Colored by Prediction)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Prediction', labels=['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_tsne_prediction.png'), dpi=300)
    plt.close()

    # Plot t-SNE visualization colored by true rating
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='true_rating', palette='Set2', s=100)
    plt.title('t-SNE Visualization of Concept Space (Colored by True Rating)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='True Rating')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_tsne_rating.png'), dpi=300)
    plt.close()

    # 6. Create a summary visualization
    plt.figure(figsize=(15, 10))

    # Plot concept activation counts
    plt.subplot(2, 2, 1)
    plt.bar(range(num_concepts), concept_activation_counts[top_concepts], color='skyblue')
    plt.xticks(range(num_concepts), [f'C{i}' for i in top_concepts], rotation=45)
    plt.title('Top Concept Activation Counts')
    plt.xlabel('Concept')
    plt.ylabel('Number of Examples')

    # Plot concept co-occurrence
    plt.subplot(2, 2, 2)
    sns.heatmap(concept_cooccurrence[top_concepts[:5]][:, top_concepts[:5]],
                annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=[f'C{i}' for i in top_concepts[:5]],
                yticklabels=[f'C{i}' for i in top_concepts[:5]])
    plt.title('Concept Co-occurrence Matrix')

    # Plot t-SNE visualization
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='x', y='y', hue='prediction', palette='viridis', s=50)
    plt.title('Concept Space (by Prediction)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Prediction', labels=['Negative', 'Positive'])

    # Plot intervention results for one example
    plt.subplot(2, 2, 4)
    concept_labels = [f'C{idx}' for idx in top_concepts[:5]]
    positive_probs = [intervention_results[j][0, 1] for j in range(5)]
    bars = plt.bar(concept_labels, positive_probs, color='lightgreen')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
    plt.title('Concept Intervention (Example 1)')
    plt.ylim(0, 1)
    plt.ylabel('Probability of Positive Class')
    plt.xlabel('Intervened Concept')

    plt.suptitle(f'Rationale-Concept Analysis for {attribute.capitalize()} Attribute', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=300)
    plt.close()

    # Save analysis results
    results = {
        'top_concepts': top_concepts.tolist(),
        'concept_activation_counts': concept_activation_counts.tolist(),
        'concept_cooccurrence': concept_cooccurrence.tolist(),
        'examples': [
            {
                'text': text,
                'true_rating': rating,
                'prediction': pred,
                'top_concept_scores': [float(all_concepts[i, c]) for c in top_concepts[:5]]
            }
            for i, (text, rating, pred) in enumerate(zip(all_texts[:10], all_true_ratings[:10], all_predictions[:10]))
        ]
    }

    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Analysis complete. Results saved to {output_dir}")

def main():
    args = parse_arguments()

    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.model_dir, f"concept_analysis_{timestamp}_{args.attribute}")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load CEBaB dataset
    cebab = load_cebab_dataset()

    # Get examples for the specified attribute
    examples = get_attribute_examples(cebab, args.attribute, args.num_examples)

    # Create model
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

    # Analyze rationale-concept relationship
    analyze_rationale_concept_relationship(
        model,
        tokenizer,
        examples,
        args.attribute,
        output_dir,
        args.rationale_threshold,
        args.concept_threshold,
        args.num_concepts
    )

if __name__ == "__main__":
    import time
    main()
