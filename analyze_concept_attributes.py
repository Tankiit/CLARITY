"""
Script to analyze concept activations across different attributes in the CEBaB dataset.
This script creates a heatmap showing how concepts are activated for different attributes.
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
import matplotlib.colors as mcolors

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
    parser = argparse.ArgumentParser(description="Analyze concept activations across attributes")
    
    # Model checkpoint
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing the model files")
    
    # Model filename
    parser.add_argument("--model_filename", type=str, default="checkpoints/best_model.pt",
                       help="Filename of the model checkpoint")
    
    # Config filename
    parser.add_argument("--config_filename", type=str, default="config.json",
                       help="Filename of the model configuration")
    
    # Attributes to analyze
    parser.add_argument("--attributes", type=str, nargs="+",
                       default=["food", "service", "ambiance", "noise"],
                       help="Attributes to analyze")
    
    # Number of examples
    parser.add_argument("--num_examples", type=int, default=50,
                       help="Number of examples to analyze per attribute")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations (default: model_dir/concept_attributes_timestamp)")
    
    # Number of top concepts to analyze
    parser.add_argument("--num_concepts", type=int, default=10,
                       help="Number of top concepts to analyze")
    
    # Concept threshold
    parser.add_argument("--concept_threshold", type=float, default=0.2,
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

def get_attribute_examples(dataset, attribute, num_examples=50):
    """Get examples for a specific attribute"""
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
        print("No examples found with meaningful ratings for the specified attribute.")
        return []
    
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
    
    return examples[:num_examples]

def analyze_concept_attributes(model, tokenizer, dataset, attributes, output_dir, 
                              num_examples=50, num_concepts=10, concept_threshold=0.2):
    """Analyze concept activations across different attributes"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process examples for each attribute
    attribute_concept_scores = {}
    
    for attribute in attributes:
        print(f"\nAnalyzing attribute: {attribute}")
        
        # Get examples for this attribute
        examples = get_attribute_examples(dataset, attribute, num_examples)
        
        if not examples:
            print(f"Skipping attribute {attribute} due to lack of examples")
            continue
        
        # Process examples
        all_concept_scores = []
        
        for i, example in enumerate(tqdm(examples, desc=f"Processing {attribute} examples")):
            # Get text
            text = example.get('description', example.get('text', ''))
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Extract concept scores
            concept_scores = outputs['concept_scores'][0].cpu().numpy()
            all_concept_scores.append(concept_scores)
        
        # Calculate average concept scores for this attribute
        if all_concept_scores:
            attribute_concept_scores[attribute] = np.mean(all_concept_scores, axis=0)
    
    # Find top concepts across all attributes
    all_scores = np.vstack([scores for scores in attribute_concept_scores.values()])
    mean_scores = np.mean(all_scores, axis=0)
    top_concepts = np.argsort(mean_scores)[-num_concepts:][::-1]
    
    # Create a DataFrame for the heatmap
    data = []
    for concept_idx in top_concepts:
        row = {'Concept': f'concept_{concept_idx}'}
        num_attributes = 0
        for attribute in attributes:
            if attribute in attribute_concept_scores:
                score = attribute_concept_scores[attribute][concept_idx]
                if score > concept_threshold:
                    row[attribute.capitalize()] = f"{score:.4f}"
                    num_attributes += 1
                else:
                    row[attribute.capitalize()] = "-"
        row['Attributes'] = num_attributes
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    
    # Convert string values to float for heatmap
    heatmap_data = df.copy()
    for attribute in attributes:
        attribute_cap = attribute.capitalize()
        if attribute_cap in heatmap_data.columns:
            heatmap_data[attribute_cap] = heatmap_data[attribute_cap].apply(
                lambda x: float(x) if x != "-" else np.nan
            )
    
    # Create a custom colormap that handles NaN values
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data[['Food', 'Service', 'Ambiance', 'Noise']],
        annot=df[['Food', 'Service', 'Ambiance', 'Noise']],
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Concept Activation Score'},
        mask=heatmap_data[['Food', 'Service', 'Ambiance', 'Noise']].isna()
    )
    
    # Add row labels
    ax.set_yticklabels(df['Concept'], rotation=0)
    
    # Add attribute count
    for i, count in enumerate(df['Attributes']):
        plt.text(4.5, i + 0.5, str(count), ha='center', va='center')
    
    # Add column for attribute count
    plt.text(4.5, -0.5, 'Attributes', ha='center', va='center', fontweight='bold')
    
    plt.title('Concept Activation Across Attributes')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'concept_attribute_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the data as CSV
    csv_path = os.path.join(output_dir, 'concept_attribute_scores.csv')
    df.to_csv(csv_path, index=False)
    
    # Save the data as JSON
    json_path = os.path.join(output_dir, 'concept_attribute_scores.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Heatmap: {heatmap_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    
    return heatmap_path

def main():
    args = parse_arguments()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.model_dir, f"concept_attributes_{timestamp}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CEBaB dataset
    cebab = load_cebab_dataset()
    
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
    
    # Analyze concept activations across attributes
    analyze_concept_attributes(
        model, 
        tokenizer, 
        cebab, 
        args.attributes, 
        output_dir,
        args.num_examples,
        args.num_concepts,
        args.concept_threshold
    )

if __name__ == "__main__":
    import time
    main()
