"""
Visualization script for legacy rationale-concept bottleneck models.

This script loads a trained model directly and generates visualizations of explanations
without trying to adapt it to the current architecture.
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
    parser = argparse.ArgumentParser(description="Visualize explanations from a legacy model")
    
    # Model checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to saved model configuration")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["yelp_polarity", "dbpedia", "sst2", "agnews"],
                       help="Dataset to use for examples")
    
    # Custom examples
    parser.add_argument("--custom_examples", type=str, default=None,
                       help="Path to JSON file with custom examples to visualize")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    
    # Number of examples
    parser.add_argument("--num_examples", type=int, default=4,
                       help="Number of examples to visualize")
    
    return parser.parse_args()

def load_custom_examples(file_path):
    """Load custom examples from a JSON file"""
    with open(file_path, 'r') as f:
        examples = json.load(f)
    
    # Check format
    if isinstance(examples, list):
        # If it's a list of strings, use as is
        if all(isinstance(ex, str) for ex in examples):
            return examples
        # If it's a list of objects, extract the 'text' field
        elif all(isinstance(ex, dict) and 'text' in ex for ex in examples):
            return [ex['text'] for ex in examples]
    
    raise ValueError("Custom examples file must contain a list of strings or objects with 'text' field")

def get_dataset_examples(dataset_name):
    """Get class names and example texts for a dataset"""
    if dataset_name == "yelp_polarity":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        examples = [
            "This restaurant was terrible. The food was cold and the service was slow.",
            "I absolutely loved this place! The staff was friendly and the food was amazing.",
            "Worst experience ever. Would not recommend to anyone.",
            "Great atmosphere and delicious food. Will definitely come back!"
        ]
    elif dataset_name == "dbpedia":
        class_names = {
            0: "Company", 1: "Educational Institution", 2: "Artist",
            3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
            6: "Building", 7: "Natural Place", 8: "Village",
            9: "Animal", 10: "Plant", 11: "Album",
            12: "Film", 13: "Written Work"
        }
        examples = [
            "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
            "Harvard University is a private Ivy League research university in Cambridge, Massachusetts.",
            "Leonardo da Vinci was an Italian polymath of the Renaissance whose areas of interest included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, literature, anatomy, geology, astronomy, botany, paleontology, and cartography.",
            "Michael Jordan is an American former professional basketball player and businessman."
        ]
    elif dataset_name == "sst2":
        class_names = {
            0: "Negative",
            1: "Positive"
        }
        examples = [
            "The movie was a complete waste of time with terrible acting and a predictable plot.",
            "This film is a masterpiece of storytelling with amazing performances and stunning visuals.",
            "I couldn't stand the slow pacing and convoluted storyline of this boring movie.",
            "The director's vision shines through in every scene, creating a truly memorable cinematic experience."
        ]
    elif dataset_name == "agnews":
        class_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        examples = [
            "Oil prices fell more than $1 a barrel on Friday on concerns that Hurricane Ivan will miss most oil and gas production in the Gulf of Mexico.",
            "Yankees catcher Jorge Posada was suspended for three games and Angels shortstop David Eckstein was penalized one game for their actions in a game this week.",
            "Microsoft Corp. is accelerating the schedule for its next Windows operating system update, code-named Longhorn, and scaling back some features to meet its new timetable.",
            "Scientists in the United States say they have developed a new type of drone aircraft that flies by flapping its wings rather than using an engine."
        ]
    else:
        class_names = {}
        examples = []
    
    return class_names, examples

def create_explanations_summary(explanations, save_path):
    """Create a summary visualization of multiple explanations"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    num_examples = min(len(explanations), 3)  # Show up to 3 examples
    gs = plt.GridSpec(num_examples, 2, figure=fig)
    
    # Process each explanation
    for i, ex in enumerate(explanations[:num_examples]):
        text = ex['text']
        explanation = ex['explanation']
        class_name = ex['class_name']
        
        # 1. Visualize rationale for this example
        ax1 = fig.add_subplot(gs[i, 0])
        
        # Extract rationale tokens
        words = text.split()
        rationale_words = explanation['rationale'].split()
        
        # Create a binary mask for highlighting
        mask = []
        for word in words:
            if any(r.lower().startswith(word.lower()) for r in rationale_words):
                mask.append(1)
            else:
                mask.append(0)
        
        # Create text visualization with highlighted rationale
        highlight_color = "#ffcccc"  # Light red
        normal_color = "#f2f2f2"    # Light gray
        
        # Plot text
        for j, (word, is_rationale) in enumerate(zip(words, mask)):
            color = highlight_color if is_rationale else normal_color
            ax1.text(j, 0, word, 
                    bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                    ha='center', va='center', fontsize=8)
        
        # Hide axes but keep frame
        ax1.set_xlim(-1, len(words))
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"Example {i+1}: {class_name} (Rationale: {explanation['rationale_percentage']:.1%} of text)", fontsize=12)
        
        # 2. Visualize top concepts for this example
        ax2 = fig.add_subplot(gs[i, 1])
        
        # Get top concepts
        concepts = explanation["top_concepts"]
        if concepts:
            concept_names = [c[0] for c in concepts]
            concept_scores = [c[1] for c in concepts]
            
            # Sort by score
            sorted_idx = np.argsort(concept_scores)
            concept_names = [concept_names[i] for i in sorted_idx]
            concept_scores = [concept_scores[i] for i in sorted_idx]
            
            # Create horizontal bar chart
            ax2.barh(concept_names, concept_scores, color='skyblue')
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Concept Score')
            ax2.set_title(f"Top Concepts for {class_name}", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "No active concepts found", 
                     ha='center', va='center', fontsize=12)
    
    # Add overall title
    plt.suptitle("Explanation Summary", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_explanation(text, explanation, class_names, save_path=None):
    """Visualize model explanation for a prediction"""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Visualize rationale
    rationale = explanation["rationale"]
    rationale_pct = explanation["rationale_percentage"]
    
    # Extract rationale tokens
    words = text.split()
    rationale_words = rationale.split()
    
    # Create a binary mask for highlighting
    mask = []
    for word in words:
        if any(r.lower().startswith(word.lower()) for r in rationale_words):
            mask.append(1)
        else:
            mask.append(0)
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Word': words,
        'Is_Rationale': mask
    })
    
    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray
    
    # Plot text
    for i, row in df.iterrows():
        word = row['Word']
        color = highlight_color if row['Is_Rationale'] == 1 else normal_color
        ax1.text(i, 0, word, 
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=10)
    
    # Hide axes but keep frame
    ax1.set_xlim(-1, len(words))
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Extracted Rationale ({rationale_pct:.1%} of text)", fontsize=14)
    
    # 2. Visualize top concepts
    concepts = explanation["top_concepts"]
    if concepts:
        concept_names = [c[0] for c in concepts]
        concept_scores = [c[1] for c in concepts]
        
        # Sort by score
        sorted_idx = np.argsort(concept_scores)
        concept_names = [concept_names[i] for i in sorted_idx]
        concept_scores = [concept_scores[i] for i in sorted_idx]
        
        # Create horizontal bar chart
        ax2.barh(concept_names, concept_scores, color='skyblue')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Concept Score')
        ax2.set_title(f"Top Concepts for Prediction: {class_names.get(explanation['prediction'], f'Class {explanation['prediction']}')}", fontsize=14)
        
    else:
        ax2.text(0.5, 0.5, "No active concepts found", 
                 ha='center', va='center', fontsize=12)
    
    # Add prediction information
    fig.text(0.5, 0.01, 
             f"Prediction: {class_names.get(explanation['prediction'], f'Class {explanation['prediction']}')} (Confidence: {explanation['confidence']:.2f})",
             ha='center', fontsize=12, bbox=dict(facecolor='#ddfcdd', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """Main function for visualization"""
    args = parse_arguments()
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset info
    class_names, default_examples = get_dataset_examples(args.dataset)
    
    # Load examples
    if args.custom_examples:
        print(f"Loading custom examples from {args.custom_examples}")
        examples = load_custom_examples(args.custom_examples)
    else:
        print(f"Using default examples for {args.dataset}")
        examples = default_examples[:args.num_examples]
    
    # Load model directly
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Generate explanations
    print(f"\nGenerating explanations for {len(examples)} examples...")
    explanations = []
    
    for i, text in enumerate(tqdm(examples, desc="Generating explanations")):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'], dataset=args.dataset)
            
            # Extract prediction
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred].item()
            
            # Extract rationale
            rationale_mask = outputs['rationale_mask'][0]
            attention_mask = inputs['attention_mask'][0]
            
            # Calculate rationale percentage
            valid_tokens = attention_mask.sum().item()
            rationale_tokens = (rationale_mask * attention_mask).sum().item()
            rationale_pct = rationale_tokens / valid_tokens if valid_tokens > 0 else 0
            
            # Get rationale text
            rationale_indices = torch.where((rationale_mask * attention_mask) > 0)[0]
            rationale_tokens = inputs['input_ids'][0][rationale_indices]
            rationale_text = tokenizer.decode(rationale_tokens)
            
            # Get concept scores
            concept_scores = outputs['concept_scores'][0].cpu().numpy()
            
            # Create explanation
            explanation = {
                'prediction': pred,
                'confidence': confidence,
                'rationale': rationale_text,
                'rationale_percentage': rationale_pct,
                'top_concepts': [('Concept ' + str(i), float(score)) for i, score in enumerate(concept_scores) if score > 0.1][:5]
            }
            
            explanations.append({
                'text': text,
                'explanation': explanation,
                'class_name': class_names.get(pred, f'Class {pred}')
            })
    
    # Create visualizations
    print(f"\nCreating visualizations in {output_dir}...")
    
    # Generate individual visualizations
    for i, ex in enumerate(explanations):
        text = ex['text']
        explanation = ex['explanation']
        
        # Visualize explanation
        save_path = os.path.join(output_dir, f"example_{i+1}_explanation.png")
        visualize_explanation(text, explanation, class_names, save_path)
        predicted_class = explanation['prediction']
        print(f"  Explanation {i+1}: {class_names.get(predicted_class, f'Class {predicted_class}')} (saved to {save_path})")
    
    # Save explanations to JSON file
    explanations_file = os.path.join(output_dir, "explanations.json")
    with open(explanations_file, 'w') as f:
        json.dump(explanations, f, indent=2, cls=NumpyEncoder)
    print(f"  Explanations saved to: {explanations_file}")
    
    # Create a summary visualization
    summary_path = os.path.join(output_dir, "explanations_summary.png")
    create_explanations_summary(explanations, summary_path)
    print(f"  Summary visualization saved to: {summary_path}")
    
    print(f"\nVisualization complete. Results saved to {output_dir}")
    
    # Try to open the summary visualization
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(summary_path)}")
    except:
        pass

if __name__ == "__main__":
    main()
