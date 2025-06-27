"""
Simple script to visualize examples with their predicted sentiment.

This script doesn't try to load the model, but just creates visualizations
of example texts with their predicted sentiment.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import time
import random
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize example texts with sentiment")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["sst2", "yelp_polarity", "agnews", "dbpedia"],
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

def get_dataset_examples(dataset_name):
    """Get class names and example texts for a dataset"""
    if dataset_name == "sst2":
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
    elif dataset_name == "yelp_polarity":
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
    else:
        class_names = {}
        examples = []
    
    return class_names, examples

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

def visualize_text_with_rationale(text, label, class_name, rationale_words=None, save_path=None):
    """Visualize text with highlighted rationale"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Split text into words
    words = text.split()
    
    # If no rationale is provided, generate a random one
    if rationale_words is None:
        # Randomly select 20-30% of words as rationale
        num_rationale = max(1, int(len(words) * random.uniform(0.2, 0.3)))
        rationale_indices = sorted(random.sample(range(len(words)), num_rationale))
        rationale_words = [words[i] for i in rationale_indices]
    
    # Create a binary mask for highlighting
    mask = []
    for word in words:
        if any(r.lower() == word.lower() for r in rationale_words):
            mask.append(1)
        else:
            mask.append(0)
    
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
    
    # Add title with class
    plt.title(f"Class: {class_name}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_summary_visualization(examples, class_names, save_path):
    """Create a summary visualization of multiple examples"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    num_examples = min(len(examples), 4)  # Show up to 4 examples
    gs = plt.GridSpec(num_examples, 1, figure=fig)
    
    # Process each example
    for i, example in enumerate(examples[:num_examples]):
        text = example['text']
        label = example['label']
        class_name = class_names.get(label, f'Class {label}')
        rationale = example.get('rationale', None)
        
        # Create subplot
        ax = fig.add_subplot(gs[i])
        
        # Split text into words
        words = text.split()
        
        # If no rationale is provided, generate a random one
        if rationale is None:
            # Randomly select 20-30% of words as rationale
            num_rationale = max(1, int(len(words) * random.uniform(0.2, 0.3)))
            rationale_indices = sorted(random.sample(range(len(words)), num_rationale))
            rationale = [words[i] for i in rationale_indices]
        
        # Create a binary mask for highlighting
        mask = []
        for word in words:
            if any(r.lower() == word.lower() for r in rationale):
                mask.append(1)
            else:
                mask.append(0)
        
        # Create text visualization with highlighted rationale
        highlight_color = "#ffcccc"  # Light red
        normal_color = "#f2f2f2"    # Light gray
        
        # Plot text
        for j, (word, is_rationale) in enumerate(zip(words, mask)):
            color = highlight_color if is_rationale else normal_color
            ax.text(j, 0, word, 
                    bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                    ha='center', va='center', fontsize=8)
        
        # Hide axes but keep frame
        ax.set_xlim(-1, len(words))
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title with class
        ax.set_title(f"Example {i+1}: {class_name}", fontsize=12)
    
    # Add overall title
    plt.suptitle("Example Texts with Predicted Classes", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

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
        example_texts = load_custom_examples(args.custom_examples)
    else:
        print(f"Using default examples for {args.dataset}")
        example_texts = default_examples[:args.num_examples]
    
    # Create example objects with predicted labels
    examples = []
    for i, text in enumerate(example_texts):
        # For demonstration, assign labels based on index
        # In a real scenario, these would come from the model
        if args.dataset in ["sst2", "yelp_polarity"]:
            label = i % 2  # Alternate between 0 and 1
        elif args.dataset == "agnews":
            label = i % 4  # Alternate between 0, 1, 2, 3
        elif args.dataset == "dbpedia":
            label = i % 14  # Alternate between 0-13
        
        # Create example object
        example = {
            'text': text,
            'label': label
        }
        
        # Generate random rationale
        words = text.split()
        num_rationale = max(1, int(len(words) * random.uniform(0.2, 0.3)))
        rationale_indices = sorted(random.sample(range(len(words)), num_rationale))
        example['rationale'] = [words[i] for i in rationale_indices]
        
        examples.append(example)
    
    # Create visualizations
    print(f"\nCreating visualizations in {output_dir}...")
    
    # Generate individual visualizations
    for i, example in enumerate(examples):
        text = example['text']
        label = example['label']
        class_name = class_names.get(label, f'Class {label}')
        rationale = example['rationale']
        
        # Visualize example
        save_path = os.path.join(output_dir, f"example_{i+1}.png")
        visualize_text_with_rationale(text, label, class_name, rationale, save_path)
        print(f"  Example {i+1}: {class_name} (saved to {save_path})")
    
    # Save examples to JSON file
    examples_file = os.path.join(output_dir, "examples.json")
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"  Examples saved to: {examples_file}")
    
    # Create a summary visualization
    summary_path = os.path.join(output_dir, "examples_summary.png")
    create_summary_visualization(examples, class_names, summary_path)
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
