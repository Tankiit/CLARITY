"""
Script to create LaTeX-friendly visualizations from the CEBaB model results.
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import torch
from PIL import Image
from transformers import AutoTokenizer
import seaborn as sns

# Set up matplotlib for LaTeX-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LaTeX-friendly visualizations")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the model visualizations")
    parser.add_argument("--output_dir", type=str, default="latex_visualizations",
                       help="Directory to save the LaTeX visualizations")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of examples to visualize")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                       help="Name of the model used")
    parser.add_argument("--attribute", type=str, default="food",
                       help="Attribute being analyzed")
    
    return parser.parse_args()

def load_example_data(example_dir):
    """Load the example data from the JSON file"""
    data_path = os.path.join(example_dir, "data.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_text_with_rationale(text, rationale_mask, tokenizer, threshold=0.2, save_path=None):
    """Create a LaTeX-friendly visualization of text with highlighted rationale"""
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
    fig, ax = plt.subplots(figsize=(8, 1.5))
    
    # Create a binary mask for highlighting
    mask = [score > threshold for score in word_scores]
    
    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray
    
    # Calculate how many words per line
    max_words_per_line = 15
    num_lines = (len(words) + max_words_per_line - 1) // max_words_per_line
    
    # Adjust figure height based on number of lines
    fig.set_figheight(0.5 + 0.5 * num_lines)
    
    # Plot text
    for i, (word, is_rationale) in enumerate(zip(words, mask)):
        line_idx = i // max_words_per_line
        word_idx = i % max_words_per_line
        
        color = highlight_color if is_rationale else normal_color
        ax.text(word_idx, -line_idx, word, 
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.3', edgecolor='none'),
                ha='center', va='center', fontsize=9)
    
    # Hide axes but keep frame
    ax.set_xlim(-1, max_words_per_line)
    ax.set_ylim(-num_lines + 0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add title
    ax.set_title(f"Text with Highlighted Rationale (threshold={threshold})", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return save_path

def visualize_rationale_to_concept(concept_scores, top_k=10, save_path=None):
    """Create a LaTeX-friendly visualization of rationale to concept mapping"""
    # Get top concepts
    top_indices = np.argsort(concept_scores)[-top_k:][::-1]  # Reverse to get descending order
    top_scores = concept_scores[top_indices]
    
    # Create concept names
    concept_names = [f'Concept {i}' for i in top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot concept scores
    bars = ax.barh(range(len(top_scores)), top_scores, color='#4C72B0', alpha=0.8)
    
    # Add concept labels
    ax.set_yticks(range(len(concept_names)))
    ax.set_yticklabels(concept_names)
    
    # Add title and labels
    ax.set_title('Top Concepts Activated by Rationale', fontsize=11)
    ax.set_xlabel('Activation Score')
    ax.set_ylabel('Concepts')
    ax.set_xlim(0, 1)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return save_path

def visualize_concept_to_prediction(logits, class_names, save_path=None):
    """Create a LaTeX-friendly visualization of concept to prediction mapping"""
    # Apply softmax to get probabilities
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot class probabilities
    bars = ax.bar(range(len(probs)), probs, color='#55A868', alpha=0.8)
    
    # Add class labels
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=0)
    
    # Add title and labels
    ax.set_title('Prediction Probabilities', fontsize=11)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return save_path

def create_combined_visualization(example_data, tokenizer, class_names, save_path=None):
    """Create a combined visualization for a single example"""
    # Extract data
    text = example_data['text']
    true_rating = example_data['true_rating']
    predicted_rating = example_data['predicted_rating']
    rationale_mask = np.array(example_data['rationale_mask'])
    concept_scores = np.array(example_data['concept_scores'])
    logits = example_data['logits']
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1.5])
    
    # 1. Text with highlighted rationale
    ax1 = fig.add_subplot(gs[0])
    
    # Calculate word-level rationale scores
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
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
    threshold = 0.2
    mask = [score > threshold for score in word_scores]
    
    # Create text visualization with highlighted rationale
    highlight_color = "#ffcccc"  # Light red
    normal_color = "#f2f2f2"    # Light gray
    
    # Calculate how many words per line
    max_words_per_line = 15
    num_lines = (len(words) + max_words_per_line - 1) // max_words_per_line
    
    # Plot text
    for i, (word, is_rationale) in enumerate(zip(words, mask)):
        line_idx = i // max_words_per_line
        word_idx = i % max_words_per_line
        
        color = highlight_color if is_rationale else normal_color
        ax1.text(word_idx, -line_idx, word, 
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.3', edgecolor='none'),
                ha='center', va='center', fontsize=9)
    
    # Hide axes but keep frame
    ax1.set_xlim(-1, max_words_per_line)
    ax1.set_ylim(-num_lines + 0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Add title
    ax1.set_title(f"Text with Highlighted Rationale", fontsize=11)
    
    # 2. Rationale to concept mapping
    ax2 = fig.add_subplot(gs[1])
    
    # Get top concepts
    top_k = 8
    top_indices = np.argsort(concept_scores)[-top_k:][::-1]  # Reverse to get descending order
    top_scores = concept_scores[top_indices]
    
    # Create concept names
    concept_names = [f'Concept {i}' for i in top_indices]
    
    # Plot concept scores
    bars = ax2.barh(range(len(top_scores)), top_scores, color='#4C72B0', alpha=0.8)
    
    # Add concept labels
    ax2.set_yticks(range(len(concept_names)))
    ax2.set_yticklabels(concept_names)
    
    # Add title and labels
    ax2.set_title('Top Concepts Activated by Rationale', fontsize=11)
    ax2.set_xlabel('Activation Score')
    ax2.set_ylabel('Concepts')
    ax2.set_xlim(0, 1)
    
    # Add grid
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 3. Concept to prediction mapping
    ax3 = fig.add_subplot(gs[2])
    
    # Apply softmax to get probabilities
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    
    # Plot class probabilities
    bars = ax3.bar(range(len(probs)), probs, color='#55A868', alpha=0.8)
    
    # Add class labels
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=0)
    
    # Add title and labels
    ax3.set_title('Prediction Probabilities', fontsize=11)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    
    # Add grid
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add overall title
    pred_class = class_names[np.argmax(probs)]
    plt.suptitle(f"True Rating: {true_rating}, Predicted: {pred_class}", fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return save_path

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Define class names for CEBaB
    class_names = ["Negative", "Positive"]
    
    # Find example directories
    example_dirs = []
    for i in range(1, args.num_examples + 1):
        example_dir = os.path.join(args.input_dir, f"example_{i}")
        if os.path.exists(example_dir):
            example_dirs.append(example_dir)
    
    # Process each example
    for i, example_dir in enumerate(example_dirs):
        print(f"Processing example {i+1}...")
        
        # Load example data
        example_data = load_example_data(example_dir)
        
        # Create output directory for this example
        example_output_dir = os.path.join(args.output_dir, f"example_{i+1}")
        os.makedirs(example_output_dir, exist_ok=True)
        
        # Create individual visualizations
        visualize_text_with_rationale(
            example_data['text'],
            np.array(example_data['rationale_mask']),
            tokenizer,
            threshold=0.2,
            save_path=os.path.join(example_output_dir, "rationale.pdf")
        )
        
        visualize_rationale_to_concept(
            np.array(example_data['concept_scores']),
            top_k=8,
            save_path=os.path.join(example_output_dir, "concepts.pdf")
        )
        
        visualize_concept_to_prediction(
            example_data['logits'],
            class_names,
            save_path=os.path.join(example_output_dir, "prediction.pdf")
        )
        
        # Create combined visualization
        create_combined_visualization(
            example_data,
            tokenizer,
            class_names,
            save_path=os.path.join(example_output_dir, "combined.pdf")
        )
    
    # Create a LaTeX file with the visualizations
    latex_path = os.path.join(args.output_dir, "cebab_visualizations.tex")
    with open(latex_path, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage{caption}\n")
        f.write("\\usepackage{subcaption}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n\n")
        
        f.write(f"\\title{{CEBaB Model Visualizations for {args.attribute.capitalize()} Attribute}}\n")
        f.write("\\author{Generated by Visualization Script}\n")
        f.write("\\date{\\today}\n")
        f.write("\\maketitle\n\n")
        
        f.write("\\section{Introduction}\n")
        f.write("This document presents visualizations of the rationale-concept bottleneck model ")
        f.write(f"applied to the CEBaB dataset, focusing on the {args.attribute} attribute. ")
        f.write("The visualizations show how the model extracts rationales from the input text, ")
        f.write("maps these rationales to concepts, and then uses the concepts to make predictions.\n\n")
        
        # Add examples
        for i in range(1, len(example_dirs) + 1):
            f.write(f"\\section{{Example {i}}}\n")
            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.9\\textwidth]{{example_{i}/combined.pdf}}\n")
            f.write(f"\\caption{{Combined visualization for Example {i}}}\n")
            f.write("\\end{figure}\n\n")
            
            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{subfigure}{0.9\\textwidth}\n")
            f.write(f"\\includegraphics[width=\\textwidth]{{example_{i}/rationale.pdf}}\n")
            f.write("\\caption{Text with highlighted rationale}\n")
            f.write("\\end{subfigure}\n\n")
            
            f.write("\\begin{subfigure}{0.45\\textwidth}\n")
            f.write(f"\\includegraphics[width=\\textwidth]{{example_{i}/concepts.pdf}}\n")
            f.write("\\caption{Top concepts activated by rationale}\n")
            f.write("\\end{subfigure}\n")
            f.write("\\hfill\n")
            f.write("\\begin{subfigure}{0.45\\textwidth}\n")
            f.write(f"\\includegraphics[width=\\textwidth]{{example_{i}/prediction.pdf}}\n")
            f.write("\\caption{Prediction probabilities}\n")
            f.write("\\end{subfigure}\n")
            f.write(f"\\caption{{Detailed visualizations for Example {i}}}\n")
            f.write("\\end{figure}\n\n")
        
        f.write("\\end{document}\n")
    
    print(f"LaTeX file created: {latex_path}")
    print(f"To compile: pdflatex {latex_path}")
    
    # Create a README file
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# CEBaB Model Visualizations for {args.attribute.capitalize()} Attribute\n\n")
        f.write("This directory contains LaTeX-friendly visualizations of the rationale-concept bottleneck model ")
        f.write(f"applied to the CEBaB dataset, focusing on the {args.attribute} attribute.\n\n")
        
        f.write("## Files\n\n")
        f.write("- `cebab_visualizations.tex`: LaTeX file that includes all visualizations\n")
        f.write("- `example_X/`: Directory containing visualizations for example X\n")
        f.write("  - `combined.pdf`: Combined visualization for the example\n")
        f.write("  - `rationale.pdf`: Text with highlighted rationale\n")
        f.write("  - `concepts.pdf`: Top concepts activated by rationale\n")
        f.write("  - `prediction.pdf`: Prediction probabilities\n\n")
        
        f.write("## Compilation\n\n")
        f.write("To compile the LaTeX file into a PDF, run:\n\n")
        f.write("```bash\n")
        f.write(f"pdflatex {os.path.basename(latex_path)}\n")
        f.write("```\n")
    
    print(f"README file created: {readme_path}")

if __name__ == "__main__":
    main()
