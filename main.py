"""
Example script for training and evaluating the optimized rationale-concept bottleneck model
on the AG News dataset with visualization of results.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import set_seed, AutoTokenizer

# Import the implementation
from optimized_rationale_concept_model import (
    ModelConfig,
    RationaleConceptBottleneckModel,
    load_and_process_dataset,
    MetricsTracker,
    train_model,
    evaluate_model,
    device
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train rationale-concept model on AG News")
    
    # Key parameters that can be adjusted
    parser.add_argument("--small", action="store_true",
                       help="Use smaller configuration for faster training")
    parser.add_argument("--fast", action="store_true",
                       help="Use LoRA and other optimizations for faster training")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations for sample predictions")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--inference_only", action="store_true",
                       help="Skip training and only run inference from a saved model")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model for inference only mode")
    parser.add_argument("--causal_discovery", action="store_true",
                       help="Run causal discovery analysis on concept activations")
    
    return parser.parse_args()

def create_config(args):
    """Create model configuration based on arguments"""
    
    # Base configuration
    config = ModelConfig(
        base_model_name="distilbert-base-uncased",
        num_labels=4,  # AG News has 4 classes
        num_concepts=50,
        batch_size=32,
        max_seq_length=128,
        learning_rate=2e-5,
        base_model_lr=1e-5,
        num_epochs=50,
        seed=args.seed,
        output_dir="agnews_models"
    )
    
    # Small configuration for quick testing
    if args.small:
        config.num_concepts = 20
        config.batch_size = 16
        config.max_seq_length = 64
        config.num_epochs = 2
    
    # Fast training configuration
    if args.fast:
        config.use_lora = True
        config.lora_r = 8
        config.enable_concept_interactions = False  # Simpler model
    
    return config

def visualize_explanation(text, explanation, save_path=None):
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
        ax2.set_title(f"Top Concepts for Prediction: Class {explanation['prediction']}", fontsize=14)
        
    else:
        ax2.text(0.5, 0.5, "No active concepts found", 
                 ha='center', va='center', fontsize=12)
    
    # Add prediction information
    fig.text(0.5, 0.01, 
             f"Prediction: Class {explanation['prediction']} (Confidence: {explanation['confidence']:.2f})",
             ha='center', fontsize=12, bbox=dict(facecolor='#ddfcdd', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_causal_discovery(model, tokenizer, dataset, config, output_dir):
    """Run causal discovery on concept activations"""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.GraphUtils import GraphUtils
        from causallearn.utils.cit import fisherz
    except ImportError:
        print("causallearn package not found. Skipping causal discovery.")
        print("Install with: pip install causal-learn")
        return None

    # Setup causal discovery directory
    causal_dir = os.path.join(output_dir, "causal_discovery")
    os.makedirs(causal_dir, exist_ok=True)

    # Prepare DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Collect concept activations
    concept_activations = []
    model.eval()
    
    print("Collecting concept activations for causal discovery...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            outputs = model(**inputs, output_concepts=True)
            concept_activations.append(outputs.concept_scores.cpu().numpy())
    
    # Create concept matrix (n_samples x n_concepts)
    concept_matrix = np.concatenate(concept_activations, axis=0)
    print(f"Collected concept activations matrix: {concept_matrix.shape}")
    
    # Run PC algorithm for causal discovery
    print("Running PC algorithm for causal discovery...")
    try:
        cg = pc(
            concept_matrix, 
            alpha=0.05, 
            indep_test=fisherz, 
            stable=True,
            uc_rule=0,
            uc_priority=0
        )
        
        # Save causal graph
        graph_path = os.path.join(causal_dir, "causal_graph.png")
        try:
            pyd = GraphUtils.to_pydot(cg.G)
            pyd.write_png(graph_path)
            print(f"Causal graph saved to: {graph_path}")
        except Exception as e:
            print(f"Could not save causal graph visualization: {e}")
        
        # Save concept activations and adjacency matrix
        np.save(os.path.join(causal_dir, "concept_activations.npy"), concept_matrix)
        np.save(os.path.join(causal_dir, "adjacency_matrix.npy"), cg.G.graph)
        
        # Save summary statistics
        with open(os.path.join(causal_dir, "causal_summary.txt"), "w") as f:
            f.write(f"Causal Discovery Results\n")
            f.write(f"========================\n\n")
            f.write(f"Number of concepts: {concept_matrix.shape[1]}\n")
            f.write(f"Number of samples: {concept_matrix.shape[0]}\n")
            f.write(f"Number of edges: {np.sum(cg.G.graph != 0)}\n")
            f.write(f"Alpha level: 0.05\n")
            f.write(f"Independence test: Fisher's Z\n\n")
            
            # Find strongly connected concepts
            edge_count = np.sum(cg.G.graph != 0, axis=1)
            top_concepts = np.argsort(edge_count)[-5:]
            f.write(f"Top 5 most connected concepts:\n")
            for i, concept_idx in enumerate(top_concepts):
                f.write(f"  Concept {concept_idx}: {edge_count[concept_idx]} connections\n")
        
        print(f"Causal discovery results saved to: {causal_dir}")
        return cg
        
    except Exception as e:
        print(f"Error during causal discovery: {e}")
        return None

def visualize_concept_intervention(model, tokenizer, text, concept_idx=0, new_values=[0.0, 0.5, 1.0], save_path=None):
    """
    Visualize the effect of intervening on a specific concept
    
    This function demonstrates how modifying specific concepts affects
    the model's predictions, helping understand what each concept represents.
    """
    # Create figure
    fig, axs = plt.subplots(len(new_values) + 1, 1, figsize=(12, 4 * (len(new_values) + 1)))
    
    # AG News class names for reference
    class_names = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }
    
    # Get original explanation
    original_explanation = model.explain_prediction(tokenizer, text)
    
    # Plot original prediction
    ax = axs[0]
    original_probs = original_explanation['class_probabilities']
    original_pred = original_explanation['prediction']
    
    # Bar plot of original prediction
    bars = ax.bar(range(len(original_probs)), original_probs, color='skyblue')
    bars[original_pred].set_color('navy')
    
    # Add class names
    ax.set_xticks(range(len(original_probs)))
    ax.set_xticklabels([class_names.get(i, f"Class {i}") for i in range(len(original_probs))])
    ax.set_ylabel('Probability')
    ax.set_title(f"Original Prediction: {class_names.get(original_pred, f'Class {original_pred}')} "
                 f"(Concept {concept_idx} = {original_explanation['top_concepts'][0][1]:.2f})")
    
    # For each new concept value
    for i, new_value in enumerate(new_values):
        # Perform intervention
        intervention = model.intervene_on_concepts(tokenizer, text, concept_idx, new_value)
        
        # Plot intervention results
        ax = axs[i + 1]
        new_probs = intervention['intervened_probs']
        new_pred = intervention['intervened_prediction']
        
        # Bar plot of intervention prediction
        bars = ax.bar(range(len(new_probs)), new_probs, color='lightgreen')
        bars[new_pred].set_color('darkgreen')
        
        # Add class names
        ax.set_xticks(range(len(new_probs)))
        ax.set_xticklabels([class_names.get(i, f"Class {i}") for i in range(len(new_probs))])
        ax.set_ylabel('Probability')
        ax.set_title(f"Concept {concept_idx} = {new_value:.1f} → Prediction: {class_names.get(new_pred, f'Class {new_pred}')}")
    
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return original_explanation, [model.intervene_on_concepts(tokenizer, text, concept_idx, v) for v in new_values]

def main():
    """Main function for training and evaluating the model"""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create model configuration
    config = create_config(args)
    
    # Print basic information
    print(f"Training rationale-concept bottleneck model on AG News")
    print(f"Using device: {device}")
    print(f"Model: {config.base_model_name}")
    print(f"Number of concepts: {config.num_concepts}")
    print(f"Using LoRA: {config.use_lora}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    
    # Initialize output directory for visualizations
    timestamp = time.strftime("%Y%m%d-%H%M%S") if not args.inference_only else "inference"
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model for inference only or train new model
    if args.inference_only and args.model_path:
        print(f"\nLoading model from {args.model_path} for inference...")
        model = RationaleConceptBottleneckModel(config)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        # Load and preprocess dataset
        datasets = load_and_process_dataset("ag_news", tokenizer, config)
        
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(config)
        
        # Create model
        model = RationaleConceptBottleneckModel(config)
        
        # Train model
        print("\nStarting training...")
        model, best_model_path = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets['train'],
            val_dataset=datasets['validation'],
            test_dataset=datasets['test'],
            config=config,
            metrics_tracker=metrics_tracker
        )
        
        # Print results
        print(f"\nTraining complete! Best model saved to: {best_model_path}")
        print(f"All outputs saved to: {metrics_tracker.output_dir}")
        
        # Update output directory for visualizations
        output_dir = metrics_tracker.output_dir
    
    # Generate explanations and visualizations
    if args.visualize or args.inference_only:
        # Create visualization directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # AG News class names
        class_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
        # Sample AG News examples for demonstration
        examples = [
            "Oil prices fell more than $1 a barrel on Friday on concerns that Hurricane Ivan will miss most oil and gas production in the Gulf of Mexico.",
            "Yankees catcher Jorge Posada was suspended for three games and Angels shortstop David Eckstein was penalized one game for their actions in a game this week.",
            "Microsoft Corp. is accelerating the schedule for its next Windows operating system update, code-named Longhorn, and scaling back some features to meet its new timetable.",
            "Scientists in the United States say they have developed a new type of drone aircraft that flies by flapping its wings rather than using an engine."
        ]
        
        # Generate and visualize explanations
        print("\nGenerating explanations and visualizations...")
        for i, text in enumerate(examples):
            # Generate explanation
            explanation = model.explain_prediction(tokenizer, text)
            
            # Add class name to explanation
            predicted_class = explanation['prediction']
            explanation['class_name'] = class_names.get(predicted_class, f"Class {predicted_class}")
            
            # Visualize explanation
            save_path = os.path.join(viz_dir, f"example_{i+1}_explanation.png")
            visualize_explanation(text, explanation, save_path)
            print(f"  Explanation {i+1}: {explanation['class_name']} (saved to {save_path})")
            
            # Visualize concept intervention (for the first example only)
            if i == 0:
                # Intervene on the most important concept
                top_concept_idx = 0
                intervention_path = os.path.join(viz_dir, f"example_{i+1}_intervention.png")
                visualize_concept_intervention(
                    model, tokenizer, text, 
                    concept_idx=top_concept_idx, 
                    new_values=[0.0, 0.5, 1.0],
                    save_path=intervention_path
                )
                print(f"  Concept intervention visualization saved to: {intervention_path}")
        
        print(f"\nAll visualizations saved to: {viz_dir}")
    
    # Run causal discovery if requested
    if args.causal_discovery:
        print("\nRunning causal discovery analysis...")
        if not args.inference_only:
            # Use validation dataset for causal discovery
            causal_graph = run_causal_discovery(
                model=model,
                tokenizer=tokenizer, 
                dataset=datasets['validation'],
                config=config,
                output_dir=output_dir
            )
            if causal_graph is not None:
                print("Causal discovery completed successfully!")
            else:
                print("Causal discovery failed or was skipped.")
        else:
            print("Causal discovery requires a dataset. Please run without --inference_only flag.")
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    import time
    main()