#!/usr/bin/env python

"""
CEBAB Experiments with Diverse Dropout Credal CBM
Trains with diverse dropout then evaluates uncertainty correlation
"""

import os
import sys
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from credal_cbm_model import CredalCBM, CredalCBMConfig, train_credal_cbm_with_diverse_dropout
from exp1_uncertainty_correlation import UncertaintyCorrelationExperiment
from run_cebab_experiments import load_cebab_data


def run_cebab_diverse_dropout_experiment(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_samples: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = './results_cebab_diverse_dropout'
):
    """
    Run CEBAB experiment with diverse dropout training + uncertainty evaluation

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_samples: Maximum samples for inference analysis
        device: Device to run on
        save_dir: Directory to save results
    """

    print("="*80)
    print("CEBAB DIVERSE DROPOUT CREDAL CBM EXPERIMENT")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Max samples for analysis: {max_samples}")
    print(f"Save directory: {save_dir}")
    print("="*80)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load CEBAB data
    print("\nLoading CEBAB dataset...")
    train_loader, val_loader = load_cebab_data(tokenizer, batch_size=batch_size)

    # Create model configuration
    config = CredalCBMConfig(
        base_model_name="distilbert-base-uncased",
        num_concepts=15,  # CEBAB has multi-aspect reviews
        num_classes=2,
        n_ensemble_heads=5,
        use_credal=True,
        use_rationales=False,
        hidden_dim=256,
        dropout=0.3,
        concept_loss_weight=0.5,
        classification_loss_weight=1.0
    )

    print("\nModel configuration:")
    print(f"  Base model: {config.base_model_name}")
    print(f"  Num concepts: {config.num_concepts}")
    print(f"  Ensemble heads: {config.n_ensemble_heads}")
    print(f"  Use credal sets: {config.use_credal}")
    print(f"  Training method: Diverse Dropout")

    # Initialize model
    print("\nInitializing Credal CBM...")
    model = CredalCBM(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # === PHASE 1: DIVERSE DROPOUT TRAINING ===
    print("\n" + "="*80)
    print("PHASE 1: DIVERSE DROPOUT TRAINING")
    print("="*80)

    trained_model = train_credal_cbm_with_diverse_dropout(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )

    # === PHASE 2: UNCERTAINTY EVALUATION ===
    print("\n" + "="*80)
    print("PHASE 2: UNCERTAINTY CORRELATION EVALUATION")
    print("="*80)

    # Initialize experiment tracker
    experiment = UncertaintyCorrelationExperiment(
        model_class=lambda: trained_model,
        tokenizer=tokenizer,
        device=device
    )

    # Run inference and analysis
    experiment.run_inference(trained_model, val_loader, max_samples=max_samples)

    print("\nComputing correlations...")
    corr_results = experiment.compute_correlations()

    print(f"\nAccuracy: {corr_results['accuracy']:.3f}")
    print(f"\nEpistemic correlation: ρ = {corr_results['epistemic']['spearman_rho']:.3f}, "
          f"p = {corr_results['epistemic']['spearman_p']:.4f}")
    print(f"Aleatoric correlation: ρ = {corr_results['aleatoric']['spearman_rho']:.3f}, "
          f"p = {corr_results['aleatoric']['spearman_p']:.4f}")

    print("\nAnalyzing uncertainty by correctness...")
    correctness_analysis = experiment.analyze_uncertainty_by_correctness()

    print(f"\nCorrect predictions (n={correctness_analysis['correct']['count']}):")
    print(f"  Epistemic: {correctness_analysis['correct']['epistemic_mean']:.4f} ± "
          f"{correctness_analysis['correct']['epistemic_std']:.4f}")
    print(f"  Aleatoric: {correctness_analysis['correct']['aleatoric_mean']:.4f} ± "
          f"{correctness_analysis['correct']['aleatoric_std']:.4f}")

    print(f"\nIncorrect predictions (n={correctness_analysis['incorrect']['count']}):")
    print(f"  Epistemic: {correctness_analysis['incorrect']['epistemic_mean']:.4f} ± "
          f"{correctness_analysis['incorrect']['epistemic_std']:.4f}")
    print(f"  Aleatoric: {correctness_analysis['incorrect']['aleatoric_mean']:.4f} ± "
          f"{correctness_analysis['incorrect']['aleatoric_std']:.4f}")

    print("\nCreating visualizations...")
    experiment.create_visualizations(save_dir)

    print("\nGenerating LaTeX table...")
    experiment.generate_table(corr_results, save_dir)

    # Save results
    import json
    results_serializable = {
        'training_method': 'diverse_dropout',
        'correlations': corr_results,
        'correctness_analysis': correctness_analysis
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'exp1_diverse_dropout_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, 'credal_cbm_cebab_diverse_dropout.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_method': 'diverse_dropout',
        'results': results_serializable
    }, checkpoint_path)

    print(f"\n{'='*80}")
    print("DIVERSE DROPOUT EXPERIMENT COMPLETE")
    print(f"Results saved to: {save_dir}")
    print("="*80)

    return model, experiment, corr_results, correctness_analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run CEBAB Diverse Dropout Credal CBM experiments')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples for analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./results_cebab_diverse_dropout', help='Save directory')

    args = parser.parse_args()

    run_cebab_diverse_dropout_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        device=args.device,
        save_dir=args.save_dir
    )
