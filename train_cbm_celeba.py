"""
Train Boolean CBM on CelebA Dataset
====================================

This script demonstrates how to train the minimal Boolean CBM on CelebA data.
It shows:
1. Loading CelebA with concept/task split
2. Creating the CBM model
3. Adding domain-specific constraints
4. Training with concept supervision
5. Evaluating with explanations

USAGE:
    python train_cbm_celeba.py --concept_set appearance --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from typing import Dict, List, Optional

# Import CBM components (from the provided code)
from celeba_cbm_dataset import get_celeba_loaders, create_celeba_concept_sets
from minimal_boolean_cbm import (
    SimpleBooleanCBM, TrainConfig, train_loop, inference_step,
    explain_prediction, create_simple_encoder, CausalDiscovery
)


def add_celeba_constraints(model: SimpleBooleanCBM, concept_set: str = 'appearance'):
    """
    Add domain-specific constraints for CelebA attributes.

    These are hand-crafted constraints based on common sense about facial attributes.
    In v2, these will be discovered automatically.
    """

    constraint_layer = model.constraints

    if concept_set == 'appearance':
        # Basic appearance constraints

        # Hair color constraints (mutually exclusive)
        constraint_layer.add_mutex('Black_Hair', 'Blond_Hair')
        constraint_layer.add_mutex('Black_Hair', 'Brown_Hair')
        constraint_layer.add_mutex('Blond_Hair', 'Brown_Hair')

        # Gender constraints
        constraint_layer.add_implies('Male', 'No_Beard')  # Most males have no beard
        constraint_layer.add_mutex('Heavy_Makeup', 'Male')  # Heavy makeup more common in females

        # Age constraints
        constraint_layer.add_implies('Young', 'No_Beard')  # Young people less likely to have beards
        constraint_layer.add_implies('Young', 'Heavy_Makeup')  # Young people more likely to wear makeup

    elif concept_set == 'facial_features':
        # Facial feature constraints

        # Attractiveness associations
        constraint_layer.add_implies('High_Cheekbones', 'Attractive')
        constraint_layer.add_implies('Oval_Face', 'Attractive')

        # Weight-related constraints
        constraint_layer.add_implies('Chubby', 'Double_Chin')
        constraint_layer.add_mutex('Narrow_Eyes', 'Big_Lips')  # Unlikely combination

    elif concept_set == 'accessories':
        # Accessory constraints

        # Makeup constraints
        constraint_layer.add_implies('Heavy_Makeup', 'Wearing_Lipstick')

        # Gender-specific accessories
        constraint_layer.add_mutex('Eyeglasses', 'Wearing_Necktie')
        constraint_layer.add_implies('Wearing_Necktie', 'Male')

    elif concept_set == 'facial_hair':
        # Facial hair constraints

        # Facial hair is mostly male
        constraint_layer.add_implies('5_o_Clock_Shadow', 'Male')
        constraint_layer.add_implies('Goatee', 'Male')
        constraint_layer.add_implies('Mustache', 'Male')

        # Beard relationships
        constraint_layer.add_mutex('No_Beard', '5_o_Clock_Shadow')
        constraint_layer.add_mutex('No_Beard', 'Goatee')
        constraint_layer.add_mutex('No_Beard', 'Mustache')

    elif concept_set == 'age_prediction':
        # Age prediction constraints

        # Age-related appearance
        constraint_layer.add_implies('Young', 'Attractive')  # Youth bias in attractiveness
        constraint_layer.add_implies('Young', 'Heavy_Makeup')
        constraint_layer.add_mutex('Gray_Hair', 'Young')
        constraint_layer.add_implies('Receding_Hairline', 'Male')


def train_cbm_celeba(
    concept_set: str = 'appearance',
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_constraint: float = 0.5,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    use_cuda: bool = True
) -> Dict:
    """
    Train a Boolean CBM on CelebA dataset.

    RETURNS:
        Dictionary with training history and model
    """

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, cbm_config = get_celeba_loaders(
        concept_set=concept_set,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        input_size=64,
        root='./data'
    )

    encoder = create_simple_encoder(
        input_channels=cbm_config['input_channels'],
        num_concepts=cbm_config['num_concepts']
    )

    model = SimpleBooleanCBM(
        concept_names=cbm_config['concept_names'],
        num_classes=cbm_config['num_classes'],
        encoder=encoder,
        hidden_dim=128
    )

    model = model.to(device)

    add_celeba_constraints(model, concept_set)

    config = TrainConfig(
        lr=lr,
        lambda_concept=1.0,
        lambda_constraint=lambda_constraint,
        lambda_thresh_reg=0.01
    )

    history, best_state = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        config=config
    )

    return {
        'model': model,
        'history': history,
        'config': cbm_config,
        'device': device
    }


def test_concept_sets(epochs: int = 10, batch_size: int = 16):
    """
    Test training on different concept sets to compare performance.
    """

    results = {}

    for concept_set in create_celeba_concept_sets().keys():
        try:
            result = train_cbm_celeba(
                concept_set=concept_set,
                epochs=epochs,
                batch_size=batch_size,
                train_size=500,
                val_size=200,
                lambda_constraint=0.3
            )

            best_acc = max([h['acc'] for h in result['history']['val']])
            results[concept_set] = best_acc

        except Exception as e:
            results[concept_set] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Boolean CBM on CelebA')
    parser.add_argument('--concept_set', type=str, default='appearance',
                       choices=list(create_celeba_concept_sets().keys()),
                       help='Which concept set to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_constraint', type=float, default=0.5,
                       help='Constraint loss weight')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Limit training set size (for debugging)')
    parser.add_argument('--val_size', type=int, default=None,
                       help='Limit validation set size (for debugging)')
    parser.add_argument('--test_all', action='store_true',
                       help='Test all concept sets with quick training')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()

    if args.test_all:
        # Quick test of all concept sets
        test_concept_sets(epochs=5, batch_size=16)
    else:
        # Full training on specified concept set
        result = train_cbm_celeba(
            concept_set=args.concept_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_constraint=args.lambda_constraint,
            train_size=args.train_size,
            val_size=args.val_size,
            use_cuda=not args.no_cuda
        )


if __name__ == "__main__":
    main()