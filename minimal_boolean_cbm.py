"""
Minimal Boolean CBM - Start Simple, Iterate
============================================

ASSUMPTIONS:
- Concept labels ARE provided (supervised)
- Single pathway (not multi-evidence yet)
- Constraints will come from causal discovery (placeholder for now)

ITERATION PLAN:
v1 (this): Basic training loop with learned thresholds
v2: Add causal discovery for constraints
v3: Add multi-evidence (spatial/appearance/semantic)
v4: Add shortcut detection + validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


# ============================================================================
# PART 1: LEARNED THRESHOLDS (Simple Version)
# ============================================================================

class LearnedThresholds(nn.Module):
    """
    Per-concept learnable thresholds.

    WHY NOT FIXED 0.5?
    - Rare concepts need lower threshold (catch more positives)
    - Noisy concepts need higher threshold (reduce false positives)
    - Learned from data, frozen at inference
    """

    def __init__(self, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts

        # Learnable thresholds (initialize at 0.5)
        self.thresholds = nn.Parameter(torch.full((num_concepts,), 0.5))

        self._frozen = False
        self.temperature = 1.0  # For soft→hard annealing

    def freeze(self):
        self._frozen = True
        self.thresholds.requires_grad_(False)
        print(f"Thresholds frozen: {self.thresholds.cpu().data.numpy().round(3)}")

    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_probs: (batch, num_concepts) in [0, 1]
        Returns:
            binary_concepts: (batch, num_concepts)
        """
        # Clamp thresholds to valid range
        thresholds = torch.clamp(self.thresholds, 0.01, 0.99)

        diff = concept_probs - thresholds.unsqueeze(0)

        if self.training and not self._frozen:
            # Straight-Through Estimator
            soft = torch.sigmoid(diff * 10 / self.temperature)  # Sharper sigmoid
            hard = (diff > 0).float()
            return hard - soft.detach() + soft
        else:
            return (diff > 0).float()


# ============================================================================
# PART 2: CONSTRAINT LAYER (Differentiable)
# ============================================================================

class ConstraintLayer(nn.Module):
    """
    Differentiable Boolean constraints.

    SUPPORTED:
    - implies(A, B): If A then B
    - mutex(A, B): Not both A and B
    - atleast_one([A, B, C]): At least one must be true

    TO ADD (v2): Causal discovery will populate these automatically
    """

    def __init__(self, concept_names: List[str]):
        super().__init__()
        self.concept_names = concept_names
        self.name_to_idx = {n: i for i, n in enumerate(concept_names)}

        # Constraint storage
        self.implies: List[Tuple[int, int]] = []
        self.mutex: List[Tuple[int, int]] = []
        self.atleast_one: List[List[int]] = []

    def add_implies(self, a: str, b: str):
        """A → B"""
        if a in self.name_to_idx and b in self.name_to_idx:
            self.implies.append((self.name_to_idx[a], self.name_to_idx[b]))
            print(f"  Added: {a} → {b}")

    def add_mutex(self, a: str, b: str):
        """¬(A ∧ B)"""
        if a in self.name_to_idx and b in self.name_to_idx:
            self.mutex.append((self.name_to_idx[a], self.name_to_idx[b]))
            print(f"  Added: ¬({a} ∧ {b})")

    def add_atleast_one(self, concepts: List[str]):
        """A ∨ B ∨ C ..."""
        indices = [self.name_to_idx[c] for c in concepts if c in self.name_to_idx]
        if indices:
            self.atleast_one.append(indices)
            print(f"  Added: atleast_one({concepts})")

    def violation_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Compute total constraint violation.

        Args:
            concepts: (batch, num_concepts) in [0, 1]
        Returns:
            scalar loss (0 = all satisfied)
        """
        loss = torch.tensor(0.0, device=concepts.device)

        # Implies: A → B  ≡  satisfaction = min(1, 1 - A + B)
        for a_idx, b_idx in self.implies:
            a, b = concepts[:, a_idx], concepts[:, b_idx]
            violation = F.relu(a - b)  # Violated when A > B
            loss = loss + violation.mean()

        # Mutex: ¬(A ∧ B)  ≡  satisfaction = 1 - A*B
        for a_idx, b_idx in self.mutex:
            a, b = concepts[:, a_idx], concepts[:, b_idx]
            violation = a * b  # Violated when both high
            loss = loss + violation.mean()

        # AtLeastOne: A ∨ B ∨ ...  ≡  satisfaction = max(A, B, ...)
        for indices in self.atleast_one:
            vals = concepts[:, indices]
            satisfaction = vals.max(dim=1)[0]
            violation = 1 - satisfaction  # Violated when all low
            loss = loss + violation.mean()

        return loss

    def num_constraints(self) -> int:
        return len(self.implies) + len(self.mutex) + len(self.atleast_one)


# ============================================================================
# PART 3: SIMPLE CBM MODEL
# ============================================================================

class SimpleBooleanCBM(nn.Module):
    """
    Minimal Boolean CBM.

    Architecture:
        Image → Encoder → concept_logits → sigmoid → concept_probs
                                                        ↓
                                              LearnedThresholds
                                                        ↓
                                              binary_concepts
                                                        ↓
                                              ConstraintLayer (soft)
                                                        ↓
                                              Predictor → task_logits
    """

    def __init__(self,
                 concept_names: List[str],
                 num_classes: int,
                 encoder: nn.Module,
                 hidden_dim: int = 128):
        super().__init__()

        self.concept_names = concept_names
        self.num_concepts = len(concept_names)
        self.num_classes = num_classes

        # Components
        self.encoder = encoder  # User provides this
        self.thresholds = LearnedThresholds(self.num_concepts)
        self.constraints = ConstraintLayer(concept_names)

        # Concept → Task predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with all intermediate values (useful for debugging).
        """
        # 1. Encode
        concept_logits = self.encoder(images)
        concept_probs = torch.sigmoid(concept_logits)

        # 2. Threshold
        binary_concepts = self.thresholds(concept_probs)

        # 3. Constraint violation
        violation = self.constraints.violation_loss(binary_concepts)

        # 4. Predict
        task_logits = self.predictor(binary_concepts)

        return {
            'concept_logits': concept_logits,
            'concept_probs': concept_probs,
            'binary_concepts': binary_concepts,
            'task_logits': task_logits,
            'constraint_violation': violation,
            'thresholds': self.thresholds.thresholds.detach()
        }

    def freeze_thresholds(self):
        self.thresholds.freeze()


# ============================================================================
# PART 4: TRAINING STEP
# ============================================================================

@dataclass
class TrainConfig:
    lr: float = 1e-3
    lambda_concept: float = 1.0      # Weight for concept supervision
    lambda_constraint: float = 0.3   # Weight for constraint violation
    lambda_thresh_reg: float = 0.01  # Keep thresholds near 0.5


def train_step(
    model: SimpleBooleanCBM,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    task_labels: torch.Tensor,
    concept_labels: torch.Tensor,  # (batch, num_concepts) binary
    config: TrainConfig
) -> Dict[str, float]:
    """
    One training step.

    LOSS = task_loss + λ_c * concept_loss + λ_const * constraint_loss + λ_reg * threshold_reg
    """
    model.train()
    optimizer.zero_grad()

    # Forward
    out = model(images)

    # === LOSSES ===

    # 1. Task loss
    task_loss = F.cross_entropy(out['task_logits'], task_labels)

    # 2. Concept loss (supervised!)
    concept_loss = F.binary_cross_entropy_with_logits(
        out['concept_logits'],
        concept_labels.float()
    )

    # 3. Constraint loss
    constraint_loss = out['constraint_violation']

    # 4. Threshold regularization (don't go too extreme)
    thresh_reg = ((out['thresholds'] - 0.5) ** 2).mean()

    # Total
    total_loss = (
        task_loss +
        config.lambda_concept * concept_loss +
        config.lambda_constraint * constraint_loss +
        config.lambda_thresh_reg * thresh_reg
    )

    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Metrics
    with torch.no_grad():
        task_acc = (out['task_logits'].argmax(1) == task_labels).float().mean()
        concept_acc = ((out['concept_probs'] > 0.5) == concept_labels).float().mean()

    return {
        'loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'concept_loss': concept_loss.item(),
        'constraint_loss': constraint_loss.item(),
        'task_acc': task_acc.item(),
        'concept_acc': concept_acc.item(),
    }


# ============================================================================
# PART 5: INFERENCE STEP
# ============================================================================

@torch.no_grad()
def inference_step(
    model: SimpleBooleanCBM,
    images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Inference with explanation.

    IMPORTANT: Call model.freeze_thresholds() before inference!
    """
    model.eval()

    out = model(images)

    # Predictions
    probs = F.softmax(out['task_logits'], dim=1)
    predictions = probs.argmax(dim=1)
    confidences = probs.max(dim=1)[0]

    return {
        'predictions': predictions,
        'confidences': confidences,
        'binary_concepts': out['binary_concepts'],
        'concept_probs': out['concept_probs'],
        'constraint_violation': out['constraint_violation']
    }


def explain_prediction(
    model: SimpleBooleanCBM,
    binary_concepts: torch.Tensor,  # (1, num_concepts)
    prediction: int
) -> str:
    """Generate human-readable explanation."""

    binary = binary_concepts[0].cpu().numpy()

    active = [name for i, name in enumerate(model.concept_names) if binary[i] > 0.5]
    inactive = [name for i, name in enumerate(model.concept_names) if binary[i] <= 0.5]

    explanation = f"Predicted class {prediction} because:\n"
    explanation += f"  Active concepts: {', '.join(active) if active else 'none'}\n"
    explanation += f"  Inactive concepts: {', '.join(inactive[:5])}{'...' if len(inactive) > 5 else ''}"

    return explanation


# ============================================================================
# PART 6: FULL TRAINING LOOP
# ============================================================================

def train_loop(
    model: SimpleBooleanCBM,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    config: TrainConfig = None
):
    """
    Complete training loop.

    Expected data format:
        train_loader yields: (images, task_labels, concept_labels)
        - images: (batch, C, H, W)
        - task_labels: (batch,)
        - concept_labels: (batch, num_concepts) binary
    """
    config = config or TrainConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = {'train': [], 'val': []}
    best_val_acc = 0

    for epoch in range(num_epochs):

        # === TRAIN ===
        model.train()
        train_metrics = []

        for images, task_labels, concept_labels in train_loader:
            # Move to device if model is on GPU
            device = next(model.parameters()).device
            images = images.to(device)
            task_labels = task_labels.to(device)
            concept_labels = concept_labels.to(device)

            # Squeeze task_labels if they have extra dimensions
            if task_labels.dim() > 1:
                task_labels = task_labels.squeeze(-1)
            task_labels = task_labels.long()

            metrics = train_step(model, optimizer, images, task_labels, concept_labels, config)
            train_metrics.append(metrics)

        # Average train metrics
        avg_train = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]}
        history['train'].append(avg_train)

        # === VALIDATE ===
        model.eval()
        val_correct, val_total = 0, 0
        val_constraint_violations = []

        with torch.no_grad():
            for images, task_labels, concept_labels in val_loader:
                device = next(model.parameters()).device
                images = images.to(device)
                task_labels = task_labels.to(device)

                # Squeeze task_labels if they have extra dimensions
                if task_labels.dim() > 1:
                    task_labels = task_labels.squeeze(-1)
                task_labels = task_labels.long()

                out = model(images)
                preds = out['task_logits'].argmax(1)
                val_correct += (preds == task_labels).sum().item()
                val_total += len(task_labels)
                val_constraint_violations.append(out['constraint_violation'].item())

        val_acc = val_correct / val_total
        val_violation = np.mean(val_constraint_violations)
        history['val'].append({'acc': val_acc, 'constraint_violation': val_violation})

        # === LOGGING ===
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {avg_train['loss']:.4f}, Task Acc: {avg_train['task_acc']:.3f}, Concept Acc: {avg_train['concept_acc']:.3f}")
            print(f"  Val   - Acc: {val_acc:.3f}, Constraint Violation: {val_violation:.4f}")
            print(f"  Thresholds: {model.thresholds.thresholds.cpu().data.numpy().round(3)}")

        # === SAVE BEST ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest Val Acc: {best_val_acc:.3f}")

    # Freeze thresholds for inference
    model.freeze_thresholds()

    return history, best_state


# ============================================================================
# PART 7: CAUSAL DISCOVERY PLACEHOLDER (v2)
# ============================================================================

class CausalDiscovery:
    """
    Placeholder for causal constraint discovery.

    v2 WILL ADD:
    - Learn DAG from concept co-occurrences
    - Convert causal edges to implies constraints
    - Detect spurious correlations → add mutex

    FOR NOW: Manual constraints
    """

    @staticmethod
    def discover_from_data(
        concept_data: torch.Tensor,  # (N, num_concepts)
        concept_names: List[str],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Simple correlation-based discovery.

        Returns list of discovered constraints.
        """
        # Convert to numpy
        data = concept_data.numpy() if isinstance(concept_data, torch.Tensor) else concept_data

        # Compute correlations
        corr_matrix = np.corrcoef(data.T)

        constraints = []
        n = len(concept_names)

        for i in range(n):
            for j in range(i+1, n):
                corr = corr_matrix[i, j]

                if corr > threshold:
                    # High positive correlation → might be implies
                    constraints.append({
                        'type': 'implies_candidate',
                        'a': concept_names[i],
                        'b': concept_names[j],
                        'correlation': corr
                    })
                elif corr < -threshold:
                    # High negative correlation → might be mutex
                    constraints.append({
                        'type': 'mutex_candidate',
                        'a': concept_names[i],
                        'b': concept_names[j],
                        'correlation': corr
                    })

        return constraints

    @staticmethod
    def add_discovered_constraints(
        constraint_layer: ConstraintLayer,
        discovered: List[Dict],
        require_confirmation: bool = True
    ):
        """Add discovered constraints to model."""

        for c in discovered:
            if require_confirmation:
                print(f"Discovered: {c}")
                confirm = input("Add? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue

            if c['type'] == 'implies_candidate':
                constraint_layer.add_implies(c['a'], c['b'])
            elif c['type'] == 'mutex_candidate':
                constraint_layer.add_mutex(c['a'], c['b'])


# ============================================================================
# PART 8: DEMO
# ============================================================================

def create_simple_encoder(input_channels: int, num_concepts: int) -> nn.Module:
    """Simple CNN encoder for demo."""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_concepts)
    )


def demo():
    """Demo the minimal Boolean CBM."""

    print("="*60)
    print("MINIMAL BOOLEAN CBM DEMO")
    print("="*60)

    # Setup
    concept_names = ['opacity', 'cardiomegaly', 'effusion', 'pneumothorax', 'nodule']
    num_concepts = len(concept_names)
    num_classes = 2

    # Create model
    encoder = create_simple_encoder(3, num_concepts)
    model = SimpleBooleanCBM(concept_names, num_classes, encoder)

    # Add manual constraints (v2: will be discovered)
    print("\nAdding constraints:")
    model.constraints.add_mutex('pneumothorax', 'effusion')  # Can't have both
    model.constraints.add_implies('cardiomegaly', 'opacity')  # Heart issues → opacity
    model.constraints.add_atleast_one(['opacity', 'nodule', 'effusion'])  # At least one finding

    print(f"\nTotal constraints: {model.constraints.num_constraints()}")

    # Fake data
    print("\nSimulating training step...")
    images = torch.randn(8, 3, 64, 64)
    task_labels = torch.randint(0, 2, (8,))
    concept_labels = torch.randint(0, 2, (8, num_concepts))

    # Train step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainConfig()

    metrics = train_step(model, optimizer, images, task_labels, concept_labels, config)
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Task Acc: {metrics['task_acc']:.3f}")
    print(f"  Concept Acc: {metrics['concept_acc']:.3f}")
    print(f"  Constraint Violation: {metrics['constraint_loss']:.4f}")

    # Inference step
    print("\nSimulating inference...")
    model.freeze_thresholds()

    test_image = torch.randn(1, 3, 64, 64)
    result = inference_step(model, test_image)

    print(f"  Prediction: {result['predictions'].item()}")
    print(f"  Confidence: {result['confidences'].item():.3f}")
    print(f"  Binary concepts: {result['binary_concepts'][0].numpy()}")

    # Explanation
    explanation = explain_prediction(model, result['binary_concepts'], result['predictions'].item())
    print(f"\n{explanation}")

    # Causal discovery demo
    print("\n" + "="*60)
    print("CAUSAL DISCOVERY (Preview)")
    print("="*60)

    # Fake concept data
    fake_concept_data = torch.rand(100, num_concepts)
    # Inject correlation: when opacity high, cardiomegaly tends high
    fake_concept_data[:, 1] = 0.7 * fake_concept_data[:, 0] + 0.3 * torch.rand(100)

    discovered = CausalDiscovery.discover_from_data(fake_concept_data, concept_names, threshold=0.5)
    print(f"\nDiscovered {len(discovered)} potential constraints:")
    for c in discovered:
        print(f"  {c['type']}: {c['a']} ↔ {c['b']} (corr: {c['correlation']:.3f})")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)

    return model


if __name__ == "__main__":
    model = demo()