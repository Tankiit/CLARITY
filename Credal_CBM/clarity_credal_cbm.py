# Credal Concept Bottleneck Model for CLARITY Framework
# Integrates uncertainty-aware concept learning with text classification
# Architecture: Text -> Rationales -> Credal Concepts (with uncertainty) -> Prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass


@dataclass
class TextConceptAnnotation:
    # Text-specific concept annotation with uncertainty
    concept_name: str
    concept_value: float  # 0-1
    supporting_rationale: str  # Text span
    rationale_start: int
    rationale_end: int
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Text ambiguity
    annotator_confidence: float = 1.0


class CredalSet:
    # Credal set for text concepts with uncertainty decomposition

    def __init__(self, extreme_points: np.ndarray,
                 concept_name: str = None,
                 aleatoric_variance: Optional[np.ndarray] = None,
                 supporting_rationales: List[str] = None):
        self.extreme_points = extreme_points
        self.n_points, self.n_classes = extreme_points.shape
        self.concept_name = concept_name
        self.aleatoric_variance = aleatoric_variance
        self.supporting_rationales = supporting_rationales or []

        # Validate probability distributions
        assert np.allclose(extreme_points.sum(axis=1), 1.0), "Invalid probability distributions"
        assert np.all(extreme_points >= 0), "Negative probabilities detected"

    def epistemic_uncertainty(self) -> float:
        # Model/knowledge uncertainty (reducible)
        ranges = np.max(self.extreme_points, axis=0) - np.min(self.extreme_points, axis=0)
        return np.mean(ranges)

    def aleatoric_uncertainty(self) -> float:
        # Text ambiguity (irreducible)
        if self.aleatoric_variance is not None:
            return np.mean(self.aleatoric_variance)
        return 0.0

    def total_uncertainty(self) -> float:
        return self.epistemic_uncertainty() + self.aleatoric_uncertainty()

    def uncertainty_decomposition(self) -> Dict:
        # Decompose uncertainty into epistemic and aleatoric components
        epistemic = self.epistemic_uncertainty()
        aleatoric = self.aleatoric_uncertainty()
        total = epistemic + aleatoric

        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'epistemic_ratio': epistemic / total if total > 0 else 0.0,
            'aleatoric_ratio': aleatoric / total if total > 0 else 0.0
        }

    def mean_probability(self, class_idx: int) -> float:
        return np.mean(self.extreme_points[:, class_idx])

    def interval_probability(self, class_idx: int) -> Tuple[float, float]:
        probs = self.extreme_points[:, class_idx]
        return float(np.min(probs)), float(np.max(probs))


class RationaleExtractor(nn.Module):
    # Extract important text spans (rationales) with uncertainty
    # Adapted from CLARITY's rationale extraction

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1,
                 min_span_size: int = 3, max_span_size: int = 20):
        super().__init__()

        self.hidden_size = hidden_size
        self.min_span_size = min_span_size
        self.max_span_size = max_span_size

        # Token-level importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

        # Uncertainty estimator for rationale selection
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict:
        # Extract rationales with uncertainty quantification
        # Returns:
        #   rationale_mask: Binary mask of selected tokens
        #   pooled_attended: Weighted representation
        #   rationale_uncertainty: Uncertainty in rationale selection
        batch_size, seq_length, _ = hidden_states.shape

        # Compute token importance scores
        importance_scores = self.importance_scorer(hidden_states).squeeze(-1)
        importance_scores = importance_scores.masked_fill(attention_mask == 0, float('-inf'))
        importance_probs = torch.softmax(importance_scores, dim=-1)

        # Estimate uncertainty in rationale selection
        rationale_uncertainty = self.uncertainty_estimator(hidden_states).squeeze(-1)
        rationale_uncertainty = rationale_uncertainty * attention_mask

        # Create rationale mask (top-k selection)
        threshold = torch.quantile(
            importance_probs[attention_mask.bool()],
            q=0.8,
            dim=0
        )
        rationale_mask = (importance_probs >= threshold).float() * attention_mask

        # Weighted pooling using importance scores
        weighted_hidden = hidden_states * importance_probs.unsqueeze(-1)
        pooled_attended = weighted_hidden.sum(dim=1)

        return {
            'rationale_mask': rationale_mask,
            'pooled_attended': pooled_attended,
            'importance_probs': importance_probs,
            'rationale_uncertainty': rationale_uncertainty.mean(dim=1)
        }


class CredalConceptMapper(nn.Module):
    # Map rationales to concepts with credal sets (uncertainty)

    def __init__(self, input_dim: int, num_concepts: int,
                 n_credal_points: int = 5, hidden_dim: int = 256):
        super().__init__()

        self.num_concepts = num_concepts
        self.n_credal_points = n_credal_points

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Multiple heads for credal set estimation (epistemic)
        self.concept_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, num_concepts)
            for _ in range(n_credal_points)
        ])

        # Aleatoric uncertainty estimator (text ambiguity)
        self.aleatoric_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Softplus()
        )

        # Concept importance (learnable)
        self.concept_importance = nn.Parameter(torch.ones(num_concepts))

    def forward(self, rationale_features: torch.Tensor) -> Dict:
        # Map rationales to credal concept sets
        # Returns:
        #   concept_credal_sets: List of CredalSet objects per sample
        #   concept_expectations: Expected concept values
        #   uncertainty_metrics: Epistemic, aleatoric, total
        batch_size = rationale_features.shape[0]

        # Shared encoding
        shared_features = self.shared_encoder(rationale_features)

        # Multiple predictions for credal sets (epistemic uncertainty)
        concept_logits = []
        for head in self.concept_heads:
            logits = torch.sigmoid(head(shared_features))
            concept_logits.append(logits)
        concept_logits = torch.stack(concept_logits, dim=1)  # [batch, n_heads, n_concepts]

        # Estimate aleatoric uncertainty (text ambiguity)
        aleatoric_vars = self.aleatoric_estimator(rationale_features)

        # Build credal sets
        batch_credal_sets = []
        concept_expectations = []

        for b in range(batch_size):
            sample_credal_sets = []
            sample_expectations = []

            for c in range(self.num_concepts):
                # Get predictions for this concept
                concept_preds = concept_logits[b, :, c].detach().cpu().numpy()

                # Create binary credal set
                extreme_points = np.column_stack([
                    1 - concept_preds,  # P(absent)
                    concept_preds       # P(present)
                ])

                # Get aleatoric variance
                aleatoric_var = aleatoric_vars[b, c].detach().cpu().numpy()

                # Create credal set
                credal_set = CredalSet(
                    extreme_points,
                    concept_name=f"concept_{c}",
                    aleatoric_variance=np.array([aleatoric_var, aleatoric_var])
                )

                sample_credal_sets.append(credal_set)
                sample_expectations.append(np.mean(concept_preds))

            batch_credal_sets.append(sample_credal_sets)
            concept_expectations.append(sample_expectations)

        # Compute uncertainty metrics
        epistemic_unc = np.array([[cs.epistemic_uncertainty()
                                   for cs in sample]
                                  for sample in batch_credal_sets])

        aleatoric_unc = aleatoric_vars.detach().cpu().numpy()
        total_unc = epistemic_unc + aleatoric_unc

        return {
            'concept_credal_sets': batch_credal_sets,
            'concept_expectations': torch.tensor(concept_expectations,
                                                 device=rationale_features.device),
            'concept_importance': torch.sigmoid(self.concept_importance),
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': total_unc
        }


class ClarityCredalCBM(nn.Module):
    # Complete CLARITY Credal CBM:
    # Text -> Rationales -> Credal Concepts -> Prediction

    def __init__(self, base_model_name: str, num_concepts: int,
                 num_classes: int, n_credal_points: int = 5,
                 concept_names: List[str] = None):
        super().__init__()

        # Text encoder (frozen or fine-tuned)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_names = concept_names or [f"concept_{i}" for i in range(num_concepts)]

        # Pipeline components
        self.rationale_extractor = RationaleExtractor(
            self.hidden_size,
            dropout_rate=0.1
        )

        self.concept_mapper = CredalConceptMapper(
            self.hidden_size,
            num_concepts,
            n_credal_points=n_credal_points
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_rationales: bool = False) -> Dict:
        # Forward pass through CLARITY Credal CBM
        # Args:
        #   input_ids: Token IDs [batch_size, seq_length]
        #   attention_mask: Attention mask
        #   labels: Optional labels for training
        #   return_rationales: Whether to return rationale information
        # Returns:
        #   Dictionary with predictions, credal sets, and uncertainties

        # 1. Encode text
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state

        # 2. Extract rationales with uncertainty
        rationale_outputs = self.rationale_extractor(hidden_states, attention_mask)

        # 3. Map rationales to credal concepts
        concept_outputs = self.concept_mapper(rationale_outputs['pooled_attended'])

        # 4. Classify using concept expectations
        concept_expectations = concept_outputs['concept_expectations']
        weighted_concepts = concept_expectations * concept_outputs['concept_importance']
        logits = self.classifier(weighted_concepts)

        # Compute loss if training
        loss = None
        loss_components = {}

        if labels is not None:
            # Classification loss
            classification_loss = F.cross_entropy(logits, labels)

            # Epistemic regularization (encourage confidence when appropriate)
            epistemic_unc = torch.tensor(
                concept_outputs['epistemic_uncertainty'].mean(),
                device=logits.device
            )
            epistemic_reg = epistemic_unc * 0.01

            # Aleatoric regularization
            aleatoric_reg = concept_outputs['aleatoric_uncertainty'].mean() * 0.01

            # Rationale sparsity (encourage concise rationales)
            rationale_sparsity = rationale_outputs['rationale_mask'].mean() * 0.03

            # Uncertainty calibration
            pred_correct = (logits.argmax(1) == labels).float()
            total_unc_mean = torch.tensor(
                concept_outputs['total_uncertainty'].mean(1),
                device=logits.device
            )
            calibration_loss = F.mse_loss(total_unc_mean, 1 - pred_correct) * 0.05

            # Total loss
            loss = (classification_loss + epistemic_reg + aleatoric_reg +
                   rationale_sparsity + calibration_loss)

            loss_components = {
                'classification': classification_loss.item(),
                'epistemic_reg': epistemic_reg.item(),
                'aleatoric_reg': aleatoric_reg.item(),
                'rationale_sparsity': rationale_sparsity.item(),
                'calibration': calibration_loss.item()
            }

        output = {
            'logits': logits,
            'loss': loss,
            'loss_components': loss_components,
            'concept_credal_sets': concept_outputs['concept_credal_sets'],
            'concept_expectations': concept_expectations,
            'uncertainty_metrics': {
                'epistemic': concept_outputs['epistemic_uncertainty'],
                'aleatoric': concept_outputs['aleatoric_uncertainty'],
                'total': concept_outputs['total_uncertainty'],
                'rationale_uncertainty': rationale_outputs['rationale_uncertainty'].detach().cpu().numpy()
            }
        }

        if return_rationales:
            output['rationale_mask'] = rationale_outputs['rationale_mask']
            output['importance_probs'] = rationale_outputs['importance_probs']

        return output

    def explain_prediction(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          tokenizer: AutoTokenizer) -> Dict:
        # Generate detailed explanation for a prediction
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask, return_rationales=True)

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get rationales
        rationale_mask = outputs['rationale_mask'][0].cpu().numpy()
        importance_probs = outputs['importance_probs'][0].cpu().numpy()

        rationale_tokens = [
            (token, float(importance_probs[i]))
            for i, token in enumerate(tokens)
            if rationale_mask[i] > 0 and token not in ['[CLS]', '[SEP]', '[PAD]']
        ]

        # Get top uncertain concepts
        credal_sets = outputs['concept_credal_sets'][0]
        concept_uncertainties = [(self.concept_names[i], cs.uncertainty_decomposition())
                                 for i, cs in enumerate(credal_sets)]
        concept_uncertainties.sort(key=lambda x: x[1]['total'], reverse=True)

        # Prediction
        pred_class = outputs['logits'][0].argmax().item()
        pred_confidence = torch.softmax(outputs['logits'][0], dim=0).max().item()

        return {
            'predicted_class': pred_class,
            'confidence': pred_confidence,
            'rationale_tokens': rationale_tokens,
            'top_uncertain_concepts': concept_uncertainties[:10],
            'mean_epistemic_uncertainty': outputs['uncertainty_metrics']['epistemic'][0].mean(),
            'mean_aleatoric_uncertainty': outputs['uncertainty_metrics']['aleatoric'][0].mean()
        }


def uncertainty_guided_active_learning(model: ClarityCredalCBM,
                                      unlabeled_loader,
                                      budget: int = 100) -> List[int]:
    # Select samples for annotation based on uncertainty decomposition
    # Prioritize high epistemic (reducible) uncertainty
    model.eval()
    sample_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(unlabeled_loader):
            outputs = model(batch['input_ids'], batch['attention_mask'])

            for i in range(batch['input_ids'].shape[0]):
                epistemic = outputs['uncertainty_metrics']['epistemic'][i].mean()
                aleatoric = outputs['uncertainty_metrics']['aleatoric'][i].mean()

                # Score: prioritize high epistemic, penalize high aleatoric
                score = epistemic / (1 + aleatoric)

                sample_scores.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'score': score,
                    'epistemic': epistemic,
                    'aleatoric': aleatoric
                })

    # Sort by score and select top samples
    sample_scores.sort(key=lambda x: x['score'], reverse=True)
    selected_indices = [s['sample_idx'] for s in sample_scores[:budget]]

    return selected_indices
