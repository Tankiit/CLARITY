"""
Optimized Rationale-Concept Bottleneck Model for Text Classification

This implementation provides a complete pipeline for training an optimized
rationale-concept bottleneck model on text classification datasets, with
specific optimizations for AG News and similar datasets
"""

import os
import argparse
import time
import json
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
logger.info(f"Using device: {device}")

#-----------------------------
# Configuration
#-----------------------------
class ModelConfig:
    """Configuration class for model architecture and training"""
    def __init__(
        self,
        base_model_name="distilbert-base-uncased",
        num_labels=4,
        num_concepts=50,
        hidden_size=768,  # Will be overridden by base model's hidden size
        dropout_rate=0.1,
        min_span_size=3,
        max_span_size=20,
        length_bonus_factor=0.01,
        concept_sparsity_weight=0.03,
        concept_diversity_weight=0.01,
        rationale_sparsity_weight=0.03,
        rationale_continuity_weight=0.1,
        classification_weight=1.0,
        target_rationale_percentage=0.2,
        enable_concept_interactions=False,
        use_skip_connection=True,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        batch_size=32,
        max_seq_length=128,
        learning_rate=2e-5,
        base_model_lr=1e-5,
        weight_decay=0.01,
        num_epochs=5,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=42,
        output_dir="models"
    ):
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.num_concepts = num_concepts
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.min_span_size = min_span_size
        self.max_span_size = max_span_size
        self.length_bonus_factor = length_bonus_factor
        self.concept_sparsity_weight = concept_sparsity_weight
        self.concept_diversity_weight = concept_diversity_weight
        self.rationale_sparsity_weight = rationale_sparsity_weight
        self.rationale_continuity_weight = rationale_continuity_weight
        self.classification_weight = classification_weight
        self.target_rationale_percentage = target_rationale_percentage
        self.enable_concept_interactions = enable_concept_interactions
        self.use_skip_connection = use_skip_connection
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.base_model_lr = base_model_lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.output_dir = output_dir

    def save(self, filepath):
        """Save configuration to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load configuration from a file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

#-----------------------------
# Optimized Rationale Extractor
#-----------------------------
class RationaleExtractor(nn.Module):
    """
    Memory-efficient rationale extractor using optimized attention

    This module extracts rationales (important spans of text) from input
    using an attention mechanism optimized for speed and memory efficiency.
    """
    def __init__(self, hidden_size, min_span_size=3, max_span_size=20,
                length_bonus_factor=0.01, dropout_rate=0.1):
        super().__init__()

        # Parameters for span extraction
        self.min_span_size = min_span_size
        self.max_span_size = max_span_size
        self.length_bonus_factor = length_bonus_factor

        # Optimized attention mechanism - single-head with reduced parameters
        self.query = nn.Linear(hidden_size, hidden_size // 2)
        self.key = nn.Linear(hidden_size, hidden_size // 2)

        # Token scoring network
        self.score_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in [self.query, self.key]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

        # Initialize final layer to output scores near zero initially
        nn.init.normal_(self.score_ffn[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_ffn[-1].bias)

    def forward(self, hidden_states, attention_mask):
        """Forward pass with optimized attention"""
        # Calculate token importance scores
        token_scores = self._calculate_token_scores(hidden_states, attention_mask)

        # Extract rationale spans efficiently
        rationale_mask = self._extract_rationale_spans_vectorized(token_scores, attention_mask)

        # Calculate token probabilities for visualization
        # Apply attention mask and add a small epsilon to avoid div by zero
        masked_scores = token_scores.masked_fill(~attention_mask.bool(), -10000.0)
        token_probs = F.softmax(masked_scores, dim=-1)

        # Apply rationale mask to get attended states
        rationale_mask_expanded = rationale_mask.unsqueeze(-1)
        attended_states = hidden_states * rationale_mask_expanded

        # Calculate the attended embeddings (weighted by rationale)
        lengths = torch.sum(rationale_mask, dim=1, keepdim=True) + 1e-6
        pooled_attended = torch.sum(attended_states, dim=1) / lengths

        return {
            'token_scores': token_scores,
            'rationale_mask': rationale_mask,
            'token_probs': token_probs,
            'attended_states': attended_states,
            'pooled_attended': pooled_attended
        }

    def _calculate_token_scores(self, hidden_states, attention_mask):
        """Calculate token importance scores with optimized attention"""
        # Calculate query and key representations
        query = self.query(hidden_states)
        key = self.key(hidden_states)

        # Calculate attention scores with scaling
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # Apply attention mask
        attn_scores = attn_scores.masked_fill(~attention_mask.unsqueeze(1).bool(), -10000.0)

        # Apply softmax along sequence dimension
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to hidden states
        context = torch.matmul(attn_probs, hidden_states)

        # Calculate token scores
        token_scores = self.score_ffn(context).squeeze(-1)

        # Apply attention mask to scores
        token_scores = token_scores.masked_fill(~attention_mask.bool(), -10000.0)

        return token_scores

    def _extract_rationale_spans_vectorized(self, scores, attention_mask):
        """Vectorized and optimized rationale extraction"""
        batch_size, seq_length = scores.shape
        device = scores.device

        # Create empty rationale masks
        rationales = torch.zeros_like(scores)

        # Get valid lengths from attention masks
        valid_lengths = attention_mask.sum(dim=1).int()

        # Calculate cumulative scores once (for all examples in batch)
        cum_scores = torch.cumsum(scores, dim=1)

        # Process all valid span lengths in parallel
        min_span = self.min_span_size
        max_span = self.max_span_size

        # For each example in batch
        for b in range(batch_size):
            valid_len = valid_lengths[b].item()
            if valid_len <= min_span:
                rationales[b, :valid_len] = 1.0
                continue

            # Get this example's cumulative scores
            ex_cum_scores = cum_scores[b, :valid_len]

            # Find best span efficiently with vectorized operations
            best_score = -float('inf')
            best_start, best_length = 0, min_span

            # For each possible span length
            max_span_for_ex = min(max_span, valid_len)
            for span_len in range(min_span, max_span_for_ex + 1):
                # Calculate scores for all spans of this length at once
                starts = torch.arange(valid_len - span_len + 1, device=device)
                span_scores = ex_cum_scores[starts + span_len - 1].clone()
                if starts[0] > 0:
                    span_scores -= ex_cum_scores[starts - 1]

                # Average scores
                avg_scores = span_scores / span_len

                # Add length bonus
                avg_scores += self.length_bonus_factor * (span_len / max_span)

                # Find best score for this length
                max_val, max_idx = torch.max(avg_scores, dim=0)
                if max_val > best_score:
                    best_score = max_val
                    best_start = max_idx.item()
                    best_length = span_len

            # Set rationale mask
            rationales[b, best_start:best_start+best_length] = 1.0

        return rationales

#-----------------------------
# Concept Mapper
#-----------------------------
class ConceptMapper(nn.Module):
    """
    Maps rationale-weighted embeddings to concept scores

    This module takes attended states from the rationale extractor
    and maps them to interpretable concept probabilities.
    """
    def __init__(self, input_dim, num_concepts, hidden_dim=384, dropout_rate=0.1,
                enable_interactions=False):
        super().__init__()

        # Simple concept encoder with reduced complexity
        self.concept_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_concepts)
        )

        # Optional concept interaction layer
        self.enable_interactions = enable_interactions
        if enable_interactions:
            self.concept_interactions = nn.Parameter(
                torch.zeros(num_concepts, num_concepts)
            )
        else:
            self.register_parameter('concept_interactions', None)

        self.num_concepts = num_concepts

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.concept_encoder:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, pooled_embeddings):
        """Map pooled embeddings to concept probabilities"""
        # Apply concept encoder to get initial concept scores
        concept_scores = self.concept_encoder(pooled_embeddings)

        # Save raw scores before interactions
        base_concept_scores = concept_scores

        # Apply concept interactions if enabled
        if self.enable_interactions and self.concept_interactions is not None:
            # Make interaction matrix symmetric and apply sigmoid
            sym_interactions = torch.sigmoid((self.concept_interactions + self.concept_interactions.t()) / 2)

            # Apply interactions
            interaction_effect = torch.matmul(concept_scores, sym_interactions)
            concept_scores = concept_scores + interaction_effect

        # Apply sigmoid to get concept probabilities
        concept_probs = torch.sigmoid(concept_scores)

        return {
            'concept_scores': concept_scores,
            'concept_probs': concept_probs,
            'base_concept_scores': base_concept_scores,
            'interaction_matrix': self.concept_interactions if self.enable_interactions else None
        }

#-----------------------------
# Unified Rationale-Concept Model
#-----------------------------
class RationaleConceptBottleneckModel(nn.Module):
    """
    Unified model that extracts rationales and maps them to concepts for classification

    This model implements the full pipeline:
    1. Encode text with efficient encoder
    2. Extract rationales with attention
    3. Map to interpretable concepts
    4. Classify based on concepts (with optional skip connection)
    """
    def __init__(self, config):
        super().__init__()

        # Load base encoder (DistilBERT for efficiency)
        self.encoder = AutoModel.from_pretrained(config.base_model_name)

        # Get actual hidden size from model
        config.hidden_size = self.encoder.config.hidden_size

        # Initialize component modules
        self.rationale_extractor = RationaleExtractor(
            hidden_size=config.hidden_size,
            min_span_size=config.min_span_size,
            max_span_size=config.max_span_size,
            length_bonus_factor=config.length_bonus_factor,
            dropout_rate=config.dropout_rate
        )

        self.concept_mapper = ConceptMapper(
            input_dim=config.hidden_size,
            num_concepts=config.num_concepts,
            hidden_dim=config.hidden_size // 2,
            dropout_rate=config.dropout_rate,
            enable_interactions=config.enable_concept_interactions
        )

        # Classification layer
        if config.use_skip_connection:
            # With skip connection, use both concepts and encoder output
            self.classifier = nn.Linear(config.num_concepts + config.hidden_size, config.num_labels)
        else:
            # Without skip connection, use only concepts
            self.classifier = nn.Linear(config.num_concepts, config.num_labels)

        # Set model attributes
        self.use_skip_connection = config.use_skip_connection
        self.num_concepts = config.num_concepts
        self.config = config

        # Apply LoRA if specified
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                # Define LoRA configuration
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=["query", "key", "value"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS
                )

                # Apply LoRA to the encoder
                self.encoder = get_peft_model(self.encoder, lora_config)
                logger.info("LoRA applied to the encoder model")
            except ImportError:
                logger.warning("PEFT not installed. LoRA not applied.")

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_attentions=False, inputs_embeds=None):
        """Forward pass through the unified model"""
        # Encode input
        if inputs_embeds is not None:
            # Use provided embeddings directly
            hidden_states = inputs_embeds
        else:
            # Get encoder outputs with attention if requested
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = encoder_outputs.last_hidden_state

        # Get CLS token embedding for skip connection
        cls_embedding = hidden_states[:, 0]

        # Extract rationales
        rationale_outputs = self.rationale_extractor(hidden_states, attention_mask)
        rationale_mask = rationale_outputs['rationale_mask']
        pooled_attended = rationale_outputs['pooled_attended']

        # Map to concepts
        concept_outputs = self.concept_mapper(pooled_attended)
        concept_probs = concept_outputs['concept_probs']

        # Classification
        if self.use_skip_connection:
            # With skip connection, use both concepts and CLS embedding
            combined_features = torch.cat([concept_probs, cls_embedding], dim=1)
            logits = self.classifier(combined_features)
        else:
            # Without skip connection, use only concepts
            logits = self.classifier(concept_probs)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Classification loss
            classification_loss = F.cross_entropy(logits, labels)

            # Regularization losses
            # 1. Concept sparsity - encourage fewer active concepts
            concept_sparsity_loss = concept_probs.mean()

            # 2. Concept diversity - encourage diversity among concepts
            if concept_probs.size(0) > 1:
                # Only compute if there are multiple examples in batch
                concept_mean = concept_probs.mean(dim=0)
                concept_diversity_loss = -torch.std(concept_mean)
            else:
                concept_diversity_loss = torch.tensor(0.0, device=classification_loss.device)

            # 3. Rationale sparsity - encourage concise rationales
            rationale_percentage = rationale_mask.sum(dim=1) / attention_mask.sum(dim=1)
            rationale_sparsity_loss = F.mse_loss(
                rationale_percentage,
                torch.ones_like(rationale_percentage) * self.config.target_rationale_percentage
            )

            # 4. Rationale continuity - encourage contiguous spans
            if rationale_mask.size(1) > 1:
                continuity_loss = torch.abs(rationale_mask[:, 1:] - rationale_mask[:, :-1]).mean()
            else:
                continuity_loss = torch.tensor(0.0, device=classification_loss.device)

            # Combine losses with configured weights
            loss = (
                self.config.classification_weight * classification_loss +
                self.config.concept_sparsity_weight * concept_sparsity_loss +
                self.config.concept_diversity_weight * concept_diversity_loss +
                self.config.rationale_sparsity_weight * rationale_sparsity_loss +
                self.config.rationale_continuity_weight * continuity_loss
            )

        # Combine all outputs
        outputs = {
            'logits': logits,
            'loss': loss,
            'rationale_mask': rationale_mask,
            'token_probs': rationale_outputs['token_probs'],
            'concept_probs': concept_probs,
            'concept_scores': concept_outputs['concept_scores'],
            'interaction_matrix': concept_outputs.get('interaction_matrix'),
            'cls_embedding': cls_embedding,
            'pooled_attended': pooled_attended
        }

        # Add attention weights if requested
        if output_attentions and 'attentions' in encoder_outputs:
            outputs['attentions'] = encoder_outputs.attentions

        return outputs

    def explain_prediction(self, tokenizer, text, min_concept_prob=0.5):
        """
        Generate explanation for a prediction

        Args:
            tokenizer: Tokenizer for preprocessing
            text: Input text
            min_concept_prob: Minimum probability for a concept to be considered active

        Returns:
            Dictionary with explanation information
        """
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        self.eval()
        with torch.no_grad():
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

        # Get prediction
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probs[0, prediction].item()

        # Get rationale text
        rationale_mask = outputs["rationale_mask"][0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Identify rationale tokens and their positions
        rationale_tokens = []
        for i, (token, mask) in enumerate(zip(tokens, rationale_mask)):
            if mask > 0.5 and inputs["attention_mask"][0, i] > 0:
                rationale_tokens.append(token)

        # Convert tokens to text while handling special tokens
        rationale_text = tokenizer.convert_tokens_to_string(rationale_tokens)

        # Get active concepts and their probabilities
        concept_probs = outputs["concept_probs"][0].cpu().numpy()

        # Get top concepts
        top_indices = np.argsort(concept_probs)[::-1][:5]  # Top 5 concepts
        top_concepts = [(f"concept_{idx}", concept_probs[idx])
                       for idx in top_indices if concept_probs[idx] > min_concept_prob]

        # Calculate rationale length as percentage of total sequence
        valid_length = inputs["attention_mask"][0].sum().item()
        rationale_length = rationale_mask.sum().item()
        rationale_percentage = rationale_length / valid_length if valid_length > 0 else 0

        # Create explanation
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "rationale": rationale_text,
            "rationale_percentage": rationale_percentage,
            "top_concepts": top_concepts,
            "class_probabilities": probs[0].cpu().numpy()
        }

        return explanation

    def intervene_on_concepts(self, tokenizer, text, concept_idx, new_value):
        """
        Perform intervention on concepts to analyze their effect

        Args:
            tokenizer: Tokenizer for preprocessing
            text: Input text
            concept_idx: Index of concept to modify
            new_value: New value for the concept (0-1)

        Returns:
            Dictionary with original and intervened predictions
        """
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Original forward pass
        self.eval()
        with torch.no_grad():
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

            # Get original prediction
            orig_logits = outputs["logits"]
            orig_probs = F.softmax(orig_logits, dim=1)
            orig_prediction = torch.argmax(orig_logits, dim=1).item()

            # Get concept probabilities
            concept_probs = outputs["concept_probs"].clone()

            # Modify the concept
            concept_probs[0, concept_idx] = new_value

            # Predict with modified concepts
            if self.use_skip_connection:
                combined = torch.cat([concept_probs, outputs["cls_embedding"]], dim=1)
                new_logits = self.classifier(combined)
            else:
                new_logits = self.classifier(concept_probs)

            new_probs = F.softmax(new_logits, dim=1)
            new_prediction = torch.argmax(new_logits, dim=1).item()

        # Get concept name (just use index for now)
        concept_name = f"concept_{concept_idx}"

        return {
            "original_prediction": orig_prediction,
            "original_probs": orig_probs[0].cpu().numpy(),
            "intervened_prediction": new_prediction,
            "intervened_probs": new_probs[0].cpu().numpy(),
            "concept_name": concept_name,
            "concept_value": {
                "original": outputs["concept_probs"][0, concept_idx].item(),
                "modified": new_value
            }
        }

#-----------------------------
# Dataset Processing
#-----------------------------
def load_and_process_dataset(dataset_name, tokenizer, config):
    """
    Load and preprocess a text classification dataset

    Args:
        dataset_name: Name of the dataset to load (e.g., 'ag_news')
        tokenizer: Tokenizer for text preprocessing
        config: Model configuration

    Returns:
        Dictionary with processed train, validation and test data
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Dataset-specific configurations
    dataset_configs = {
        'ag_news': {
            'name': 'ag_news',
            'text_field': 'text',
            'label_field': 'label',
            'num_labels': 4
        },
        'yelp_polarity': {
            'name': 'yelp_polarity',
            'text_field': 'text',
            'label_field': 'label',
            'num_labels': 2
        },
        'sst2': {
            'name': 'glue',
            'subset': 'sst2',
            'text_field': 'sentence',
            'label_field': 'label',
            'num_labels': 2
        },
        'dbpedia': {
            'name': 'dbpedia_14',
            'text_field': 'content',
            'label_field': 'label',
            'num_labels': 14
        }
    }

    # Check if dataset is supported
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(dataset_configs.keys())}")

    # Get dataset configuration
    ds_config = dataset_configs[dataset_name]

    # Update model config with num_labels
    config.num_labels = ds_config['num_labels']

    # Load dataset with cache
    cache_dir = os.path.join("cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    if 'subset' in ds_config:
        dataset = load_dataset(ds_config['name'], ds_config['subset'], cache_dir=cache_dir)
    else:
        dataset = load_dataset(ds_config['name'], cache_dir=cache_dir)

    # Preprocess function
    def preprocess_function(examples):
        # Tokenize texts
        result = tokenizer(
            examples[ds_config['text_field']],
            max_length=config.max_seq_length,
            padding="max_length",
            truncation=True
        )

        # Add labels
        result["labels"] = examples[ds_config['label_field']]

        return result

    # Process datasets
    # For datasets without a validation split, create one from the train split
    if 'validation' not in dataset and 'train' in dataset:
        # Create a validation split from training data
        train_val = dataset['train'].train_test_split(test_size=0.1, seed=config.seed)
        datasets = {
            'train': train_val['train'],
            'validation': train_val['test']
        }
        if 'test' in dataset:
            datasets['test'] = dataset['test']
        else:
            datasets['test'] = dataset['validation'] if 'validation' in dataset else train_val['test']
    else:
        datasets = dataset

    # Preprocess datasets
    processed_datasets = {}
    for split in datasets:
        # Use map for efficient preprocessing
        processed_datasets[split] = datasets[split].map(
            preprocess_function,
            batched=True,
            desc=f"Preprocessing {split}",
            num_proc=4,  # Parallel processing
            remove_columns=datasets[split].column_names
        )

        # Set format to PyTorch tensors
        processed_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    logger.info(f"Dataset processing complete")
    logger.info(f"Train size: {len(processed_datasets['train'])}")
    logger.info(f"Validation size: {len(processed_datasets['validation'])}")
    logger.info(f"Test size: {len(processed_datasets['test'])}")

    return processed_datasets

#-----------------------------
# Metrics Tracker
#-----------------------------
class MetricsTracker:
    """
    Tracks and visualizes training metrics and model explanations
    """
    def __init__(self, config):
        self.config = config

        # Create output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join(
            config.output_dir,
            f"{timestamp}_{config.base_model_name.split('/')[-1]}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize metrics storage
        self.batch_metrics = {'step': [], 'epoch': []}
        self.epoch_metrics = {'epoch': []}
        self.examples = {}

        # Best model tracking
        self.best_val_metric = 0
        self.best_val_epoch = 0

    def update_batch_metrics(self, metrics, step, epoch):
        """Track metrics for each batch"""
        self.batch_metrics['step'].append(step)
        self.batch_metrics['epoch'].append(epoch)

        for name, value in metrics.items():
            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

    def update_epoch_metrics(self, metrics, epoch, split='train'):
        """Track metrics for each epoch"""
        if 'epoch' not in self.epoch_metrics:
            self.epoch_metrics['epoch'] = []
        if epoch not in self.epoch_metrics['epoch']:
            self.epoch_metrics['epoch'].append(epoch)

        for name, value in metrics.items():
            key = f"{split}_{name}"
            if key not in self.epoch_metrics:
                self.epoch_metrics[key] = []
            self.epoch_metrics[key].append(value)

        # Track best validation metric
        if split == 'val' and 'accuracy' in metrics:
            if metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = metrics['accuracy']
                self.best_val_epoch = epoch

    def save_example_explanation(self, text, explanation, step=None):
        """Save example explanations"""
        # Generate unique example ID
        example_id = len(self.examples) + 1
        if step is not None:
            example_id = f"{example_id}_step{step}"

        self.examples[example_id] = {
            'text': text,
            'explanation': explanation
        }

    def generate_training_plots(self):
        """Generate training progress plots"""
        if len(self.epoch_metrics['epoch']) == 0:
            return

        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot loss curves
        if 'train_loss' in self.epoch_metrics and 'val_loss' in self.epoch_metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epoch_metrics['epoch'], self.epoch_metrics['train_loss'], 'b-', label='Train Loss')
            plt.plot(self.epoch_metrics['epoch'], self.epoch_metrics['val_loss'], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'loss_curves.png'))
            plt.close()

        # Plot accuracy curves
        if 'train_accuracy' in self.epoch_metrics and 'val_accuracy' in self.epoch_metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epoch_metrics['epoch'], self.epoch_metrics['train_accuracy'], 'b-', label='Train Accuracy')
            plt.plot(self.epoch_metrics['epoch'], self.epoch_metrics['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'))
            plt.close()

    def export_summary_report(self):
        """Export training summary report"""
        # Create report directory
        report_path = os.path.join(self.output_dir, 'training_summary.json')

        # Prepare summary data
        summary = {
            'config': self.config.__dict__,
            'metrics': {
                'best_val_accuracy': self.best_val_metric,
                'best_val_epoch': self.best_val_epoch
            }
        }

        # Add last epoch metrics
        if len(self.epoch_metrics['epoch']) > 0:
            last_epoch = self.epoch_metrics['epoch'][-1]
            summary['metrics']['last_epoch'] = last_epoch

            for key in self.epoch_metrics:
                if key != 'epoch' and key.startswith('val_'):
                    summary['metrics'][key] = self.epoch_metrics[key][-1]

        # Add test metrics if available
        if 'test_accuracy' in self.epoch_metrics:
            summary['metrics']['test_accuracy'] = self.epoch_metrics['test_accuracy'][-1]

        # Save summary report
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return report_path

    def save_model_explanations(self, model, tokenizer, dataset, num_examples=10):
        """Save model explanations for test examples"""
        # Create explanations directory
        explanations_dir = os.path.join(self.output_dir, 'explanations')
        os.makedirs(explanations_dir, exist_ok=True)

        # Set model to evaluation mode
        model.eval()

        # Get examples from test set
        examples = []
        labels = []

        # Select examples from test set
        test_loader = DataLoader(dataset['test'], batch_size=1, shuffle=True)
        for batch in test_loader:
            examples.append(batch)
            labels.append(batch['labels'].item())

            if len(examples) >= num_examples:
                break

        # Generate explanations
        all_explanations = []

        for i, batch in enumerate(examples):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get input text
            input_ids = batch['input_ids'][0]
            text = tokenizer.decode(input_ids, skip_special_tokens=True)

            # Generate explanation
            explanation = model.explain_prediction(tokenizer, text)

            # Add true label
            explanation['true_label'] = labels[i]

            # Save explanation
            all_explanations.append({
                'text': text,
                'explanation': explanation
            })

        # Save all explanations
        with open(os.path.join(explanations_dir, 'test_explanations.json'), 'w') as f:
            json.dump(all_explanations, f, indent=2)

        return all_explanations

#-----------------------------
# Training Function
#-----------------------------
def train_model(model, tokenizer, train_dataset, val_dataset, test_dataset, config, metrics_tracker):
    """
    Train and evaluate the model

    Args:
        model: Model to train
        tokenizer: Tokenizer for text processing
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Model configuration
        metrics_tracker: Metrics tracker

    Returns:
        Trained model and best model path
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    # Move model to device
    model = model.to(device)

    # Prepare optimizer and scheduler
    # Different learning rates for base model and other components
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': config.base_model_lr},
        {'params': model.rationale_extractor.parameters()},
        {'params': model.concept_mapper.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=config.learning_rate, weight_decay=config.weight_decay)

    # Learning rate scheduler with linear warmup and decay
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision training
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Set up model checkpoint directory
    checkpoint_dir = os.path.join(metrics_tracker.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency")

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_val_accuracy = 0.0

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward and backward pass with mixed precision
            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Update weights and learning rate
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                # Standard forward and backward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs['loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Update weights and learning rate
                optimizer.step()
                scheduler.step()

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Track metrics
            train_loss += loss.item()
            train_steps += 1
            global_step += 1

            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

            # Log batch metrics
            if global_step % 100 == 0:
                # Get predictions for accuracy
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == batch['labels']).float().mean().item()

                # Log metrics
                metrics_tracker.update_batch_metrics(
                    {
                        'loss': loss.item(),
                        'accuracy': accuracy,
                        'lr': scheduler.get_last_lr()[0]
                    },
                    global_step,
                    epoch
                )

                # Save example explanation
                if global_step % 500 == 0:
                    text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
                    explanation = model.explain_prediction(tokenizer, text)
                    metrics_tracker.save_example_explanation(text, explanation, global_step)

        # Calculate epoch metrics
        train_loss = train_loss / train_steps if train_steps > 0 else 0

        # Validation
        val_results = evaluate_model(model, val_loader)

        # Log epoch metrics
        metrics_tracker.update_epoch_metrics(
            {'loss': train_loss},
            epoch,
            'train'
        )
        metrics_tracker.update_epoch_metrics(
            val_results,
            epoch,
            'val'
        )

        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")

        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model saved! Accuracy: {best_val_accuracy:.4f}")

        # Generate training plots
        metrics_tracker.generate_training_plots()

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader)

    # Log test metrics
    metrics_tracker.update_epoch_metrics(
        test_results,
        config.num_epochs,
        'test'
    )

    logger.info(f"Test results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")
    if 'f1' in test_results:
        logger.info(f"  F1 Score: {test_results['f1']:.4f}")

    # Save model explanations
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    metrics_tracker.save_model_explanations(model, tokenizer, datasets)

    # Save final summary report
    report_path = metrics_tracker.export_summary_report()
    logger.info(f"Training summary saved to: {report_path}")

    return model, best_model_path

#-----------------------------
# Evaluation Function
#-----------------------------
def evaluate_model(model, dataloader):
    """
    Evaluate model on a data loader

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data

    Returns:
        Dictionary with evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()

    all_labels = []
    all_preds = []
    all_losses = []

    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Get predictions
            logits = outputs['logits']
            loss = outputs['loss']

            preds = torch.argmax(logits, dim=1)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_losses.append(loss.item())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # For multi-class classification, use macro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    # Calculate average loss
    avg_loss = sum(all_losses) / len(all_losses)

    # Return all metrics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#-----------------------------
# Main Function
#-----------------------------
def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a rationale-concept bottleneck model")

    # Dataset and model arguments
    parser.add_argument("--dataset", type=str, default="ag_news",
                        choices=["ag_news", "yelp_polarity", "sst2", "dbpedia"],
                        help="Dataset to train on")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Base model name (e.g., distilbert-base-uncased)")
    parser.add_argument("--num_concepts", type=int, default=50,
                        help="Number of concepts in the bottleneck")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--base_model_lr", type=float, default=1e-5,
                        help="Learning rate for base model")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Architecture arguments
    parser.add_argument("--enable_concept_interactions", action="store_true",
                        help="Enable concept interaction matrix")
    parser.add_argument("--disable_skip_connection", action="store_true",
                        help="Disable skip connection from encoder to classifier")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for efficient fine-tuning")

    # Rationale arguments
    parser.add_argument("--min_span_size", type=int, default=3,
                        help="Minimum size of rationale spans")
    parser.add_argument("--max_span_size", type=int, default=20,
                        help="Maximum size of rationale spans")
    parser.add_argument("--target_rationale_pct", type=float, default=0.2,
                        help="Target percentage of sequence to be rationale")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model and outputs")

    # Mixed-precision and performance options
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16/mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create model configuration
    config = ModelConfig(
        base_model_name=args.model,
        num_concepts=args.num_concepts,
        min_span_size=args.min_span_size,
        max_span_size=args.max_span_size,
        target_rationale_percentage=args.target_rationale_pct,
        enable_concept_interactions=args.enable_concept_interactions,
        use_skip_connection=not args.disable_skip_connection,
        use_lora=args.use_lora,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        base_model_lr=args.base_model_lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        output_dir=args.output_dir
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Load and preprocess dataset
    datasets = load_and_process_dataset(args.dataset, tokenizer, config)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)

    # Save configuration
    config_path = os.path.join(metrics_tracker.output_dir, 'config.json')
    config.save(config_path)

    # Initialize model
    model = RationaleConceptBottleneckModel(config)

    # Apply gradient checkpointing if specified
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Train model
    model, best_model_path = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets['train'],
        val_dataset=datasets['validation'],
        test_dataset=datasets['test'],
        config=config,
        metrics_tracker=metrics_tracker
    )

    logger.info(f"Training complete! Best model saved to: {best_model_path}")
    logger.info(f"All outputs saved to: {metrics_tracker.output_dir}")

if __name__ == "__main__":
    main()
