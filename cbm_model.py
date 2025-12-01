"""
Base Concept Bottleneck Model (CBM) Framework

This module provides a base class for implementing Concept Bottleneck Models
that can be easily extended with different architectures and features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision import models


class BaseCBM(nn.Module, ABC):
    """
    Abstract base class for Concept Bottleneck Models.

    This class defines the common interface and basic functionality that all CBM
    implementations should follow. Subclasses need to implement the specific
    architecture details.
    """

    def __init__(self, num_concepts, num_classes, feature_dim=512):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # These should be initialized by subclasses
        self.feature_extractor = None
        self.concept_predictor = None
        self.task_predictor = None

    @abstractmethod
    def _build_feature_extractor(self):
        """Build the feature extractor component. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_concept_predictor(self):
        """Build the concept predictor component. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_task_predictor(self):
        """Build the task predictor component. Must be implemented by subclasses."""
        pass

    def forward(self, images):
        """
        Forward pass through the CBM.

        Args:
            images: Input images tensor (batch_size, channels, height, width)

        Returns:
            Dictionary containing:
                - concepts: Predicted concepts (batch_size, num_concepts)
                - task_logits: Raw task predictions (batch_size, num_classes)
                - task_predictions: Softmax task predictions (batch_size, num_classes)
        """
        # Extract features
        features = self._extract_features(images)

        # Predict concepts
        concepts = self.concept_predictor(features)

        # Predict task
        task_logits = self.task_predictor(concepts)

        return {
            'concepts': concepts,
            'task_logits': task_logits,
            'task_predictions': F.softmax(task_logits, dim=-1)
        }

    def _extract_features(self, images):
        """Extract features from input images."""
        if hasattr(self.feature_extractor, 'parameters') and not any(p.requires_grad for p in self.feature_extractor.parameters()):
            with torch.no_grad():
                features = self.feature_extractor(images)
        else:
            features = self.feature_extractor(images)

        # Handle different feature extractor outputs
        if len(features.shape) == 4:  # Convolutional feature maps
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        elif len(features.shape) == 3:  # Sequence features
            features = features.mean(dim=1)  # Global average pooling

        return features

    def compute_loss(self, images, task_labels, concept_labels=None, concept_weight=0.1):
        """
        Compute the total loss for the CBM.

        Args:
            images: Input images
            task_labels: Task labels for supervised learning
            concept_labels: Concept labels for supervised concept learning (optional)
            concept_weight: Weight for concept loss term

        Returns:
            Dictionary containing different loss components
        """
        output = self.forward(images)

        # Task loss
        task_loss = F.cross_entropy(output['task_logits'], task_labels)

        # Concept loss (if labels available)
        if concept_labels is not None:
            concept_loss = F.binary_cross_entropy(
                output['concepts'], concept_labels.float()
            )
        else:
            concept_loss = torch.tensor(0.0, device=images.device)

        total_loss = task_loss + concept_weight * concept_loss

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'concept_loss': concept_loss
        }

    def predict_concepts(self, images):
        """Predict only concepts from input images."""
        features = self._extract_features(images)
        concepts = self.concept_predictor(features)
        return concepts

    def predict_task_from_concepts(self, concepts):
        """Predict task labels from given concepts."""
        task_logits = self.task_predictor(concepts)
        return {
            'task_logits': task_logits,
            'task_predictions': F.softmax(task_logits, dim=-1)
        }

    def freeze_feature_extractor(self):
        """Freeze the feature extractor parameters."""
        if self.feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("✅ Feature extractor frozen")

    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor parameters."""
        if self.feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print("✅ Feature extractor unfrozen")


class BaselineCBM(BaseCBM):
    """
    Simplest possible CBM implementation

    Architecture:
    Input Image → ResNet-18 → Concepts → Task Prediction
    """

    def __init__(self, num_concepts=40, num_classes=2, pretrained=True):
        super().__init__(num_concepts=num_concepts, num_classes=num_classes)

        # Build components
        self._build_feature_extractor(pretrained=pretrained)
        self._build_concept_predictor()
        self._build_task_predictor()

        print(f"✅ Baseline CBM initialized: {num_concepts} concepts → {num_classes} classes")

    def _build_feature_extractor(self, pretrained=True):
        """Build frozen pretrained ResNet-18 feature extractor."""
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        if pretrained:
            self.freeze_feature_extractor()

    def _build_concept_predictor(self):
        """Build the concept predictor MLP."""
        self.concept_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_concepts),
            nn.Sigmoid()
        )

    def _build_task_predictor(self):
        """Build the task predictor MLP."""
        self.task_predictor = nn.Sequential(
            nn.Linear(self.num_concepts, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )


class SimpleCBM(BaseCBM):
    """
    A simpler CBM variant with smaller networks for faster training

    Architecture:
    Input Image → Simple CNN → Concepts → Task Prediction
    """

    def __init__(self, num_concepts=20, num_classes=2, input_channels=3):
        super().__init__(num_concepts=num_concepts, num_classes=num_classes, feature_dim=128)
        self.input_channels = input_channels

        self._build_feature_extractor()
        self._build_concept_predictor()
        self._build_task_predictor()

        print(f"✅ Simple CBM initialized: {num_concepts} concepts → {num_classes} classes")

    def _build_feature_extractor(self):
        """Build a simple CNN feature extractor."""
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _build_concept_predictor(self):
        """Build a simple concept predictor."""
        self.concept_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_concepts),
            nn.Sigmoid()
        )

    def _build_task_predictor(self):
        """Build a simple task predictor."""
        self.task_predictor = nn.Sequential(
            nn.Linear(self.num_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )


def test_models():
    """Test both CBM implementations."""
    print("🧪 Testing CBM implementations...")

    # Test data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 64, 64)
    task_labels = torch.randint(0, 2, (batch_size,))
    concept_labels = torch.rand(batch_size, 40)

    # Test BaselineCBM
    print("\n📋 Testing BaselineCBM:")
    baseline_model = BaselineCBM(num_concepts=40, num_classes=2)

    baseline_output = baseline_model(dummy_images)
    print(f"✅ Forward pass: concepts {baseline_output['concepts'].shape}, logits {baseline_output['task_logits'].shape}")

    baseline_losses = baseline_model.compute_loss(dummy_images, task_labels, concept_labels)
    print(f"✅ Loss: {baseline_losses['total_loss'].item():.4f}")

    # Test SimpleCBM
    print("\n📋 Testing SimpleCBM:")
    simple_model = SimpleCBM(num_concepts=20, num_classes=2)

    simple_output = simple_model(dummy_images)
    print(f"✅ Forward pass: concepts {simple_output['concepts'].shape}, logits {simple_output['task_logits'].shape}")

    simple_losses = simple_model.compute_loss(dummy_images, task_labels)
    print(f"✅ Loss: {simple_losses['total_loss'].item():.4f}")

    # Test concept-only prediction
    print("\n📋 Testing concept-only prediction:")
    concepts = baseline_model.predict_concepts(dummy_images)
    print(f"✅ Concepts shape: {concepts.shape}")

    # Test task from concepts
    task_from_concepts = baseline_model.predict_task_from_concepts(concepts)
    print(f"✅ Task from concepts shape: {task_from_concepts['task_logits'].shape}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_models()