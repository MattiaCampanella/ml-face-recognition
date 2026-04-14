from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ResNet18Config:
	"""Configuration for the baseline ResNet-18 face model."""

	pretrained: bool = True
	embedding_dim: int = 512
	normalize_embeddings: bool = True
	classifier_num_classes: Optional[int] = None


class BaselineResNet18(nn.Module):
	"""ResNet-18 backbone with optional embedding projection and classifier head."""

	def __init__(
		self,
		pretrained: bool = True,
		embedding_dim: int = 512,
		normalize_embeddings: bool = True,
		classifier_num_classes: Optional[int] = None,
	) -> None:
		super().__init__()

		if embedding_dim <= 0:
			raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")

		weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		backbone = resnet18(weights=weights)

		in_features = backbone.fc.in_features
		backbone.fc = nn.Identity()
		self.backbone = backbone

		self.embedding_layer = nn.Identity()
		if embedding_dim != in_features:
			self.embedding_layer = nn.Linear(in_features, embedding_dim)

		self.embedding_dim = embedding_dim
		self.normalize_embeddings = normalize_embeddings

		self.classifier: Optional[nn.Linear] = None
		if classifier_num_classes is not None:
			if classifier_num_classes <= 1:
				raise ValueError(
					"classifier_num_classes must be > 1 when provided, "
					f"got {classifier_num_classes}."
				)
			self.classifier = nn.Linear(embedding_dim, classifier_num_classes)

	@classmethod
	def from_config(cls, cfg: ResNet18Config) -> "BaselineResNet18":
		return cls(
			pretrained=cfg.pretrained,
			embedding_dim=cfg.embedding_dim,
			normalize_embeddings=cfg.normalize_embeddings,
			classifier_num_classes=cfg.classifier_num_classes,
		)

	def forward_features(self, images: Tensor) -> Tensor:
		"""Return image embeddings from the backbone (+ optional projection)."""
		embeddings = self.backbone(images)
		embeddings = self.embedding_layer(embeddings)

		if self.normalize_embeddings:
			embeddings = nn.functional.normalize(embeddings, dim=1)

		return embeddings

	def forward(self, images: Tensor) -> dict[str, Tensor]:
		"""Run a full forward pass."""
		embeddings = self.forward_features(images)
		outputs: dict[str, Tensor] = {"embeddings": embeddings}

		if self.classifier is not None:
			outputs["logits"] = self.classifier(embeddings)

		return outputs


def build_baseline_resnet18(
	pretrained: bool = True,
	embedding_dim: int = 512,
	normalize_embeddings: bool = True,
	classifier_num_classes: Optional[int] = None,
) -> BaselineResNet18:
	"""Factory function used by training/inference scripts."""
	return BaselineResNet18(
		pretrained=pretrained,
		embedding_dim=embedding_dim,
		normalize_embeddings=normalize_embeddings,
		classifier_num_classes=classifier_num_classes,
	)
