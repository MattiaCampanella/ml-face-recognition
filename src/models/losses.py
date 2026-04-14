from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TripletMiningStats:
	loss: float
	hard_positive_distance: float
	hard_negative_distance: float
	valid_anchors: int
	total_anchors: int


def pairwise_distance_matrix(embeddings: Tensor, *, squared: bool = False) -> Tensor:
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
	if embeddings.size(0) == 0:
		raise ValueError("Cannot compute pairwise distances for an empty batch.")

	distances = torch.cdist(embeddings, embeddings, p=2)
	if squared:
		distances = distances.pow(2)
	return distances


def batch_hard_triplet_loss(
	embeddings: Tensor,
	labels: Tensor,
	*,
	margin: float = 0.2,
	squared: bool = False,
	normalize_embeddings: bool = False,
) -> tuple[Tensor, TripletMiningStats]:
	"""Compute the batch-hard triplet loss with online hard positive/negative mining."""
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
	if labels.ndim != 1:
		labels = labels.view(-1)
	if embeddings.size(0) != labels.size(0):
		raise ValueError("embeddings and labels must have the same batch size.")

	if normalize_embeddings:
		embeddings = F.normalize(embeddings, dim=1)

	distances = pairwise_distance_matrix(embeddings, squared=squared)
	identity_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
	eye_mask = torch.eye(identity_mask.size(0), dtype=torch.bool, device=identity_mask.device)
	positive_mask = identity_mask & ~eye_mask
	negative_mask = ~identity_mask

	hard_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
	hard_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values

	valid_mask = torch.isfinite(hard_positive) & torch.isfinite(hard_negative)
	if valid_mask.any():
		loss = F.relu(hard_positive[valid_mask] - hard_negative[valid_mask] + margin).mean()
	else:
		loss = distances.sum() * 0.0

	stats = TripletMiningStats(
		loss=float(loss.detach().cpu().item()),
		hard_positive_distance=float(hard_positive[valid_mask].mean().detach().cpu().item()) if valid_mask.any() else 0.0,
		hard_negative_distance=float(hard_negative[valid_mask].mean().detach().cpu().item()) if valid_mask.any() else 0.0,
		valid_anchors=int(valid_mask.sum().item()),
		total_anchors=int(labels.numel()),
	)
	return loss, stats
