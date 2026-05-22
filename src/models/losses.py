from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
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
	mining_strategy: str = "hard",
	margin_type: str = "soft",
) -> tuple[Tensor, TripletMiningStats]:
	"""Compute triplet loss with configurable online mining strategy.

	Supported strategies:
	- "hard": hardest negative in the batch.
	- "semi_hard": closest negative with d(a,p) < d(a,n) < d(a,p)+margin, hard fallback.
	- "easy_semi_hard": closest negative with d(a,n) > d(a,p), hard fallback.

	margin_type:
	- "soft": loss = mean(softplus(d_pos - d_neg))  — smooth, no explicit margin.
	- "hinge": loss = mean(relu(d_pos - d_neg + margin))  — classic triplet with hard margin.
	"""
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
	if labels.ndim != 1:
		labels = labels.view(-1)
	if embeddings.size(0) != labels.size(0):
		raise ValueError("embeddings and labels must have the same batch size.")

	if normalize_embeddings:
		embeddings = F.normalize(embeddings, dim=1)

	strategy = mining_strategy.lower()
	if strategy not in {"hard", "semi_hard", "easy_semi_hard"}:
		raise ValueError(
			"Unsupported mining_strategy. "
			f"Expected one of ('hard', 'semi_hard', 'easy_semi_hard'), got: {mining_strategy}"
		)
	margin_mode = margin_type.lower()
	if margin_mode not in {"soft", "hinge"}:
		raise ValueError(
			"Unsupported margin_type. "
			f"Expected one of ('soft', 'hinge'), got: {margin_type}"
		)

	distances = pairwise_distance_matrix(embeddings, squared=squared)
	identity_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
	eye_mask = torch.eye(identity_mask.size(0), dtype=torch.bool, device=identity_mask.device)
	positive_mask = identity_mask & ~eye_mask
	negative_mask = ~identity_mask

	hard_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
	hard_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values

	if strategy == "hard":
		negative_choice = hard_negative
	else:
		anchor_positive = hard_positive.unsqueeze(1)
		if strategy == "semi_hard":
			candidate_mask = (
				negative_mask
				& (distances > anchor_positive)
				& (distances < (anchor_positive + margin))
			)
		else:
			candidate_mask = negative_mask & (distances > anchor_positive)

		candidate_negative = distances.masked_fill(~candidate_mask, float("inf")).min(dim=1).values
		negative_choice = torch.where(torch.isfinite(candidate_negative), candidate_negative, hard_negative)

	valid_mask = torch.isfinite(hard_positive) & torch.isfinite(negative_choice)
	if valid_mask.any():
		diff = hard_positive[valid_mask] - negative_choice[valid_mask]
		if margin_mode == "hinge":
			loss = F.relu(diff + margin).mean()
		else:
			loss = F.softplus(diff).mean()
	else:
		loss = distances.sum() * 0.0

	stats = TripletMiningStats(
		loss=float(loss.detach().cpu().item()),
		hard_positive_distance=float(hard_positive[valid_mask].mean().detach().cpu().item()) if valid_mask.any() else 0.0,
		hard_negative_distance=float(negative_choice[valid_mask].mean().detach().cpu().item()) if valid_mask.any() else 0.0,
		valid_anchors=int(valid_mask.sum().item()),
		total_anchors=int(labels.numel()),
	)
	return loss, stats


class ArcFaceLoss(nn.Module):
	"""ArcFace: Additive Angular Margin Loss for Face Recognition, including
	Sub-center ArcFace support.

	Reference:
	- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
	- Boutros et al., "Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces", ECCV 2020.

	The loss adds a fixed angular margin *m* to the angle between an
	embedding and its ground-truth class centre, then scales the resulting
	cosine similarities by *s* before computing cross-entropy.
	When k > 1, it implements Sub-center ArcFace, keeping K sub-centers
	per class and selecting the one with the maximum cosine similarity.

	Args:
		num_classes: Number of training identities (classes).
		embedding_dim: Dimension of the L2-normalised embeddings.
		s: Logit scale (typically 30–64). Controls the "peakiness" of the softmax.
		m: Angular margin in radians (typically 0.3–0.5).
		k: Number of sub-centers per class (k=1 is standard ArcFace).
		easy_margin: When True, uses a simplified boundary condition.
	"""

	def __init__(
		self,
		num_classes: int,
		embedding_dim: int,
		*,
		s: float = 30.0,
		m: float = 0.5,
		k: int = 3,
		easy_margin: bool = False,
	) -> None:
		super().__init__()
		if num_classes <= 0:
			raise ValueError(f"num_classes must be > 0, got {num_classes}")
		if embedding_dim <= 0:
			raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
		if s <= 0:
			raise ValueError(f"s (scale) must be > 0, got {s}")
		if not (0.0 < m < math.pi / 2):
			raise ValueError(f"m (margin) must be in (0, π/2), got {m}")
		if k <= 0:
			raise ValueError(f"k (sub-centers) must be > 0, got {k}")

		self.num_classes = num_classes
		self.embedding_dim = embedding_dim
		self.s = s
		self.m = m
		self.k = k
		self.easy_margin = easy_margin

		# Learnable class-centre matrix W.
		# For Sub-center, we have K centers per identity.
		self.W = nn.Parameter(torch.empty(num_classes, k, embedding_dim))
		nn.init.xavier_uniform_(self.W)

		# Pre-compute trigonometric constants for the margin.
		self.cos_m: float = math.cos(m)
		self.sin_m: float = math.sin(m)
		# Safe-guard threshold: cos(π - m)
		self.th: float = math.cos(math.pi - m)   # = -cos(m)
		self.mm: float = math.sin(math.pi - m) * m  # = sin(m)*m

	def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
		"""Compute ArcFace / Sub-center ArcFace loss.

		Args:
			embeddings: L2-normalised embeddings, shape (B, embedding_dim).
			labels: Integer class indices, shape (B,).

		Returns:
			Scalar loss tensor.
		"""
		if embeddings.ndim != 2:
			raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
		if labels.ndim != 1:
			labels = labels.view(-1)

		# Normalise class-centre weights on the last dimension (D).
		W_norm = F.normalize(self.W, dim=2)  # (C, K, D)

		# cos(θ_i) = e · W_i
		# embeddings is (B, D). We want a (B, C) matrix where each (b, c)
		# is max_k (e_b · W_{c, k}).
		# First compute all cosines: reshape W_norm to (C*K, D), multiply, then reshape back.
		cos_theta_all = embeddings @ W_norm.view(-1, self.embedding_dim).t()  # (B, C*K)
		cos_theta_all = cos_theta_all.view(-1, self.num_classes, self.k)      # (B, C, K)

		# Sub-center ArcFace: take the max over the K sub-centers.
		if self.k > 1:
			cos_theta, _ = cos_theta_all.max(dim=2)  # (B, C)
		else:
			cos_theta = cos_theta_all.squeeze(2)     # (B, C)

		cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # numerical safety

		# sin(θ) = sqrt(1 - cos²(θ)).
		sin_theta = (1.0 - cos_theta.pow(2)).clamp(min=0.0).sqrt()

		# cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m).
		cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

		if self.easy_margin:
			# Only apply the margin when cos(θ) > 0 (θ < π/2).
			cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
		else:
			# For θ > π - m the angular boundary would exceed π, so fall back
			# to the linear lower bound: cos(θ) - mm.
			cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
									  cos_theta - self.mm)

		# Replace logit of the ground-truth class with cos(θ + m).
		one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).to(embeddings.dtype)
		logits = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)

		return F.cross_entropy(logits, labels)

	def extra_repr(self) -> str:
		return (
			f"num_classes={self.num_classes}, embedding_dim={self.embedding_dim}, "
			f"k={self.k}, s={self.s}, m={self.m:.4f} ({math.degrees(self.m):.1f}°), "
			f"easy_margin={self.easy_margin}"
		)
