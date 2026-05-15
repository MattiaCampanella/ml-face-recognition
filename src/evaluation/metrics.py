from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class RetrievalMetrics:
	map_at_k: dict[int, float]
	queries_evaluated: int
	mean_relevant_per_query: float


def average_precision_at_k(sorted_relevance: np.ndarray, k: int) -> float:
	truncated = sorted_relevance[:k].astype(bool)
	if not truncated.any():
		return 0.0

	precision_sum = 0.0
	hits = 0
	for rank, is_relevant in enumerate(truncated, start=1):
		if is_relevant:
			hits += 1
			precision_sum += hits / float(rank)

	denominator = min(int(sorted_relevance.sum()), k)
	if denominator <= 0:
		return 0.0
	return precision_sum / float(denominator)


def retrieval_map_at_k(similarity_matrix: torch.Tensor, targets: torch.Tensor, topk: Iterable[int]) -> RetrievalMetrics:
	if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
		raise ValueError("similarity_matrix must be square.")
	if similarity_matrix.shape[0] != targets.shape[0]:
		raise ValueError("targets must have the same number of rows as similarity_matrix.")

	ks = sorted({int(k) for k in topk if int(k) > 0})
	if not ks:
		raise ValueError("topk must contain at least one positive integer.")

	max_k = max(ks)
	num_samples = similarity_matrix.shape[0]
	
	# Disable gradients or detach
	similarity_matrix = similarity_matrix.detach()
	targets = targets.detach()
	
	# Temporarily mask self-similarity so query doesn't match itself
	original_diag = similarity_matrix.diag().clone()
	similarity_matrix.fill_diagonal_(float('-inf'))

	# Find top max_k probabilities and their indices on the same device as the similarity matrix
	_, topk_indices = torch.topk(similarity_matrix, k=min(max_k, num_samples - 1), dim=1)

	# Restore diagonal to preserve original matrix
	similarity_matrix.diagonal().copy_(original_diag)

	# Compute matched labels without CPU roundtrips
	_, inverse_indices, counts = torch.unique(targets, return_inverse=True, return_counts=True)
	total_relevant_per_query = counts[inverse_indices] - 1
	valid_queries_mask = total_relevant_per_query > 0
	queries = int(valid_queries_mask.sum().item())

	if queries == 0:
		map_values = {k: 0.0 for k in ks}
		return RetrievalMetrics(map_at_k=map_values, queries_evaluated=0, mean_relevant_per_query=0.0)

	topk_labels = targets[topk_indices]
	relevance = (topk_labels == targets.unsqueeze(1))

	max_k_eff = topk_indices.size(1)
	relevance_float = relevance.to(dtype=torch.float32)
	cumulative_hits = torch.cumsum(relevance_float, dim=1)
	ranks = torch.arange(1, max_k_eff + 1, device=relevance.device, dtype=torch.float32).view(1, -1)
	precision_at_rank = cumulative_hits / ranks
	precision_at_rank = precision_at_rank * relevance_float
	precision_cumsum = torch.cumsum(precision_at_rank, dim=1)

	denominator_base = total_relevant_per_query.to(dtype=torch.float32)
	map_values: dict[int, float] = {}
	for k in ks:
		k_val = int(k)
		k_eff = min(k_val, max_k_eff)
		if k_eff <= 0:
			map_values[k_val] = 0.0
			continue
		precision_sum = precision_cumsum[:, k_eff - 1]
		denominator = torch.clamp(denominator_base, max=float(k_val))
		ap = torch.zeros_like(denominator_base, dtype=torch.float32)
		ap[valid_queries_mask] = precision_sum[valid_queries_mask] / denominator[valid_queries_mask]
		map_values[k_val] = float(ap[valid_queries_mask].mean().item())

	mean_relevant = float(denominator_base[valid_queries_mask].mean().item())
	return RetrievalMetrics(
		map_at_k=map_values,
		queries_evaluated=queries,
		mean_relevant_per_query=mean_relevant,
	)
