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
	
	# Compute matched labels
	# Avoid O(N^2) memory footprint by counting global target occurrences instead of an NxN mask
	unique_labels, counts = torch.unique(targets, return_counts=True)
	label_counts = {k.item(): v.item() for k, v in zip(unique_labels, counts)}
	# For each query, the number of relevant items is the total count of that label minus 1 (itself)
	total_relevant_per_query = torch.tensor(
		[label_counts[t.item()] - 1 for t in targets],
		device=targets.device
	)
	
	topk_labels = targets[topk_indices]
	relevance = (topk_labels == targets.unsqueeze(1))
	
	# Move to CPU for the sequence evaluation
	relevance_np = relevance.cpu().numpy()
	total_relevant_np = total_relevant_per_query.cpu().numpy()
	
	map_totals = {k: 0.0 for k in ks}
	queries = 0
	relevant_counts: list[int] = []

	for query_index in range(num_samples):
		rel_count = int(total_relevant_np[query_index])
		if rel_count == 0:
			continue

		queries += 1
		relevant_counts.append(rel_count)
		query_relevance = relevance_np[query_index]

		for k in ks:
			truncated = query_relevance[:k]
			if not truncated.any():
				continue

			hits = 0.0
			precision_sum = 0.0
			for rank, is_relevant in enumerate(truncated, start=1):
				if is_relevant:
					hits += 1.0
					precision_sum += hits / rank

			denominator = min(rel_count, k)
			if denominator > 0:
				map_totals[k] += precision_sum / float(denominator)

	map_values = {k: (map_totals[k] / queries if queries > 0 else 0.0) for k in ks}
	mean_relevant = float(np.mean(relevant_counts)) if relevant_counts else 0.0
	return RetrievalMetrics(map_at_k=map_values, queries_evaluated=queries, mean_relevant_per_query=mean_relevant)
