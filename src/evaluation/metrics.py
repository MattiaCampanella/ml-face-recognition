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


def retrieval_map_at_k(similarity_matrix: np.ndarray, targets: np.ndarray, topk: Iterable[int]) -> RetrievalMetrics:
	if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
		raise ValueError("similarity_matrix must be square.")
	if similarity_matrix.shape[0] != targets.shape[0]:
		raise ValueError("targets must have the same number of rows as similarity_matrix.")

	ks = sorted({int(k) for k in topk if int(k) > 0})
	if not ks:
		raise ValueError("topk must contain at least one positive integer.")

	map_totals = {k: 0.0 for k in ks}
	relevant_counts: list[int] = []
	queries = 0

	for query_index in range(similarity_matrix.shape[0]):
		scores = similarity_matrix[query_index].copy()
		scores[query_index] = -np.inf
		order = np.argsort(scores)[::-1]
		relevance = (targets[order] == targets[query_index]).astype(np.int32)
		num_relevant = int(relevance.sum())
		if num_relevant == 0:
			continue

		queries += 1
		relevant_counts.append(num_relevant)
		for k in ks:
			map_totals[k] += average_precision_at_k(relevance, k)

	map_values = {k: (map_totals[k] / queries if queries > 0 else 0.0) for k in ks}
	mean_relevant = float(np.mean(relevant_counts)) if relevant_counts else 0.0
	return RetrievalMetrics(map_at_k=map_values, queries_evaluated=queries, mean_relevant_per_query=mean_relevant)
