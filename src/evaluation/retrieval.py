from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from src.evaluation.metrics import RetrievalMetrics, retrieval_map_at_k


@dataclass(frozen=True)
class RetrievalSearchResult:
	query_index: int
	query_label: int
	top_indices: list[int]
	top_labels: list[int]
	top_scores: list[float]


def _normalize_embeddings(embeddings: Tensor, *, l2_normalize: bool) -> Tensor:
	if l2_normalize:
		return torch.nn.functional.normalize(embeddings, dim=1)
	return embeddings


def pairwise_similarity(
	embeddings: Tensor,
	*,
	metric: str = "cosine",
	l2_normalize: bool = True,
) -> Tensor:
	metric_name = metric.lower()
	prepared = _normalize_embeddings(embeddings, l2_normalize=l2_normalize)

	if metric_name == "cosine":
		return prepared @ prepared.t()
	if metric_name == "euclidean":
		return -torch.cdist(prepared, prepared, p=2)
	raise ValueError(f"Unsupported retrieval metric: {metric}")


def retrieve_topk(
	embeddings: Tensor,
	targets: Tensor,
	*,
	topk: int = 10,
	metric: str = "cosine",
	l2_normalize: bool = True,
	similarity: Optional[Tensor] = None,
	query_indices: Optional[Iterable[int]] = None,
	max_queries: Optional[int] = None,
) -> list[RetrievalSearchResult]:
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
	if targets.ndim != 1:
		targets = targets.view(-1)
	if embeddings.size(0) != targets.numel():
		raise ValueError("embeddings and targets must have the same number of rows.")
	if max_queries is not None and max_queries < 0:
		raise ValueError("max_queries must be >= 0 when provided.")

	if similarity is None:
		similarity = pairwise_similarity(embeddings, metric=metric, l2_normalize=l2_normalize)
	else:
		if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
			raise ValueError("similarity must be a square matrix.")
		if similarity.shape[0] != embeddings.shape[0]:
			raise ValueError("similarity must have the same number of rows as embeddings.")
	targets_np = targets.detach().cpu().numpy()
	results: list[RetrievalSearchResult] = []

	num_samples = similarity.size(0)
	if query_indices is None:
		query_indices_list = list(range(num_samples))
	else:
		query_indices_list = [int(index) for index in query_indices]
		for index in query_indices_list:
			if index < 0 or index >= num_samples:
				raise IndexError(f"query index {index} is out of range for {num_samples} samples.")
	if max_queries is not None:
		query_indices_list = query_indices_list[: int(max_queries)]
	if not query_indices_list:
		return results

	candidate_count = max(0, num_samples - 1)
	if candidate_count == 0:
		for query_index in query_indices_list:
			results.append(
				RetrievalSearchResult(
					query_index=query_index,
					query_label=int(targets_np[query_index]),
					top_indices=[],
					top_labels=[],
					top_scores=[],
				)
			)
		return results

	k = min(topk, candidate_count)
	if query_indices is None and max_queries is None:
		original_diag = similarity.diag().clone()
		similarity.fill_diagonal_(float("-inf"))
		values, indices = torch.topk(similarity, k=k, dim=1)
		similarity.diagonal().copy_(original_diag)

		indices_list = indices.detach().cpu().tolist()
		values_list = values.detach().cpu().tolist()

		for query_index in range(num_samples):
			top_indices = indices_list[query_index]
			top_scores = values_list[query_index]
			results.append(
				RetrievalSearchResult(
					query_index=query_index,
					query_label=int(targets_np[query_index]),
					top_indices=top_indices,
					top_labels=[int(targets_np[index]) for index in top_indices],
					top_scores=[float(score) for score in top_scores],
				)
			)
		return results

	query_tensor = torch.tensor(query_indices_list, device=similarity.device, dtype=torch.long)
	scores = similarity.index_select(0, query_tensor).clone()
	row_indices = torch.arange(query_tensor.numel(), device=similarity.device)
	scores[row_indices, query_tensor] = float("-inf")
	values, indices = torch.topk(scores, k=k, dim=1)

	indices_list = indices.detach().cpu().tolist()
	values_list = values.detach().cpu().tolist()

	for offset, query_index in enumerate(query_indices_list):
		top_indices = indices_list[offset]
		top_scores = values_list[offset]
		results.append(
			RetrievalSearchResult(
				query_index=query_index,
				query_label=int(targets_np[query_index]),
				top_indices=top_indices,
				top_labels=[int(targets_np[index]) for index in top_indices],
				top_scores=[float(score) for score in top_scores],
			)
		)

	return results


def evaluate_retrieval(
	embeddings: Tensor,
	targets: Tensor,
	*,
	topk: Iterable[int] = (1, 5, 10),
	metric: str = "cosine",
	l2_normalize: bool = True,
) -> tuple[RetrievalMetrics, Tensor]:
	if targets.ndim != 1:
		targets = targets.view(-1)

	prepared = _normalize_embeddings(embeddings.detach(), l2_normalize=l2_normalize)
	similarity = pairwise_similarity(prepared, metric=metric, l2_normalize=False)
	metrics = retrieval_map_at_k(similarity, targets, topk=topk)
	return metrics, similarity
