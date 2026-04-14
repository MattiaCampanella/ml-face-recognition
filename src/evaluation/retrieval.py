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
) -> list[RetrievalSearchResult]:
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
	if targets.ndim != 1:
		targets = targets.view(-1)

	similarity = pairwise_similarity(embeddings, metric=metric, l2_normalize=l2_normalize)
	targets_np = targets.detach().cpu().numpy()
	results: list[RetrievalSearchResult] = []

	for query_index in range(similarity.size(0)):
		scores = similarity[query_index].clone()
		scores[query_index] = float("-inf")
		candidate_count = max(0, scores.numel() - 1)
		if candidate_count == 0:
			results.append(
				RetrievalSearchResult(
					query_index=query_index,
					query_label=int(targets_np[query_index]),
					top_indices=[],
					top_labels=[],
					top_scores=[],
				)
			)
			continue

		values, indices = torch.topk(scores, k=min(topk, candidate_count))
		indices_list = indices.detach().cpu().tolist()
		results.append(
			RetrievalSearchResult(
				query_index=query_index,
				query_label=int(targets_np[query_index]),
				top_indices=indices_list,
				top_labels=[int(targets_np[index]) for index in indices_list],
				top_scores=[float(score) for score in values.detach().cpu().tolist()],
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
	similarity_np = similarity.cpu().numpy()
	targets_np = targets.detach().cpu().numpy()
	metrics = retrieval_map_at_k(similarity_np, targets_np, topk=topk)
	return metrics, similarity
