"""Clustering strategies — pure numpy + sklearn, no PyTorch dependency."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


class ClusterStrategy(ABC):
    """Base class for clustering strategies."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign cluster labels to embeddings. Returns (N,) int array."""
        ...


class DBSCANStrategy(ClusterStrategy):
    def __init__(self, eps: float = 0.25, min_samples: int = 2) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)

        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        return model.fit_predict(embeddings)


class AgglomerativeStrategy(ClusterStrategy):
    def __init__(self, threshold: float = 0.25, linkage: str = "average") -> None:
        self.threshold = threshold
        self.linkage = linkage

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)

        # Compute cosine distance matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normalized = embeddings / norms
        similarity = normalized @ normalized.T
        distance = np.clip(1.0 - similarity, 0.0, None)
        np.fill_diagonal(distance, 0.0)

        kwargs = {
            "n_clusters": None,
            "distance_threshold": self.threshold,
            "linkage": self.linkage,
        }
        # Handle sklearn API change
        if "metric" in AgglomerativeClustering.__init__.__code__.co_varnames:
            kwargs["metric"] = "precomputed"
        else:
            kwargs["affinity"] = "precomputed"

        model = AgglomerativeClustering(**kwargs)
        return model.fit_predict(distance)


def get_strategy(algorithm: str, **params) -> ClusterStrategy:
    """Factory for clustering strategies."""
    if algorithm == "dbscan":
        return DBSCANStrategy(
            eps=params.get("eps", 0.25),
            min_samples=params.get("min_samples", 2),
        )
    elif algorithm == "agglomerative":
        return AgglomerativeStrategy(
            threshold=params.get("threshold", 0.25),
            linkage=params.get("linkage", "average"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
