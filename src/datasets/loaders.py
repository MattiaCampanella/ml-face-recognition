from __future__ import annotations

import random
from collections import defaultdict
from math import ceil
from typing import Iterable, Sequence

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
	"""Sample P identities and K images per identity for metric learning batches."""

	def __init__(
		self,
		labels: Sequence[int] | Iterable[int],
		p: int,
		k: int,
		*,
		shuffle: bool = True,
		drop_last: bool = True,
	) -> None:
		if p <= 0 or k <= 0:
			raise ValueError(f"p and k must be positive, got p={p}, k={k}.")

		self.p = int(p)
		self.k = int(k)
		self.shuffle = shuffle
		self.drop_last = drop_last

		bucketed: dict[int, list[int]] = defaultdict(list)
		label_list = list(labels)
		for index, label in enumerate(label_list):
			bucketed[int(label)].append(index)

		self.class_to_indices = {label: indices for label, indices in bucketed.items() if indices}
		self.labels = sorted(self.class_to_indices)
		if len(self.labels) < self.p:
			raise ValueError(
				f"PKBatchSampler requires at least p identities, got {len(self.labels)} classes and p={self.p}."
			)

		self._num_batches = max(1, ceil(len(label_list) / float(self.p * self.k)))

	def __len__(self) -> int:
		return self._num_batches

	def _sample_indices_for_label(self, label: int) -> list[int]:
		indices = self.class_to_indices[label]
		if len(indices) >= self.k:
			return random.sample(indices, self.k)
		return [random.choice(indices) for _ in range(self.k)]

	def __iter__(self):
		for _ in range(self._num_batches):
			chosen_labels = random.sample(self.labels, self.p) if self.shuffle else self.labels[: self.p]
			batch: list[int] = []
			for label in chosen_labels:
				batch.extend(self._sample_indices_for_label(label))
			yield batch
