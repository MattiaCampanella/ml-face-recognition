from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.make_split import CasiaSample, CasiaWebFaceParser


@dataclass(frozen=True)
class FaceSample:
	image_path: Path
	identity_id: int
	label: int
	identity_name: str


def _resolve_path(path_value: str | Path, root: Path) -> Path:
	path = Path(path_value)
	if path.is_absolute():
		return path
	return root / path


def _resolve_image_path(data_root: Path, sample: CasiaSample) -> Path:
	canonical_path = data_root / sample.image_rel_path
	if canonical_path.exists():
		return canonical_path

	legacy_path = data_root / f"{sample.identity_id:06d}" / Path(sample.image_rel_path).name
	if legacy_path.exists():
		return legacy_path

	return canonical_path


def build_image_transforms(image_size: int, train: bool) -> transforms.Compose:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	if train:
		return transforms.Compose(
			[
				transforms.Resize((image_size + 32, image_size + 32)),
				transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			]
		)

	return transforms.Compose(
		[
			transforms.Resize((image_size + 32, image_size + 32)),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


class CasiaFaceDataset(Dataset):
	"""PyTorch dataset built from the CASIA-WebFace metadata split."""

	def __init__(
		self,
		data_root: str | Path,
		split_file: str | Path,
		split_name: str,
		*,
		image_size: int = 224,
		train: bool = False,
		label_mapping: Optional[dict[int, int]] = None,
		drop_unmapped_labels: bool = False,
	) -> None:
		self.data_root = Path(data_root)
		self.split_file = Path(split_file)
		if not self.split_file.is_absolute():
			self.split_file = Path.cwd() / self.split_file

		with self.split_file.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)

		if split_name not in payload.get("splits", {}):
			raise KeyError(f"Split '{split_name}' not found in {self.split_file}.")

		parser = CasiaWebFaceParser(self.data_root)
		samples = parser.parse_samples()
		split_identity_ids = set(payload["splits"][split_name])
		selected_samples = [sample for sample in samples if sample.identity_id in split_identity_ids]

		self.samples: list[FaceSample] = []
		missing_images = 0
		for sample in selected_samples:
			if label_mapping is None:
				label = sample.identity_id
			elif sample.identity_id in label_mapping:
				label = label_mapping[sample.identity_id]
			elif drop_unmapped_labels:
				continue
			else:
				label = sample.identity_id
			resolved_image_path = _resolve_image_path(self.data_root, sample)
			if not resolved_image_path.exists():
				missing_images += 1
				continue
			self.samples.append(
				FaceSample(
					image_path=resolved_image_path,
					identity_id=sample.identity_id,
					label=label,
					identity_name=sample.identity_name,
				)
			)

		if not self.samples:
			raise ValueError(f"No samples found for split '{split_name}'.")
		if missing_images > 0:
			print(
				f"[dataset:{split_name}] skipped {missing_images} samples with missing image files."
			)

		self.transform = build_image_transforms(image_size=image_size, train=train)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		sample = self.samples[index]
		with Image.open(sample.image_path) as image:
			image = image.convert("RGB")
		return {
			"images": self.transform(image),
			"labels": sample.label,
			"identity_id": sample.identity_id,
			"identity_name": sample.identity_name,
			"image_path": str(sample.image_path),
		}


def build_train_label_mapping(split_file: str | Path) -> dict[int, int]:
	path = Path(split_file)
	if not path.is_absolute():
		path = Path.cwd() / path
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	train_ids = payload["splits"]["train"]
	return {identity_id: idx for idx, identity_id in enumerate(sorted(train_ids))}
