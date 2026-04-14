from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class CasiaSample:
    """Single sample metadata parsed from CASIA-WebFace train.lst."""

    sample_index: int
    identity_id: int
    identity_name: str
    image_rel_path: str
    source_path: str


def _sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_relative_image_path(path_token: str) -> str:
    normalized = path_token.replace("\\", "/")
    marker = "CASIA-WebFace/"
    marker_idx = normalized.find(marker)
    if marker_idx != -1:
        return normalized[marker_idx + len(marker) :]

    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return normalized


class CasiaWebFaceParser:
    """Parser and split utility for CASIA-WebFace metadata files."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.lst_file = self.data_root / "train.lst"
        self.property_file = self.data_root / "property"

        if not self.lst_file.exists():
            raise FileNotFoundError(f"Missing CASIA lst file: {self.lst_file}")

    def parse_samples(self) -> List[CasiaSample]:
        samples: List[CasiaSample] = []
        with self.lst_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                fields = line.split()
                if len(fields) < 3:
                    raise ValueError(
                        f"Invalid format in {self.lst_file} at line {line_number}: {line}"
                    )

                # Official CASIA lst format: aligned_flag, image_path, identity_id, ...
                source_path = fields[1]
                identity_id = int(fields[2])
                image_rel_path = _extract_relative_image_path(source_path)
                identity_name = image_rel_path.split("/", 1)[0]

                samples.append(
                    CasiaSample(
                        sample_index=len(samples),
                        identity_id=identity_id,
                        identity_name=identity_name,
                        image_rel_path=image_rel_path,
                        source_path=source_path,
                    )
                )

        return samples

    @staticmethod
    def _group_by_identity(samples: Iterable[CasiaSample]) -> Dict[int, List[CasiaSample]]:
        grouped: Dict[int, List[CasiaSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.identity_id].append(sample)
        return dict(grouped)

    @staticmethod
    def _compute_identity_split_sizes(
        total_identities: int, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> tuple[int, int, int]:
        if total_identities < 3:
            raise ValueError(
                "Need at least 3 identities to create train/val/test disjoint splits."
            )

        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Ratios must sum to 1.0, got {ratio_sum:.6f} "
                f"({train_ratio}, {val_ratio}, {test_ratio})."
            )

        train_n = max(1, int(round(total_identities * train_ratio)))
        val_n = max(1, int(round(total_identities * val_ratio)))
        test_n = total_identities - train_n - val_n

        if test_n <= 0:
            # Rebalance to guarantee all three non-empty splits.
            test_n = 1
            if train_n >= val_n and train_n > 1:
                train_n -= 1
            else:
                val_n -= 1

        if min(train_n, val_n, test_n) <= 0:
            raise ValueError(
                "Unable to create non-empty disjoint splits with the requested ratios."
            )

        return train_n, val_n, test_n

    def create_identity_disjoint_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
        min_images_per_identity: int = 2,
    ) -> dict:
        samples = self.parse_samples()
        grouped = self._group_by_identity(samples)

        eligible_identity_ids = sorted(
            identity_id
            for identity_id, identity_samples in grouped.items()
            if len(identity_samples) >= min_images_per_identity
        )

        if len(eligible_identity_ids) < 3:
            raise ValueError(
                "Not enough identities after filtering by min_images_per_identity. "
                f"Found {len(eligible_identity_ids)}."
            )

        rng = random.Random(seed)
        rng.shuffle(eligible_identity_ids)

        train_n, val_n, test_n = self._compute_identity_split_sizes(
            total_identities=len(eligible_identity_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_ids = sorted(eligible_identity_ids[:train_n])
        val_ids = sorted(eligible_identity_ids[train_n : train_n + val_n])
        test_ids = sorted(eligible_identity_ids[train_n + val_n : train_n + val_n + test_n])

        split_sets = {
            "train": set(train_ids),
            "val": set(val_ids),
            "test": set(test_ids),
        }

        # Safety check: no shared identities across splits.
        if split_sets["train"] & split_sets["val"]:
            raise RuntimeError("train/val identity overlap detected")
        if split_sets["train"] & split_sets["test"]:
            raise RuntimeError("train/test identity overlap detected")
        if split_sets["val"] & split_sets["test"]:
            raise RuntimeError("val/test identity overlap detected")

        sample_counts = {
            split_name: sum(len(grouped[idx]) for idx in ids)
            for split_name, ids in split_sets.items()
        }

        return {
            "schema_version": "1.0",
            "dataset": "CASIA-WebFace",
            "split_strategy": "identity_disjoint",
            "source": {
                "data_root": str(self.data_root.as_posix()),
                "train_lst": str(self.lst_file.as_posix()),
                "train_lst_sha256": _sha256_file(self.lst_file),
                "property": str(self.property_file.as_posix()) if self.property_file.exists() else None,
            },
            "reproducibility": {
                "seed": seed,
                "ratios": {
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio,
                },
                "min_images_per_identity": min_images_per_identity,
            },
            "stats": {
                "total_samples": len(samples),
                "eligible_identities": len(eligible_identity_ids),
                "identities_per_split": {
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "test": len(test_ids),
                },
                "samples_per_split": sample_counts,
            },
            "splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
        }


def _default_output_path(version: str) -> Path:
    return Path("data") / "splits" / f"casia_identity_split_{version}.json"


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse CASIA-WebFace metadata and generate reproducible identity-disjoint "
            "train/val/test splits."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data") / "casia-webface",
        help="Path containing CASIA metadata files (train.lst, property, ...).",
    )
    parser.add_argument("--version", type=str, default="v1", help="Split version tag.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train identity ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation identity ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test identity ratio.")
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=2,
        help="Keep only identities with at least this many samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to data/splits/casia_identity_split_<version>.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output split file.",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    output_path = args.output if args.output is not None else _default_output_path(args.version)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Split file already exists: {output_path}. Use --overwrite to replace it."
        )

    casia_parser = CasiaWebFaceParser(data_root=args.data_root)
    split_payload = casia_parser.create_identity_disjoint_split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_images_per_identity=args.min_images_per_identity,
    )

    split_payload["version"] = args.version

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)
        handle.write("\n")

    print(f"Saved split: {output_path}")
    print(
        "Identity counts => "
        f"train: {split_payload['stats']['identities_per_split']['train']}, "
        f"val: {split_payload['stats']['identities_per_split']['val']}, "
        f"test: {split_payload['stats']['identities_per_split']['test']}"
    )


if __name__ == "__main__":
    main()
