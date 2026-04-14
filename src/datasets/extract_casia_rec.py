from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class LstEntry:
    rel_path: str
    identity_id: int


def _normalize_rel_path(path_token: str) -> str:
    normalized = path_token.replace("\\", "/")
    marker = "CASIA-WebFace/"
    marker_idx = normalized.find(marker)
    if marker_idx != -1:
        return normalized[marker_idx + len(marker) :]

    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return normalized


def parse_lst(lst_path: Path) -> list[LstEntry]:
    entries: list[LstEntry] = []
    with lst_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Invalid train.lst line {line_no}: {line}")

            source_path = fields[1]
            identity_id = int(fields[2])
            rel_path = _normalize_rel_path(source_path)
            entries.append(LstEntry(rel_path=rel_path, identity_id=identity_id))

    if not entries:
        raise ValueError(f"No entries found in {lst_path}")

    return entries


def _load_recordio_module():
    try:
        import mxnet as mx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "mxnet is required to extract .rec files. "
            "Create and use the dedicated extraction env: "
            "conda env create -f environment.extract.yml && conda activate dl-project-extract"
        ) from exc
    return mx


def _decode_record_to_pil(mx_module, record: bytes) -> Image.Image:
    _, image_bin = mx_module.recordio.unpack(record)
    image_nd = mx_module.image.imdecode(image_bin)
    image_np = image_nd.asnumpy()
    image = Image.fromarray(image_np)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CASIA-WebFace images from train.rec/train.idx using train.lst metadata."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data") / "casia-webface",
        help="Folder containing train.rec, train.idx and train.lst",
    )
    parser.add_argument(
        "--rec-file",
        type=Path,
        default=None,
        help="Optional explicit path to .rec file",
    )
    parser.add_argument(
        "--idx-file",
        type=Path,
        default=None,
        help="Optional explicit path to .idx file",
    )
    parser.add_argument(
        "--lst-file",
        type=Path,
        default=None,
        help="Optional explicit path to train.lst file",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output folder for extracted images. Default: --data-root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N samples (debug)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite images that already exist",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_root = args.data_root
    rec_path = args.rec_file or (data_root / "train.rec")
    idx_path = args.idx_file or (data_root / "train.idx")
    lst_path = args.lst_file or (data_root / "train.lst")
    output_root = args.output_root or data_root

    if not rec_path.exists():
        raise FileNotFoundError(f"Missing rec file: {rec_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing idx file: {idx_path}")
    if not lst_path.exists():
        raise FileNotFoundError(f"Missing lst file: {lst_path}")

    entries = parse_lst(lst_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be > 0, got {args.limit}")
        entries = entries[: args.limit]

    mx = _load_recordio_module()
    record_reader = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")

    output_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_existing = 0
    decode_failures = 0

    for idx, entry in tqdm(
        enumerate(entries),
        total=len(entries),
        desc="Extracting CASIA images",
        unit="img",
    ):
        # Preserve the relative image path declared in train.lst so the dataset loader
        # can resolve files exactly as recorded in the metadata.
        output_path = output_root / entry.rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        # RecordIO stores a header at index 0; actual samples start at index 1.
        record = record_reader.read_idx(idx + 1)
        if record is None:
            decode_failures += 1
            continue

        try:
            image = _decode_record_to_pil(mx, record)
            image.save(output_path)
            saved += 1
        except Exception:
            decode_failures += 1

    print("Extraction complete.")
    print(f"data_root: {data_root}")
    print(f"output_root: {output_root}")
    print(f"requested_entries: {len(entries)}")
    print(f"saved: {saved}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"decode_failures: {decode_failures}")

    if decode_failures > 0:
        print("Warning: some records could not be decoded. Consider retrying with --overwrite.")


if __name__ == "__main__":
    main()
