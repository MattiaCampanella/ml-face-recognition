from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class LstEntry:
    rel_path: str
    identity_id: int
    label_vector: tuple[float, ...]


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
            try:
                label_vector = tuple(float(value) for value in fields[2:])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric labels in {lst_path} at line {line_no}: {line}"
                ) from exc

            identity_id = int(label_vector[0])
            rel_path = _normalize_rel_path(source_path)
            entries.append(
                LstEntry(
                    rel_path=rel_path,
                    identity_id=identity_id,
                    label_vector=label_vector,
                )
            )

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


def _flatten_label_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened: list[object] = []
        for item in value:
            flattened.extend(_flatten_label_values(item))
        return flattened
    return [value]


def _normalize_record_label(label: object) -> tuple[float, ...]:
    if hasattr(label, "tolist"):
        try:
            label = label.tolist()
        except Exception:
            pass
    flattened = _flatten_label_values(label)
    return tuple(float(item) for item in flattened if item is not None)


def _labels_match(record_label: tuple[float, ...], entry: LstEntry) -> bool:
    if not record_label:
        return False
    record_id = int(round(record_label[0]))
    if record_id != entry.identity_id:
        return False
    if len(record_label) == 1:
        return True
    if len(record_label) != len(entry.label_vector):
        return True
    for record_value, entry_value in zip(record_label, entry.label_vector):
        if abs(record_value - entry_value) > 1e-3:
            return False
    return True


def _read_record_keys(record_reader) -> list[int]:
    keys_attr = getattr(record_reader, "keys", None)
    if keys_attr is None:
        return []
    try:
        keys = keys_attr() if callable(keys_attr) else keys_attr
        key_values = [int(key) for key in keys]
    except Exception:
        return []
    return sorted(key for key in key_values if key != 0)


def _decode_record_to_pil(mx_module, record: bytes) -> tuple[object, Image.Image]:
    header, image_bin = mx_module.recordio.unpack(record)
    image_nd = mx_module.image.imdecode(image_bin)
    image_np = image_nd.asnumpy()
    image = Image.fromarray(image_np)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return header, image


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

    record_keys = _read_record_keys(record_reader)
    if not record_keys:
        raise RuntimeError(
            "Unable to read record keys from train.idx. "
            "Cannot safely align train.lst with RecordIO samples."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_existing = 0
    decode_failures = 0
    read_failures = 0
    matched_entries = 0
    skipped_entries = 0
    processed_records = 0
    empty_label_records = 0

    skipped_entry_samples: list[dict[str, object]] = []
    unmatched_record_samples: list[dict[str, object]] = []
    sample_limit = 50

    entry_idx = 0

    for record_key in tqdm(
        record_keys,
        total=len(record_keys),
        desc="Extracting CASIA images",
        unit="rec",
    ):
        if entry_idx >= len(entries):
            break

        processed_records += 1
        record = record_reader.read_idx(record_key)
        if record is None:
            read_failures += 1
            continue

        try:
            header, image = _decode_record_to_pil(mx, record)
        except Exception:
            decode_failures += 1
            continue

        record_label = _normalize_record_label(getattr(header, "label", None))
        if not record_label and entry_idx < len(entries):
            empty_label_records += 1
            record_label = (float(entries[entry_idx].identity_id),)
        while entry_idx < len(entries) and not _labels_match(record_label, entries[entry_idx]):
            skipped_entries += 1
            if len(skipped_entry_samples) < sample_limit:
                skipped_entry_samples.append(
                    {
                        "entry_index": entry_idx,
                        "identity_id": entries[entry_idx].identity_id,
                        "rel_path": entries[entry_idx].rel_path,
                    }
                )
            entry_idx += 1

        if entry_idx >= len(entries):
            if len(unmatched_record_samples) < sample_limit:
                unmatched_record_samples.append(
                    {
                        "record_key": int(record_key),
                        "record_label": record_label,
                    }
                )
            break

        entry = entries[entry_idx]
        matched_entries += 1

        # Preserve the relative image path declared in train.lst so the dataset loader
        # can resolve files exactly as recorded in the metadata.
        output_path = output_root / entry.rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        else:
            try:
                image.save(output_path)
                saved += 1
            except Exception:
                decode_failures += 1

        entry_idx += 1

    remaining_records = len(record_keys) - processed_records
    remaining_entries = len(entries) - entry_idx

    report = {
        "data_root": str(data_root),
        "rec_file": str(rec_path),
        "idx_file": str(idx_path),
        "lst_file": str(lst_path),
        "output_root": str(output_root),
        "lst_entries": len(entries),
        "record_keys": len(record_keys),
        "processed_records": processed_records,
        "matched_entries": matched_entries,
        "saved": saved,
        "skipped_existing": skipped_existing,
        "read_failures": read_failures,
        "decode_failures": decode_failures,
        "skipped_entries": skipped_entries,
        "empty_label_records": empty_label_records,
        "remaining_entries": remaining_entries,
        "remaining_records": remaining_records,
        "limit": args.limit,
        "sample_skipped_entries": skipped_entry_samples,
        "sample_unmatched_records": unmatched_record_samples,
    }

    report_path = output_root / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Extraction complete.")
    print(f"data_root: {data_root}")
    print(f"output_root: {output_root}")
    print(f"requested_entries: {len(entries)}")
    print(f"record_keys: {len(record_keys)}")
    print(f"matched_entries: {matched_entries}")
    print(f"saved: {saved}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"read_failures: {read_failures}")
    print(f"decode_failures: {decode_failures}")
    print(f"skipped_entries: {skipped_entries}")
    print(f"empty_label_records: {empty_label_records}")
    print(f"remaining_entries: {remaining_entries}")
    print(f"remaining_records: {remaining_records}")
    print(f"report: {report_path}")

    if decode_failures > 0:
        print("Warning: some records could not be decoded. Consider retrying with --overwrite.")


if __name__ == "__main__":
    main()
