from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.face_dataset import CasiaFaceDataset, build_train_label_mapping
from src.evaluation.clustering import extract_embeddings
from src.evaluation.retrieval import evaluate_retrieval, retrieve_topk
from src.models.resnet18 import build_baseline_resnet18
from src.utils.config import ensure_dir, load_yaml_config, resolve_repo_path


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract embeddings and compute retrieval metrics on a trained checkpoint."
	)
	parser.add_argument("--config", type=str, default="experiments/configs/base.yaml")
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--split", type=str, default="val")
	parser.add_argument("--output-dir", type=str, default=None)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--topk", type=int, nargs="+", default=None)
	return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
	if device_str == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_str)


def _find_latest_checkpoint(runs_root: Path) -> Path:
	candidates = list(runs_root.glob("*/checkpoints/best.pt")) or list(runs_root.glob("*/checkpoints/last.pt"))
	if not candidates:
		raise FileNotFoundError(
			f"No checkpoint found under {runs_root}. Train a run first or pass --checkpoint."
		)
	return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _infer_output_dir(checkpoint_path: Path) -> Path:
	return checkpoint_path.parent.parent / "artifacts" / "retrieval"


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	model.load_state_dict(state_dict)


def _select_targets(labels: torch.Tensor, identity_ids: torch.Tensor | None) -> torch.Tensor:
	if identity_ids is not None:
		return identity_ids
	return labels


def main() -> None:
	args = _parse_args()
	config = load_yaml_config(args.config).data
	device = _resolve_device(args.device)

	data_cfg = config["data"]
	data_root = resolve_repo_path(data_cfg["root_dir"])
	split_file = resolve_repo_path(data_cfg["split_file"])
	image_size = int(data_cfg.get("image_size", 224))

	label_mapping = build_train_label_mapping(split_file)
	dataset = CasiaFaceDataset(
		data_root=data_root,
		split_file=split_file,
		split_name=args.split,
		image_size=image_size,
		train=False,
		label_mapping=label_mapping,
		drop_unmapped_labels=False,
	)
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	model_cfg = config["model"]
	classifier_cfg = model_cfg.get("classifier_head", {})
	num_classes = int(classifier_cfg.get("num_classes") or len(label_mapping))
	classifier_enabled = bool(classifier_cfg.get("enabled", True))
	model = build_baseline_resnet18(
		pretrained=bool(model_cfg.get("pretrained", True)),
		embedding_dim=int(model_cfg.get("embedding_dim", 512)),
		normalize_embeddings=bool(model_cfg.get("normalize_embeddings", True)),
		classifier_num_classes=(num_classes if classifier_enabled else None),
	).to(device)

	runs_root = resolve_repo_path(config.get("output", {}).get("root_dir", "experiments/runs"))
	checkpoint_path = resolve_repo_path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint(runs_root)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	artifact_dir = ensure_dir(Path(args.output_dir) if args.output_dir else _infer_output_dir(checkpoint_path))
	_load_checkpoint(model, checkpoint_path, device)

	embeddings, labels, identity_ids = extract_embeddings(
		model,
		dataloader,
		device,
		return_labels=True,
		return_identity_ids=True,
		amp_enabled=bool(config.get("system", {}).get("amp", False)),
	)
	targets = _select_targets(labels, identity_ids)
	topk = args.topk if args.topk else list(config.get("retrieval_eval", {}).get("topk", [1, 5, 10]))
	retrieval_cfg = config.get("retrieval_eval", {})
	metric_name = str(retrieval_cfg.get("distance", "cosine"))
	l2_normalize = bool(retrieval_cfg.get("l2_normalize", True))

	metrics, similarity = evaluate_retrieval(
		embeddings,
		targets,
		topk=topk,
		metric=metric_name,
		l2_normalize=l2_normalize,
	)
	results = retrieve_topk(
		embeddings,
		targets,
		topk=max(topk),
		metric=metric_name,
		l2_normalize=l2_normalize,
	)

	targets_np = targets.detach().cpu().numpy()
	summary = {
		"config": str(resolve_repo_path(args.config)),
		"checkpoint": str(checkpoint_path),
		"split": args.split,
		"num_samples": int(len(dataset)),
		"num_identities": int(len(np.unique(targets_np))),
		"embedding_dim": int(embeddings.shape[1]),
		"metric": metric_name,
		"l2_normalize": l2_normalize,
		"topk": [int(k) for k in topk],
		"map_at_k": {str(k): float(v) for k, v in metrics.map_at_k.items()},
		"queries_evaluated": int(metrics.queries_evaluated),
		"mean_relevant_per_query": float(metrics.mean_relevant_per_query),
		"similarity_matrix_shape": list(similarity.shape),
	}

	metrics_path = artifact_dir / f"retrieval_metrics_{args.split}.json"
	metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

	examples_path = artifact_dir / f"retrieval_examples_{args.split}.json"
	examples_payload = [
		{
			"query_index": result.query_index,
			"query_label": result.query_label,
			"top_indices": result.top_indices,
			"top_labels": result.top_labels,
			"top_scores": result.top_scores,
		}
		for result in results[: min(20, len(results))]
	]
	examples_path.write_text(json.dumps(examples_payload, indent=2) + "\n", encoding="utf-8")

	print(f"Retrieval evaluation complete. Artifacts saved to {artifact_dir}")
	print(json.dumps(summary["map_at_k"], indent=2))


if __name__ == "__main__":
	main()