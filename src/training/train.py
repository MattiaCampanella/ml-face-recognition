from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.face_dataset import CasiaFaceDataset, build_train_label_mapping
from src.datasets.loaders import PKBatchSampler
from src.models.resnet18 import build_baseline_resnet18
from src.training.trainer import train_supervised, train_triplet_learning
from src.utils.config import ensure_dir, load_yaml_config, make_run_name, resolve_repo_path, save_json
from src.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the baseline ResNet-18 face classifier.")
	parser.add_argument("--config", type=str, default="experiments/configs/base.yaml")
	return parser.parse_args()


def _build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
	optimizer_cfg = config["train"]["optimizer"]
	name = optimizer_cfg.get("name", "adamw").lower()
	lr = float(optimizer_cfg.get("lr", 3e-4))
	weight_decay = float(optimizer_cfg.get("weight_decay", 1e-4))

	if name == "adamw":
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	if name == "adam":
		return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
	scheduler_cfg = config["train"].get("scheduler", {})
	name = scheduler_cfg.get("name", "cosine").lower()
	params = scheduler_cfg.get("params", {})

	if name == "cosine":
		t_max = int(params.get("t_max", config["train"]["epochs"]))
		min_lr = float(params.get("min_lr", 1e-6))
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
	if name in {"none", ""}:
		return None
	raise ValueError(f"Unsupported scheduler: {name}")


def _resolve_data_root(config: dict) -> Path:
	data_root = resolve_repo_path(config["data"]["root_dir"])
	if data_root.exists():
		return data_root
	fallback = resolve_repo_path(Path("data") / "casia-webface")
	if fallback.exists():
		return fallback
	raise FileNotFoundError(
		f"Could not find CASIA-WebFace data root at {data_root} or fallback {fallback}."
	)


def _resolve_split_file(config: dict) -> Path:
	split_file = resolve_repo_path(config["data"]["split_file"])
	if split_file.exists():
		return split_file

	if split_file.suffix.lower() == ".yaml":
		json_fallback = split_file.with_suffix(".json")
		if json_fallback.exists():
			return json_fallback

	raise FileNotFoundError(f"Missing split file: {split_file}")


def main() -> None:
	args = _parse_args()
	loaded = load_yaml_config(args.config)
	config = loaded.data
	loss_cfg = config.get("loss", {})
	loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()

	seed_cfg = config.get("system", {})
	set_seed(
		seed=int(seed_cfg.get("seed", 42)),
		deterministic=bool(seed_cfg.get("deterministic", True)),
		benchmark=bool(seed_cfg.get("benchmark", False)),
	)

	data_root = _resolve_data_root(config)
	split_file = _resolve_split_file(config)

	train_label_mapping = build_train_label_mapping(split_file)
	train_dataset = CasiaFaceDataset(
		data_root=data_root,
		split_file=split_file,
		split_name="train",
		image_size=int(config["data"].get("image_size", 224)),
		train=True,
		label_mapping=train_label_mapping,
		drop_unmapped_labels=True,
	)
	val_dataset = CasiaFaceDataset(
		data_root=data_root,
		split_file=split_file,
		split_name="val",
		image_size=int(config["data"].get("image_size", 224)),
		train=False,
		label_mapping=train_label_mapping,
		drop_unmapped_labels=False,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=int(config["retrieval_eval"].get("batch_size", 128)),
		shuffle=False,
		num_workers=int(seed_cfg.get("num_workers", 4)),
		pin_memory=bool(seed_cfg.get("pin_memory", True)),
	)

	model_cfg = config["model"]
	triplet_cfg = loss_cfg.setdefault("params", {}) if loss_name == "triplet" else {}
	if loss_name == "triplet":
		# Keep a single normalization stage for triplet training.
		model_normalize_default = bool(model_cfg.get("normalize_embeddings", True))
		original_model_normalize = model_normalize_default
		triplet_normalize = bool(triplet_cfg.get("normalize_embeddings", model_normalize_default))
		model_cfg["normalize_embeddings"] = triplet_normalize
		triplet_cfg["normalize_embeddings"] = False
		if original_model_normalize != triplet_normalize:
			print(
				"[train] Overriding model.normalize_embeddings to match triplet normalization setting."
			)
	classifier_cfg = model_cfg.get("classifier_head", {})
	num_classes = int(classifier_cfg.get("num_classes") or len(train_label_mapping))
	classifier_enabled = bool(classifier_cfg.get("enabled", True))
	if loss_name == "triplet":
		classifier_enabled = False

	model = build_baseline_resnet18(
		pretrained=bool(model_cfg.get("pretrained", True)),
		embedding_dim=int(model_cfg.get("embedding_dim", 512)),
		normalize_embeddings=bool(model_cfg.get("normalize_embeddings", True)),
		classifier_num_classes=(num_classes if classifier_enabled else None),
	)

	optimizer = _build_optimizer(model, config)
	scheduler = _build_scheduler(optimizer, config)
	output_cfg = config.get("output", {})
	run_root = ensure_dir(output_cfg.get("root_dir", "experiments/runs"))
	run_name = make_run_name(config)
	run_dir = ensure_dir(run_root / run_name)
	checkpoints_dir = ensure_dir(run_dir / output_cfg.get("dirs", {}).get("checkpoints", "checkpoints"))
	grad_clip_cfg = config["train"].get("grad_clip", {})
	grad_clip_max_norm = (
		float(grad_clip_cfg.get("max_norm", 1.0)) if bool(grad_clip_cfg.get("enabled", False)) else None
	)
	checkpoint_cfg = config.get("checkpoint", {})
	monitor = str(checkpoint_cfg.get("monitor", "train_loss"))
	monitor_mode = str(checkpoint_cfg.get("mode", "min"))
	if monitor == "train_loss":
		monitor = "val_map_at_5"
		monitor_mode = "max"
	config["checkpoint"]["monitor"] = monitor
	config["checkpoint"]["mode"] = monitor_mode
	save_json(run_dir / "resolved_config.json", config)
	epochs = int(config["train"].get("epochs", 30))
	log_every_steps = int(config["train"].get("log_every_steps", 50))
	device = seed_cfg.get("device", "auto")
	amp_enabled = bool(seed_cfg.get("amp", False))
	retrieval_cfg = config.get("retrieval_eval", {})
	val_topk = tuple(int(k) for k in retrieval_cfg.get("topk", [1, 5, 10]))
	val_metric = str(retrieval_cfg.get("distance", "cosine"))
	val_l2_normalize = bool(retrieval_cfg.get("l2_normalize", True))

	if loss_name == "triplet":
		sampler_cfg = config["train"].get("sampler", {})
		p = int(sampler_cfg.get("p", 16))
		k = int(sampler_cfg.get("k", 4))
		margin_curriculum_cfg = triplet_cfg.get("margin_curriculum", {})
		margin_schedule = str(margin_curriculum_cfg.get("schedule", "constant"))
		margin_start = float(margin_curriculum_cfg.get("start", triplet_cfg.get("margin", 0.2)))
		margin_end = float(margin_curriculum_cfg.get("end", triplet_cfg.get("margin", 0.2)))
		margin_warmup_epochs = int(margin_curriculum_cfg.get("warmup_epochs", 0))
		mining_curriculum_cfg = triplet_cfg.get("mining_curriculum", {})
		mining_phase1 = str(mining_curriculum_cfg.get("phase1", "easy_semi_hard"))
		mining_phase2 = str(mining_curriculum_cfg.get("phase2", "semi_hard"))
		mining_phase3 = mining_curriculum_cfg.get("phase3")
		mining_phase3 = None if mining_phase3 is None else str(mining_phase3)
		mining_phase1_epochs = int(
			mining_curriculum_cfg.get("phase1_epochs", mining_curriculum_cfg.get("warmup_epochs", 0))
		)
		mining_phase2_epochs = int(mining_curriculum_cfg.get("phase2_epochs", 0))
		mining_warmup_epochs = mining_phase1_epochs
		batch_sampler = PKBatchSampler(
			labels=[sample.label for sample in train_dataset.samples],
			p=p,
			k=k,
			shuffle=bool(sampler_cfg.get("shuffle", True)),
			drop_last=bool(sampler_cfg.get("drop_last", True)),
		)
		train_loader = DataLoader(
			train_dataset,
			batch_sampler=batch_sampler,
			num_workers=int(seed_cfg.get("num_workers", 4)),
			pin_memory=bool(seed_cfg.get("pin_memory", True)),
		)
		history = train_triplet_learning(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			epochs=epochs,
			optimizer=optimizer,
			scheduler=scheduler,
			device=device,
			margin=float(triplet_cfg.get("margin", 0.2)),
			margin_start=margin_start,
			margin_end=margin_end,
			margin_schedule=margin_schedule,
			margin_warmup_epochs=margin_warmup_epochs,
			normalize_embeddings=bool(triplet_cfg.get("normalize_embeddings", False)),
			mining_phase1_strategy=mining_phase1,
			mining_phase2_strategy=mining_phase2,
			mining_phase3_strategy=mining_phase3,
			mining_phase1_epochs=mining_phase1_epochs,
			mining_phase2_epochs=mining_phase2_epochs,
			mining_warmup_epochs=mining_warmup_epochs,
			amp_enabled=amp_enabled,
			grad_clip_max_norm=grad_clip_max_norm,
			log_every_steps=log_every_steps,
			checkpoint_dir=checkpoints_dir,
			monitor=monitor,
			monitor_mode=monitor_mode,
			val_retrieval_topk=val_topk,
			val_retrieval_metric=val_metric,
			val_retrieval_l2_normalize=val_l2_normalize,
		)
	else:
		train_loader = DataLoader(
			train_dataset,
			batch_size=int(config["train"].get("batch_size", 64)),
			shuffle=True,
			num_workers=int(seed_cfg.get("num_workers", 4)),
			pin_memory=bool(seed_cfg.get("pin_memory", True)),
		)
		history = train_supervised(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			epochs=epochs,
			optimizer=optimizer,
			scheduler=scheduler,
			criterion=torch.nn.CrossEntropyLoss(),
			device=device,
			amp_enabled=amp_enabled,
			grad_clip_max_norm=grad_clip_max_norm,
			log_every_steps=log_every_steps,
			checkpoint_dir=checkpoints_dir,
			monitor=monitor,
			monitor_mode=monitor_mode,
			val_mode="retrieval",
			val_retrieval_topk=val_topk,
			val_retrieval_metric=val_metric,
			val_retrieval_l2_normalize=val_l2_normalize,
		)

	save_json(run_dir / "training_history.json", history)
	print(f"Training complete. Artifacts saved to {run_dir}")


if __name__ == "__main__":
	main()
