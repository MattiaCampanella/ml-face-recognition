from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.evaluation.clustering import extract_embeddings
from src.evaluation.retrieval import evaluate_retrieval
from src.models.losses import TripletMiningStats, batch_hard_triplet_loss


@dataclass
class EpochMetrics:
    epoch: int
    split: str
    loss: float
    accuracy: float
    num_samples: int
    elapsed_seconds: float


@dataclass
class RetrievalEpochMetrics:
    epoch: int
    split: str
    map_at_k: dict[int, float]
    queries_evaluated: int
    mean_relevant_per_query: float
    elapsed_seconds: float


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _move_batch_to_device(batch: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    """Accept tuple/list batches or dict batches and move tensors to the selected device."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    elif isinstance(batch, dict):
        if "images" in batch and "labels" in batch:
            images, labels = batch["images"], batch["labels"]
        elif "image" in batch and "label" in batch:
            images, labels = batch["image"], batch["label"]
        else:
            raise KeyError(
                "Dictionary batch must contain either ('images', 'labels') "
                "or ('image', 'label')."
            )
    else:
        raise TypeError(
            "Unsupported batch format. Expected tuple/list(batch_x, batch_y) "
            "or dict with image/label keys."
        )

    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def _forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    outputs = model(images)

    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError(
                "Model output dictionary does not contain 'logits'. "
                "Enable classifier head for supervised classification training."
            )
        return outputs["logits"]

    if torch.is_tensor(outputs):
        return outputs

    raise TypeError("Model forward output must be Tensor or dict containing 'logits'.")


def _compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return float(correct) / float(total) if total > 0 else 0.0


def _forward_embeddings(model: nn.Module, images: Tensor) -> Tensor:
    outputs = model(images)

    if isinstance(outputs, dict):
        if "embeddings" not in outputs:
            raise KeyError(
                "Model output dictionary does not contain 'embeddings'. "
                "Use a backbone that exposes embeddings for metric learning."
            )
        return outputs["embeddings"]

    if torch.is_tensor(outputs):
        return outputs

    raise TypeError("Model forward output must be Tensor or dict containing 'embeddings'.")


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    grad_clip_max_norm: Optional[float] = None,
    log_every_steps: int = 50,
) -> EpochMetrics:
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_t = time.perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        images, labels = _move_batch_to_device(batch, device)
        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = _forward_logits(model, images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip_max_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_correct = (torch.argmax(logits, dim=1) == labels).sum().item()
            total_loss += loss.item() * batch_size
            total_correct += int(batch_correct)
            total_samples += batch_size

        if log_every_steps > 0 and step % log_every_steps == 0:
            avg_loss = total_loss / max(1, total_samples)
            avg_acc = total_correct / max(1, total_samples)
            print(
                f"[train] step={step} samples={total_samples} "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
            )

    elapsed = time.perf_counter() - start_t
    return EpochMetrics(
        epoch=-1,
        split="train",
        loss=total_loss / max(1, total_samples),
        accuracy=total_correct / max(1, total_samples),
        num_samples=total_samples,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    split_name: str = "val",
) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_t = time.perf_counter()

    for batch in dataloader:
        images, labels = _move_batch_to_device(batch, device)
        logits = _forward_logits(model, images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        batch_correct = (torch.argmax(logits, dim=1) == labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += int(batch_correct)
        total_samples += batch_size

    elapsed = time.perf_counter() - start_t
    return EpochMetrics(
        epoch=-1,
        split=split_name,
        loss=total_loss / max(1, total_samples),
        accuracy=total_correct / max(1, total_samples),
        num_samples=total_samples,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def run_retrieval_eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    topk: Iterable[int] = (1, 5, 10),
    metric: str = "cosine",
    l2_normalize: bool = True,
    split_name: str = "val",
    amp_enabled: bool = False,
) -> RetrievalEpochMetrics:
    model.eval()
    start_t = time.perf_counter()

    embeddings, _, identity_ids = extract_embeddings(
        model,
        dataloader,
        device,
        return_labels=False,
        return_identity_ids=True,
        amp_enabled=amp_enabled,
    )

    if identity_ids is None:
        raise ValueError("Retrieval validation requires identity_id values in the batch.")

    metrics, _ = evaluate_retrieval(
        embeddings,
        identity_ids,
        topk=topk,
        metric=metric,
        l2_normalize=l2_normalize,
    )

    elapsed = time.perf_counter() - start_t
    return RetrievalEpochMetrics(
        epoch=-1,
        split=split_name,
        map_at_k=metrics.map_at_k,
        queries_evaluated=metrics.queries_evaluated,
        mean_relevant_per_query=metrics.mean_relevant_per_query,
        elapsed_seconds=elapsed,
    )


class SupervisedTrainer:
    """Simple supervised classification trainer for baseline fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: str | torch.device = "auto",
        amp_enabled: bool = False,
        grad_clip_max_norm: Optional[float] = None,
        log_every_steps: int = 50,
        checkpoint_dir: Optional[str | Path] = None,
        monitor: str = "val_loss",
        monitor_mode: str = "min",
        val_mode: str = "classification",
        val_retrieval_topk: Iterable[int] = (1, 5, 10),
        val_retrieval_metric: str = "cosine",
        val_retrieval_l2_normalize: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler

        self.amp_enabled = amp_enabled
        self.grad_clip_max_norm = grad_clip_max_norm
        self.log_every_steps = log_every_steps

        self.monitor = monitor
        self.monitor_mode = monitor_mode
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be one of {'min', 'max'}.")

        self.val_mode = val_mode
        if self.val_mode not in {"classification", "retrieval"}:
            raise ValueError("val_mode must be one of {'classification', 'retrieval' }.")
        self.val_retrieval_topk = tuple(int(k) for k in val_retrieval_topk)
        self.val_retrieval_metric = val_retrieval_metric
        self.val_retrieval_l2_normalize = val_retrieval_l2_normalize

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, Any]] = []
        self.best_value: Optional[float] = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.monitor_mode == "min":
            return value < self.best_value
        return value > self.best_value

    def _resolve_monitor_value(
        self,
        train_metrics: EpochMetrics,
        val_metrics: Optional[EpochMetrics | RetrievalEpochMetrics],
    ) -> float:
        if self.monitor == "train_loss":
            return train_metrics.loss
        if self.monitor == "train_acc":
            return train_metrics.accuracy

        if self.monitor.startswith("val_map_at_") and isinstance(val_metrics, RetrievalEpochMetrics):
            try:
                monitor_k = int(self.monitor.rsplit("_", 1)[-1])
            except ValueError as exc:
                raise ValueError(f"Unsupported monitor value: {self.monitor}") from exc
            return float(val_metrics.map_at_k.get(monitor_k, 0.0))

        if self.monitor == "val_loss" and isinstance(val_metrics, EpochMetrics):
            return val_metrics.loss
        if self.monitor == "val_acc" and isinstance(val_metrics, EpochMetrics):
            return val_metrics.accuracy

        raise ValueError(
            f"Monitor '{self.monitor}' is incompatible with val_mode='{self.val_mode}'."
        )

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        if self.checkpoint_dir is None:
            return

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_value": self.best_value,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
        }

        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler_state_dict"] = self.scheduler.state_dict()

        last_path = self.checkpoint_dir / "last.pt"
        torch.save(payload, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_path)

    def _save_history(self) -> None:
        if self.checkpoint_dir is None:
            return

        history_path = self.checkpoint_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
            handle.write("\n")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
    ) -> list[dict[str, Any]]:
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}.")

        for epoch in range(1, epochs + 1):
            train_metrics = run_train_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                amp_enabled=self.amp_enabled,
                grad_clip_max_norm=self.grad_clip_max_norm,
                log_every_steps=self.log_every_steps,
            )
            train_metrics.epoch = epoch

            epoch_payload: dict[str, Any] = {
                "train": asdict(train_metrics),
            }

            val_metrics: Optional[EpochMetrics | RetrievalEpochMetrics] = None
            if val_loader is not None:
                if self.val_mode == "retrieval":
                    val_metrics = run_retrieval_eval_epoch(
                        model=self.model,
                        dataloader=val_loader,
                        device=self.device,
                        topk=self.val_retrieval_topk,
                        metric=self.val_retrieval_metric,
                        l2_normalize=self.val_retrieval_l2_normalize,
                        split_name="val",
                        amp_enabled=self.amp_enabled,
                    )
                else:
                    val_metrics = run_eval_epoch(
                        model=self.model,
                        dataloader=val_loader,
                        criterion=self.criterion,
                        device=self.device,
                        split_name="val",
                    )

                val_metrics.epoch = epoch
                epoch_payload["val"] = asdict(val_metrics)
                monitor_value = self._resolve_monitor_value(train_metrics, val_metrics)
            else:
                monitor_value = self._resolve_monitor_value(train_metrics, None)

            if self.scheduler is not None:
                # Support both ReduceLROnPlateau and standard schedulers.
                if hasattr(self.scheduler, "step"):
                    try:
                        self.scheduler.step(monitor_value)
                    except TypeError:
                        self.scheduler.step()

            is_best = self._is_better(float(monitor_value))
            if is_best:
                self.best_value = float(monitor_value)

            self._save_checkpoint(epoch=epoch, is_best=is_best)

            epoch_payload["monitor_value"] = float(monitor_value)
            epoch_payload["is_best"] = is_best
            self.history.append(epoch_payload)
            self._save_history()

            train_loss = epoch_payload["train"]["loss"]
            train_acc = epoch_payload["train"]["accuracy"]
            if "val" in epoch_payload:
                if self.val_mode == "retrieval":
                    val_map = epoch_payload["val"]["map_at_k"]
                    val_summary = " ".join(
                        f"val_map@{k}={float(v):.4f}" for k, v in sorted(val_map.items())
                    )
                    print(
                        f"[epoch {epoch:03d}] "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"{val_summary} best={self.best_value:.4f}"
                    )
                else:
                    val_loss = epoch_payload["val"]["loss"]
                    val_acc = epoch_payload["val"]["accuracy"]
                    print(
                        f"[epoch {epoch:03d}] "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                        f"best={self.best_value:.4f}"
                    )
            else:
                print(
                    f"[epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"best={self.best_value:.4f}"
                )

        return self.history


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    epochs: int,
    optimizer: Optimizer,
    scheduler: Optional[Any] = None,
    criterion: Optional[nn.Module] = None,
    device: str | torch.device = "auto",
    amp_enabled: bool = False,
    grad_clip_max_norm: Optional[float] = None,
    log_every_steps: int = 50,
    checkpoint_dir: Optional[str | Path] = None,
    monitor: str = "val_loss",
    monitor_mode: str = "min",
    val_mode: str = "classification",
    val_retrieval_topk: Iterable[int] = (1, 5, 10),
    val_retrieval_metric: str = "cosine",
    val_retrieval_l2_normalize: bool = True,
) -> list[dict[str, Any]]:
    """Functional wrapper around SupervisedTrainer.fit for easier script integration."""
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        amp_enabled=amp_enabled,
        grad_clip_max_norm=grad_clip_max_norm,
        log_every_steps=log_every_steps,
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        monitor_mode=monitor_mode,
        val_mode=val_mode,
        val_retrieval_topk=val_retrieval_topk,
        val_retrieval_metric=val_retrieval_metric,
        val_retrieval_l2_normalize=val_retrieval_l2_normalize,
    )
    return trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs)


@dataclass
class TripletEpochMetrics:
    epoch: int
    split: str
    loss: float
    hard_positive_distance: float
    hard_negative_distance: float
    valid_anchors: int
    total_anchors: int
    elapsed_seconds: float


def run_triplet_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    margin: float = 0.2,
    normalize_embeddings: bool = False,
    amp_enabled: bool = False,
    grad_clip_max_norm: Optional[float] = None,
    log_every_steps: int = 50,
) -> TripletEpochMetrics:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_valid_anchors = 0
    total_anchors = 0
    total_hard_positive = 0.0
    total_hard_negative = 0.0
    total_samples = 0
    start_t = time.perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        images, labels = _move_batch_to_device(batch, device)
        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            embeddings = _forward_embeddings(model, images)
            loss, stats = batch_hard_triplet_loss(
                embeddings,
                labels,
                margin=margin,
                normalize_embeddings=normalize_embeddings,
            )

        scaler.scale(loss).backward()

        if grad_clip_max_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_loss += loss.item() * batch_size
            total_valid_anchors += stats.valid_anchors
            total_anchors += stats.total_anchors
            total_hard_positive += stats.hard_positive_distance * max(1, stats.valid_anchors)
            total_hard_negative += stats.hard_negative_distance * max(1, stats.valid_anchors)
            total_samples += batch_size

        if log_every_steps > 0 and step % log_every_steps == 0:
            avg_loss = total_loss / max(1, total_samples)
            print(
                f"[triplet] step={step} samples={total_samples} loss={avg_loss:.4f} "
                f"valid_anchors={total_valid_anchors}/{total_anchors}"
            )

    elapsed = time.perf_counter() - start_t
    mean_valid = max(1, total_valid_anchors)
    return TripletEpochMetrics(
        epoch=-1,
        split="train",
        loss=total_loss / max(1, total_samples),
        hard_positive_distance=total_hard_positive / mean_valid,
        hard_negative_distance=total_hard_negative / mean_valid,
        valid_anchors=total_valid_anchors,
        total_anchors=total_anchors,
        elapsed_seconds=elapsed,
    )


def train_triplet_learning(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    optimizer: Optimizer,
    scheduler: Optional[Any] = None,
    device: str | torch.device = "auto",
    margin: float = 0.2,
    normalize_embeddings: bool = False,
    amp_enabled: bool = False,
    grad_clip_max_norm: Optional[float] = None,
    log_every_steps: int = 50,
    checkpoint_dir: Optional[str | Path] = None,
    monitor: str = "train_loss",
    monitor_mode: str = "min",
    val_loader: Optional[DataLoader] = None,
    val_retrieval_topk: Iterable[int] = (1, 5, 10),
    val_retrieval_metric: str = "cosine",
    val_retrieval_l2_normalize: bool = True,
) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}.")

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_value: Optional[float] = None

    def is_better(value: float) -> bool:
        nonlocal best_value
        if best_value is None:
            return True
        if monitor_mode == "min":
            return value < best_value
        return value > best_value

    def resolve_monitor_value(
        train_metrics: TripletEpochMetrics,
        val_metrics: Optional[RetrievalEpochMetrics],
    ) -> float:
        if monitor == "train_loss":
            return train_metrics.loss
        if monitor == "train_hard_negative_distance":
            return train_metrics.hard_negative_distance
        if monitor.startswith("val_map_at_") and val_metrics is not None:
            try:
                monitor_k = int(monitor.rsplit("_", 1)[-1])
            except ValueError as exc:
                raise ValueError(f"Unsupported monitor value: {monitor}") from exc
            return float(val_metrics.map_at_k.get(monitor_k, 0.0))
        raise ValueError(
            f"Monitor '{monitor}' is incompatible with triplet validation mode."
        )

    for epoch in range(1, epochs + 1):
        metrics = run_triplet_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=resolved_device,
            margin=margin,
            normalize_embeddings=normalize_embeddings,
            amp_enabled=amp_enabled,
            grad_clip_max_norm=grad_clip_max_norm,
            log_every_steps=log_every_steps,
        )
        metrics.epoch = epoch
        epoch_payload = {"train": asdict(metrics)}
        val_metrics: Optional[RetrievalEpochMetrics] = None
        if val_loader is not None:
            val_metrics = run_retrieval_eval_epoch(
                model=model,
                dataloader=val_loader,
                device=resolved_device,
                topk=val_retrieval_topk,
                metric=val_retrieval_metric,
                l2_normalize=val_retrieval_l2_normalize,
                split_name="val",
                amp_enabled=amp_enabled,
            )
            val_metrics.epoch = epoch
            epoch_payload["val"] = asdict(val_metrics)

        monitor_value = resolve_monitor_value(metrics, val_metrics)
        if scheduler is not None and hasattr(scheduler, "step"):
            try:
                scheduler.step(monitor_value)
            except TypeError:
                scheduler.step()
        is_best = is_better(float(monitor_value))
        if is_best:
            best_value = float(monitor_value)

        if checkpoint_path is not None:
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_value": best_value,
                "monitor": monitor,
                "monitor_mode": monitor_mode,
            }
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(payload, checkpoint_path / "last.pt")
            if is_best:
                torch.save(payload, checkpoint_path / "best.pt")

        epoch_payload["monitor_value"] = float(monitor_value)
        epoch_payload["is_best"] = is_best
        history.append(epoch_payload)

        if checkpoint_path is not None:
            history_path = checkpoint_path / "history.json"
            with history_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
                handle.write("\n")

        if val_metrics is not None:
            val_summary = " ".join(
                f"val_map@{k}={float(v):.4f}" for k, v in sorted(val_metrics.map_at_k.items())
            )
            print(
                f"[triplet epoch {epoch:03d}] loss={metrics.loss:.4f} "
                f"hard_pos={metrics.hard_positive_distance:.4f} hard_neg={metrics.hard_negative_distance:.4f} "
                f"valid_anchors={metrics.valid_anchors}/{metrics.total_anchors} {val_summary} "
                f"best={best_value:.4f}"
            )
        else:
            print(
                f"[triplet epoch {epoch:03d}] loss={metrics.loss:.4f} "
                f"hard_pos={metrics.hard_positive_distance:.4f} hard_neg={metrics.hard_negative_distance:.4f} "
                f"valid_anchors={metrics.valid_anchors}/{metrics.total_anchors} best={best_value:.4f}"
            )

    return history
