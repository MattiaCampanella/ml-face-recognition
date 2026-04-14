from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.face_dataset import CasiaFaceDataset, build_train_label_mapping
from src.evaluation.clustering import (
    apply_pca,
    apply_tsne,
    extract_embeddings,
    plot_2d_embeddings,
    plot_pca_explained_variance,
)
from src.models.resnet18 import build_baseline_resnet18
from src.utils.config import ensure_dir, load_yaml_config, resolve_repo_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings and perform clustering analysis (PCA, t-SNE)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., experiments/configs/base.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., experiments/runs/.../checkpoints/best.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to analyze (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. If None, infers from checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components (default: 2).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30.0).",
    )
    parser.add_argument(
        "--tsne-iter",
        type=int,
        default=1000,
        help="t-SNE iterations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _infer_output_dir(checkpoint_path: Path) -> Path:
    """Infer output directory from checkpoint path structure."""
    # Assume checkpoint is at: experiments/runs/{run_name}/checkpoints/{checkpoint_name}.pt
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent  # Go up from checkpoints/
    output_dir = run_dir / "artifacts" / "clustering"
    return output_dir


def main() -> None:
    args = _parse_args()

    # Setup
    config = load_yaml_config(args.config).data
    device = _resolve_device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else _infer_output_dir(checkpoint_path)
    artifact_dir = ensure_dir(output_dir)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output directory: {artifact_dir}")
    print(f"[INFO] Split: {args.split}")

    # Load data configuration
    data_cfg = config["data"]
    data_root = resolve_repo_path(data_cfg["root_dir"])
    split_file = resolve_repo_path(data_cfg["split_file"])
    image_size = int(data_cfg.get("image_size", 224))

    # Build label mapping from training split
    print(f"[INFO] Building label mapping from split file: {split_file}")
    label_mapping = build_train_label_mapping(split_file)
    print(f"[INFO] Label mapping: {len(label_mapping)} identities in training split")

    # Create dataset for the specified split
    print(f"[INFO] Building dataset for split '{args.split}'...")
    dataset = CasiaFaceDataset(
        data_root=data_root,
        split_file=split_file,
        split_name=args.split,
        image_size=image_size,
        train=False,
        label_mapping=label_mapping,
        drop_unmapped_labels=False,
    )
    print(f"[INFO] Dataset size: {len(dataset)} samples")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    print("[INFO] Building model...")
    model_cfg = config["model"]
    num_classes = len(label_mapping)
    print(f"[INFO] Number of classes: {num_classes}")
    classifier_cfg = model_cfg.get("classifier_head", {})
    classifier_enabled = bool(classifier_cfg.get("enabled", True))

    model = build_baseline_resnet18(
        pretrained=bool(model_cfg.get("pretrained", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        normalize_embeddings=bool(model_cfg.get("normalize_embeddings", True)),
        classifier_num_classes=(num_classes if classifier_enabled else None),
    )
    model = model.to(device)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Extract embeddings
    print("[INFO] Extracting embeddings...")
    embeddings, labels, _ = extract_embeddings(
        model,
        dataloader,
        device,
        return_labels=True,
        amp_enabled=config["system"].get("amp", False),
    )
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    print(f"[INFO] Extracted embeddings shape: {embeddings_np.shape}")
    print(f"[INFO] Unique identities: {len(np.unique(labels_np))}")

    # Apply PCA
    print(f"[INFO] Applying PCA with {args.pca_components} components...")
    pca_2d, pca_obj = apply_pca(embeddings_np, n_components=args.pca_components)

    # Plot PCA explained variance
    print("[INFO] Plotting PCA explained variance...")
    pca_var_plot = artifact_dir / f"pca_explained_variance_{args.split}.png"
    plot_pca_explained_variance(pca_obj, save_path=pca_var_plot)

    # Plot 2D PCA embeddings
    print("[INFO] Plotting 2D PCA embeddings...")
    pca_plot = artifact_dir / f"pca_2d_{args.split}.png"
    plot_2d_embeddings(
        pca_2d,
        labels_np,
        title=f"PCA 2D Projection ({args.split} split)",
        save_path=pca_plot,
    )

    # Apply t-SNE
    print(f"[INFO] Applying t-SNE (perplexity={args.tsne_perplexity}, iter={args.tsne_iter})...")
    tsne_2d, tsne_obj = apply_tsne(
        embeddings_np,
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_iter,
        random_state=args.seed,
    )

    # Plot 2D t-SNE embeddings
    print("[INFO] Plotting 2D t-SNE embeddings...")
    tsne_plot = artifact_dir / f"tsne_2d_{args.split}.png"
    plot_2d_embeddings(
        tsne_2d,
        labels_np,
        title=f"t-SNE 2D Projection ({args.split} split)",
        save_path=tsne_plot,
    )

    # Save results metadata
    metadata = {
        "split": args.split,
        "num_samples": len(dataset),
        "num_identities": int(len(np.unique(labels_np))),
        "embedding_dim": int(embeddings_np.shape[1]),
        "pca_explained_variance_ratio": pca_obj.explained_variance_ratio_.tolist(),
        "pca_cumsum_variance": np.cumsum(pca_obj.explained_variance_ratio_).tolist(),
        "tsne_perplexity": args.tsne_perplexity,
        "plots": {
            "pca_explained_variance": str(pca_var_plot.relative_to(artifact_dir.parent)),
            "pca_2d": str(pca_plot.relative_to(artifact_dir.parent)),
            "tsne_2d": str(tsne_plot.relative_to(artifact_dir.parent)),
        },
    }

    metadata_file = artifact_dir / f"clustering_metadata_{args.split}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {metadata_file}")

    print("[INFO] Clustering analysis complete!")


if __name__ == "__main__":
    main()
