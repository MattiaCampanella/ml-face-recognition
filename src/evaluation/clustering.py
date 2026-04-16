from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor, nn
from torch.utils.data import DataLoader


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    return_labels: bool = True,
    return_identity_ids: bool = False,
    amp_enabled: bool = False,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Extract embeddings from a model for all samples in a dataloader.

    Args:
        model: Neural network model with embedding head.
        dataloader: DataLoader providing (image, label) batches or dicts.
        device: Torch device for inference.
        return_labels: Whether to return label tensor.
        return_identity_ids: Whether to return identity ID tensor (requires 'identity_id' in batch).
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        embeddings: (num_samples, embedding_dim) tensor.
        labels: (num_samples,) tensor if return_labels=True, else None.
        identity_ids: (num_samples,) tensor if return_identity_ids=True, else None.
    """
    model.eval()
    embeddings_list = []
    labels_list = [] if return_labels else None
    identity_ids_list = [] if return_identity_ids else None

    with torch.no_grad():
        for batch in dataloader:
            # Extract images and labels from batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                if "images" in batch:
                    images, labels = batch["images"], batch["labels"]
                elif "image" in batch:
                    images, labels = batch["image"], batch["label"]
                else:
                    raise KeyError("Batch must contain image and label keys.")
            else:
                raise TypeError("Unsupported batch format.")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass with AMP if enabled
            with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
                outputs = model(images)

            # Extract embeddings (last layer before classifier)
            if isinstance(outputs, dict) and "embeddings" in outputs:
                batch_embeddings = outputs["embeddings"]
            elif isinstance(outputs, dict) and "embedding" in outputs:
                batch_embeddings = outputs["embedding"]
            elif isinstance(outputs, dict) and "logits" in outputs:
                # If only logits available, try to get embeddings from model's intermediate output
                # This is a fallback; ideally model should expose embeddings
                raise RuntimeError(
                    "Model output contains only logits. "
                    "Ensure the model exposes embeddings in its output dict."
                )
            else:
                raise TypeError(
                    "Model output must be a dict containing 'embeddings' or 'embedding' key, "
                    f"got {type(outputs)}."
                )

            embeddings_list.append(batch_embeddings.cpu())
            if return_labels:
                labels_list.append(labels.cpu())

            if return_identity_ids and isinstance(batch, dict) and "identity_id" in batch:
                identity_ids = batch["identity_id"].to(device)
                identity_ids_list.append(identity_ids.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0) if return_labels else None
    identity_ids = torch.cat(identity_ids_list, dim=0) if return_identity_ids else None

    return embeddings, labels, identity_ids


def apply_pca(embeddings: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Apply PCA to reduce embeddings dimensionality.

    Args:
        embeddings: (num_samples, embedding_dim) array.
        n_components: Number of dimensions to reduce to (default: 2).

    Returns:
        transformed: (num_samples, n_components) array.
        pca: Fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embeddings)
    return transformed, pca


def apply_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, TSNE]:
    """Apply t-SNE to reduce embeddings dimensionality.

    Args:
        embeddings: (num_samples, embedding_dim) array.
        n_components: Number of dimensions to reduce to (default: 2).
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of iterations for t-SNE.
        random_state: Random seed.

    Returns:
        transformed: (num_samples, n_components) array.
        tsne: Fitted TSNE object.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    transformed = tsne.fit_transform(embeddings)
    return transformed, tsne


def plot_2d_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "2D Embedding Space",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 10),
    alpha: float = 0.6,
    s: int = 20,
) -> None:
    """Plot 2D embeddings with color-coded labels.

    Args:
        embeddings_2d: (num_samples, 2) array with 2D coordinates.
        labels: (num_samples,) array with identity labels/IDs.
        title: Plot title.
        save_path: Path to save the figure (PNG/PDF). If None, displays only.
        figsize: Figure size (width, height).
        alpha: Point transparency.
        s: Point size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    num_unique = len(unique_labels)

    # Generate colormap
    if num_unique <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_unique))
    elif num_unique <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_unique))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_unique))

    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[label_to_color[label]],
            label=f"ID {label}" if num_unique <= 20 else None,
            alpha=alpha,
            s=s,
            edgecolors="none",
        )

    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if num_unique <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pca_explained_variance(
    pca: PCA,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot PCA explained variance ratio.

    Args:
        pca: Fitted PCA object.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Explained variance by component
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("Explained Variance by Component", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker="o", linestyle="-", linewidth=2)
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)
