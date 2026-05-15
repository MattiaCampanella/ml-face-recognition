from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import streamlit as st
import torch
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.evaluation.clustering import extract_embeddings
from src.evaluation.retrieval import pairwise_similarity
from src.models.resnet18 import build_baseline_resnet18

CONFIG_PATH = DEMO_DIR / "resolved_config.json"
CHECKPOINT_PATH = DEMO_DIR / "best.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageEntry:
    name: str
    data: bytes


class UploadDataset(Dataset):
    def __init__(self, items: Iterable[ImageEntry], transform: transforms.Compose) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.items[index]
        image = Image.open(io.BytesIO(entry.data)).convert("RGB")
        return self.transform(image), 0


def load_demo_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing demo config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@st.cache_resource(show_spinner=False)
def load_model(device_str: str) -> tuple[torch.nn.Module, dict, list[str], list[str]]:
    cfg = load_demo_config()
    model_cfg = cfg.get("model", {})

    model = build_baseline_resnet18(
        pretrained=bool(model_cfg.get("pretrained", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        normalize_embeddings=bool(model_cfg.get("normalize_embeddings", True)),
        classifier_num_classes=None,
    )

    device = torch.device(device_str)
    model = model.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, cfg, incompatible.missing_keys, incompatible.unexpected_keys


def filter_checkpoint_mismatch(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> tuple[bool, list[str], list[str]]:
    allowed_prefixes = ("classifier.",)
    filtered_missing = [key for key in missing_keys if not key.startswith(allowed_prefixes)]
    filtered_unexpected = [key for key in unexpected_keys if not key.startswith(allowed_prefixes)]
    has_mismatch = bool(filtered_missing or filtered_unexpected)
    return has_mismatch, filtered_missing, filtered_unexpected


def normalize_state_dict_keys(state_dict: dict) -> dict:
    prefixes = ("_orig_mod.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


@st.cache_data(show_spinner=False)
def compute_embeddings(
    items: list[ImageEntry],
    *,
    image_size: int,
    batch_size: int,
    device_str: str,
    amp_enabled: bool,
    model_normalizes: bool,
) -> torch.Tensor:
    model, _, _, _ = load_model(device_str)
    device = torch.device(device_str)

    dataset = UploadDataset(items, build_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings, _, _ = extract_embeddings(
        model,
        loader,
        device,
        return_labels=False,
        amp_enabled=amp_enabled and device.type == "cuda",
    )

    if not model_normalizes:
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    return embeddings


def parse_uploads(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> tuple[list[ImageEntry], list[Image.Image], list[str]]:
    items: list[ImageEntry] = []
    previews: list[Image.Image] = []
    errors: list[str] = []

    for file in uploaded_files:
        data = file.getvalue()
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # pragma: no cover - UI error handling
            errors.append(f"{file.name}: {exc}")
            continue
        items.append(ImageEntry(name=file.name, data=data))
        previews.append(image)

    return items, previews, errors


def build_agglomerative(distance_matrix: np.ndarray, *, linkage: str, distance_threshold: float) -> np.ndarray:
    kwargs = {
        "n_clusters": None,
        "distance_threshold": distance_threshold,
        "linkage": linkage,
    }
    if "metric" in AgglomerativeClustering.__init__.__code__.co_varnames:
        kwargs["metric"] = "precomputed"
    else:
        kwargs["affinity"] = "precomputed"
    model = AgglomerativeClustering(**kwargs)
    return model.fit_predict(distance_matrix)


def cluster_embeddings(
    embeddings: torch.Tensor,
    *,
    method: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    agg_threshold: float,
    agg_linkage: str,
) -> np.ndarray:
    if embeddings.numel() == 0:
        return np.array([], dtype=int)
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=int)

    if method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
        return model.fit_predict(embeddings.numpy())

    similarity = pairwise_similarity(embeddings, metric="cosine", l2_normalize=False)
    distance = (1.0 - similarity).clamp(min=0).cpu().numpy()
    np.fill_diagonal(distance, 0.0)
    return build_agglomerative(distance, linkage=agg_linkage, distance_threshold=agg_threshold)


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    return base.replace("/", "_").replace("\\", "_")


def build_cluster_names(labels: np.ndarray) -> dict[int, str]:
    unique_labels = sorted({int(label) for label in labels if int(label) >= 0})
    mapping = {label: f"cluster_{index:03d}" for index, label in enumerate(unique_labels)}
    if -1 in labels:
        mapping[-1] = "noise"
    return mapping


def build_zip(items: list[ImageEntry], labels: np.ndarray, mapping: dict[int, str]) -> bytes:
    buffer = io.BytesIO()
    name_counts: dict[str, int] = {}

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry, label in zip(items, labels):
            cluster_name = mapping[int(label)]
            clean_name = sanitize_filename(entry.name)
            key = f"{cluster_name}/{clean_name}"
            count = name_counts.get(key, 0)
            name_counts[key] = count + 1
            if count > 0:
                stem = Path(clean_name).stem
                suffix = Path(clean_name).suffix
                clean_name = f"{stem}_{count}{suffix}"
            archive.writestr(f"{cluster_name}/{clean_name}", entry.data)

    buffer.seek(0)
    return buffer.read()


def render_cluster_previews(
    items: list[ImageEntry],
    previews: list[Image.Image],
    labels: np.ndarray,
    mapping: dict[int, str],
) -> None:
    preview_width = 140
    clusters: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        cluster_name = mapping[int(label)]
        clusters.setdefault(cluster_name, []).append(index)

    def cluster_sort_key(name: str) -> tuple[int, str]:
        if name == "noise":
            return (1, name)
        return (0, name)

    for cluster_name in sorted(clusters.keys(), key=cluster_sort_key):
        indices = clusters[cluster_name]
        st.subheader(f"{cluster_name} ({len(indices)})")
        columns = st.columns(min(4, len(indices)))
        for idx, image_index in enumerate(indices):
            with columns[idx % len(columns)]:
                st.image(previews[image_index], caption=items[image_index].name, width=preview_width)


st.set_page_config(page_title="Face clustering demo", layout="wide")

st.title("Face clustering demo")
st.write("Upload images with a single cropped face per file. The app clusters them by identity.")

with st.sidebar:
    st.header("Settings")
    device_choice = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    algorithm = st.selectbox("Algorithm", options=["DBSCAN", "Agglomerative"], index=0)
    batch_size = st.slider("Batch size", min_value=1, max_value=128, value=32, step=1)

    st.markdown("---")
    st.subheader("DBSCAN")
    dbscan_eps = st.slider("eps", min_value=0.05, max_value=1.0, value=0.25, step=0.01)
    dbscan_min_samples = st.slider("min_samples", min_value=1, max_value=10, value=2, step=1)

    st.markdown("---")
    st.subheader("Agglomerative")
    agg_threshold = st.slider("distance_threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.01)
    agg_linkage = st.selectbox("linkage", options=["average", "complete", "single"], index=0)

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one image to start.")
    st.stop()

items, previews, errors = parse_uploads(uploaded_files)
if errors:
    st.warning("Some files could not be opened:")
    for message in errors:
        st.write(f"- {message}")

if not items:
    st.error("No valid images found.")
    st.stop()

run = st.button("Run clustering", type="primary")

if run:
    resolved_device = resolve_device(device_choice)
    if device_choice == "cuda" and resolved_device == "cpu":
        st.warning("CUDA not available. Using CPU.")

    _model, cfg, missing_keys, unexpected_keys = load_model(resolved_device)
    model_cfg = cfg.get("model", {})
    system_cfg = cfg.get("system", {})

    image_size = int(cfg.get("data", {}).get("image_size", 224))
    amp_enabled = bool(system_cfg.get("amp", False))
    model_normalizes = bool(model_cfg.get("normalize_embeddings", True))

    has_mismatch, filtered_missing, filtered_unexpected = filter_checkpoint_mismatch(
        missing_keys,
        unexpected_keys,
    )
    if has_mismatch:
        st.warning("Checkpoint mismatch detected. Results may be unreliable.")
        with st.expander("Show checkpoint details"):
            if filtered_missing:
                st.write("Missing keys:")
                st.write(filtered_missing)
            if filtered_unexpected:
                st.write("Unexpected keys:")
                st.write(filtered_unexpected)
    elif missing_keys or unexpected_keys:
        st.info("Classifier head weights were ignored for embedding inference.")

    with st.spinner("Extracting embeddings..."):
        embeddings = compute_embeddings(
            items,
            image_size=image_size,
            batch_size=batch_size,
            device_str=resolved_device,
            amp_enabled=amp_enabled,
            model_normalizes=model_normalizes,
        )

    method = "dbscan" if algorithm == "DBSCAN" else "agglomerative"
    with st.spinner("Clustering embeddings..."):
        labels = cluster_embeddings(
            embeddings,
            method=method,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            agg_threshold=agg_threshold,
            agg_linkage=agg_linkage,
        )

    if labels.size == 0:
        st.error("No embeddings to cluster.")
        st.stop()

    mapping = build_cluster_names(labels)

    st.success("Clustering completed.")
    render_cluster_previews(items, previews, labels, mapping)

    zip_bytes = build_zip(items, labels, mapping)
    st.download_button(
        "Download clusters as zip",
        data=zip_bytes,
        file_name="clusters.zip",
        mime="application/zip",
    )
