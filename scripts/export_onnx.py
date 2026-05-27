"""Export the trained ResNet-18 face embedding model to ONNX format.

Usage:
    python scripts/export_onnx.py [--output model.onnx] [--checkpoint path/to/best.pt]

If no checkpoint is provided, it downloads from HuggingFace (C0MPLX/triplet/best.pt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.resnet18 import build_baseline_resnet18

CHECKPOINT_REPO_ID = "C0MPLX/triplet"
CHECKPOINT_FILENAME = "best.pt"
CHECKPOINT_URL = (
    f"https://huggingface.co/{CHECKPOINT_REPO_ID}/resolve/main/{CHECKPOINT_FILENAME}?download=1"
)

EMBEDDING_DIM = 512
IMAGE_SIZE = 224


def normalize_state_dict_keys(state_dict: dict) -> dict:
    prefixes = ("_orig_mod.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def download_checkpoint(dest: Path) -> Path:
    print(f"Downloading checkpoint from {CHECKPOINT_URL} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(CHECKPOINT_URL) as response, dest.open("wb") as f:
        f.write(response.read())
    print(f"Saved to {dest}")
    return dest


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = build_baseline_resnet18(
        pretrained=False,
        embedding_dim=EMBEDDING_DIM,
        normalize_embeddings=True,
        classifier_num_classes=None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=False)
    # Filter out classifier keys (expected missing)
    real_missing = [k for k in incompatible.missing_keys if not k.startswith("classifier.")]
    if real_missing:
        print(f"WARNING: Missing keys: {real_missing}")
    if incompatible.unexpected_keys:
        print(f"WARNING: Unexpected keys: {incompatible.unexpected_keys}")

    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # We export only the embedding extraction (forward_features)
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, images):
            return self.model.forward_features(images)

    wrapper = EmbeddingWrapper(model)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def verify(model: torch.nn.Module, onnx_path: Path) -> None:
    import onnxruntime as ort

    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    # PyTorch reference
    with torch.no_grad():
        pt_output = model.forward_features(dummy_input).numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"images": dummy_input.numpy()})[0]

    max_diff = np.abs(pt_output - ort_output).max()
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("PASS: ONNX output matches PyTorch output.")
    else:
        print("WARNING: Outputs differ more than expected. Check model export.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export face model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="backend/model.onnx", help="Output ONNX path")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify ONNX output")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = REPO_ROOT / "demo" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = REPO_ROOT / "scripts" / "best.pt"
            download_checkpoint(ckpt_path)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    output_path = Path(args.output)

    print("Loading model...")
    model = load_model(ckpt_path)

    print("Exporting to ONNX...")
    export_onnx(model, output_path)

    if args.verify:
        print("Verifying ONNX output...")
        verify(model, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! Model size: {size_mb:.1f} MB")
    print(f"Upload to HuggingFace: huggingface-cli upload {CHECKPOINT_REPO_ID} {output_path} model.onnx")


if __name__ == "__main__":
    main()
