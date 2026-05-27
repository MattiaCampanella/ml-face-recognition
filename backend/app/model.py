"""Singleton model service: download from HuggingFace + ONNX Runtime inference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .config import get_model_path, settings

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class ModelService:
    """Singleton that manages ONNX model download, caching, and inference."""

    _instance: ModelService | None = None
    _session = None

    def __new__(cls) -> ModelService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def ensure_model(self) -> Path:
        """Download model from HuggingFace if not already cached."""
        model_path = get_model_path()

        if model_path.exists():
            logger.info("Model already cached at %s", model_path)
            return model_path

        logger.info(
            "Downloading model from %s/%s ...",
            settings.model_repo_id,
            settings.model_filename,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=settings.model_repo_id,
            filename=settings.model_filename,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place file in a subfolder; move if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != model_path:
            downloaded_path.rename(model_path)
            
        # Try to download the external data file as well (.data)
        try:
            downloaded_data = hf_hub_download(
                repo_id=settings.model_repo_id,
                filename=f"{settings.model_filename}.data",
                local_dir=str(model_path.parent),
                local_dir_use_symlinks=False,
            )
            data_path = Path(downloaded_data)
            expected_data_path = Path(f"{str(model_path)}.data")
            if data_path != expected_data_path:
                data_path.rename(expected_data_path)
        except Exception:
            logger.info("No external data file (.data) found, skipping.")

        logger.info("Model downloaded to %s", model_path)
        return model_path

    def get_session(self):
        """Return the ONNX InferenceSession (lazy-loaded, singleton)."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        model_path = self.ensure_model()
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX session loaded.")
        return self._session

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize, normalize, and convert to CHW float32 array."""
        size = settings.image_size
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)

        arr = np.array(image, dtype=np.float32) / 255.0  # HWC [0,1]
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # Normalize
        arr = arr.transpose(2, 0, 1)  # CHW
        return arr

    def predict(self, images: list[Image.Image]) -> np.ndarray:
        """Run inference on a list of PIL images. Returns (N, 512) embeddings."""
        session = self.get_session()

        all_embeddings = []
        batch_size = settings.max_batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch = np.stack(
                [self.preprocess_image(img) for img in batch_images],
                axis=0,
            )

            outputs = session.run(None, {"images": batch})
            all_embeddings.append(outputs[0])

        return np.concatenate(all_embeddings, axis=0)


# Module-level singleton
model_service = ModelService()
