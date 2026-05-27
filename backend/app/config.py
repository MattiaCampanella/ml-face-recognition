"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_repo_id: str = "C0MPLX/triplet"
    model_filename: str = "model.onnx"
    model_cache_dir: str = "/opt/render/project/.model_cache"

    image_size: int = 224
    max_images: int = 50
    max_batch_size: int = 8

    allowed_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def get_model_path() -> Path:
    return Path(settings.model_cache_dir) / settings.model_filename
