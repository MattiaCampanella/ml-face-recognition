"""FastAPI application — face clustering API."""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .clustering import get_strategy
from .config import settings
from .model import model_service
from .schemas import ClusterGroup, ClusterParams, ClusterResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure model is downloaded and session is ready."""
    logger.info("Starting up — loading model...")
    model_service.ensure_model()
    model_service.get_session()
    logger.info("Model ready.")
    yield


app = FastAPI(
    title="Face Clustering API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_service._session is not None,
    )


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_images(
    files: list[UploadFile] = File(...),
    params: str = Form(default="{}"),
):
    """Accept uploaded images + clustering parameters, return cluster assignments."""

    # Validate params JSON
    try:
        params_dict = json.loads(params)
        cluster_params = ClusterParams(**params_dict)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid params: {exc}")

    # Validate file count
    if len(files) > settings.max_images:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum is {settings.max_images}.",
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No images provided.")

    # Read and validate images
    images: list[Image.Image] = []
    for upload in files:
        content_type = upload.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{upload.filename}' is not an image.",
            )
        try:
            data = await upload.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            images.append(img)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read image '{upload.filename}'.",
            )

    # Run inference
    embeddings = model_service.predict(images)

    # Run clustering
    strategy = get_strategy(
        algorithm=cluster_params.algorithm.value,
        eps=cluster_params.eps,
        min_samples=cluster_params.min_samples,
        threshold=cluster_params.threshold,
        linkage=cluster_params.linkage,
    )
    labels = strategy.fit(embeddings)

    # Build response
    clusters_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters_map.setdefault(int(label), []).append(idx)

    noise_count = len(clusters_map.get(-1, []))

    cluster_groups = []
    sorted_labels = sorted(clusters_map.keys())
    for label_id in sorted_labels:
        if label_id == -1:
            name = "noise"
        else:
            name = f"cluster_{label_id:03d}"
        cluster_groups.append(
            ClusterGroup(
                cluster_id=label_id,
                name=name,
                image_indices=clusters_map[label_id],
            )
        )

    return ClusterResponse(
        clusters=cluster_groups,
        total_images=len(images),
        noise_count=noise_count,
    )
