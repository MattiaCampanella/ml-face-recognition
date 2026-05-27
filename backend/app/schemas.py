"""Pydantic schemas for request/response validation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Algorithm(str, Enum):
    dbscan = "dbscan"
    agglomerative = "agglomerative"


class ClusterParams(BaseModel):
    algorithm: Algorithm = Algorithm.dbscan
    eps: float = Field(default=0.25, ge=0.05, le=1.0)
    min_samples: int = Field(default=2, ge=1, le=10)
    threshold: float = Field(default=0.25, ge=0.05, le=1.0)
    linkage: str = Field(default="average", pattern="^(average|complete|single)$")


class ClusterGroup(BaseModel):
    cluster_id: int
    name: str
    image_indices: list[int]


class ClusterResponse(BaseModel):
    clusters: list[ClusterGroup]
    total_images: int
    noise_count: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
