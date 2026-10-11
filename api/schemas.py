"""Request/response schemas for the optional web API."""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field

    class InferenceRequest(BaseModel):
        input_path: str = Field(..., description="Path to a multiband raster accessible by the service")
        uncertainty_threshold: float | None = None
        samples: int = 1

    class InferenceResponse(BaseModel):
        prediction_path: str
        uncertainty_path: str | None = None
        metadata: dict[str, Any] = {}
except ImportError:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass
    class InferenceRequest:
        input_path: str
        uncertainty_threshold: float | None = None
        samples: int = 1

    @dataclass
    class InferenceResponse:
        prediction_path: str
        uncertainty_path: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

