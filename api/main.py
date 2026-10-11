"""FastAPI entry point."""

from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException

    from .pipeline import InferencePipeline
    from .schemas import InferenceRequest, InferenceResponse

    def create_app(pipeline: InferencePipeline | None = None) -> FastAPI:
        app = FastAPI(title="TriSpecWave-Mamba-UQ API", version="0.1.0")
        active_pipeline = pipeline

        @app.get("/health")
        def health():
            return {"status": "ok", "model_loaded": active_pipeline is not None}

        @app.post("/infer", response_model=InferenceResponse)
        def infer(request: InferenceRequest):
            if active_pipeline is None:
                raise HTTPException(status_code=503, detail="No inference pipeline configured")
            output = os.path.splitext(request.input_path)[0] + "_sr.tif"
            result = active_pipeline.run_file(request.input_path, output, samples=request.samples)
            return InferenceResponse(prediction_path=output, metadata={"shape": list(result["mean"].shape)})

        return app

    app = create_app()
except ImportError:  # pragma: no cover - optional API dependency
    app = None

    def create_app(*args, **kwargs):
        raise ImportError("Install fastapi and uvicorn to use the web API")

