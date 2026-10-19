"""FastAPI service exposing /predict with request logging and Prometheus metrics."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, Field

from heartml import __version__, config

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(config.MODEL_PATH)))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(config.METADATA_PATH)))
LOG_FILE = os.getenv("LOG_FILE", "")

logger = logging.getLogger("heartml.api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    logger.propagate = False

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["path"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["prediction"])
CONFIDENCE = Histogram("model_confidence", "Probability of predicted class",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_LOADED = Gauge("model_loaded", "1 when the model is loaded")


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=0, le=3, examples=[3])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[0])
    thalach: float = Field(..., ge=30, le=250, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=0, le=2, examples=[0])
    ca: int = Field(..., ge=0, le=4, examples=[0])
    thal: int = Field(..., ge=0, le=3, examples=[1])


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_name = "unknown"
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        if METADATA_PATH.exists():
            app.state.model_name = json.loads(METADATA_PATH.read_text()).get(
                "selected_model", "unknown")
        logger.info(json.dumps({"event": "model_loaded", "path": str(MODEL_PATH),
                                "model": app.state.model_name}))
    else:
        MODEL_LOADED.set(0)
        logger.error(json.dumps({"event": "model_missing", "path": str(MODEL_PATH)}))
    yield


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
        logger.info(json.dumps({
            "event": "request", "request_id": request_id, "method": request.method,
            "path": path, "status": status, "latency_ms": round(elapsed * 1000, 2),
            "client": request.client.host if request.client else None,
        }))


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": request.app.state.model_name, "version": __version__}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = pd.DataFrame([features.model_dump()])[config.FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)
    confidence = proba if pred == 1 else 1 - proba
    PREDICTIONS.labels(str(pred)).inc()
    CONFIDENCE.observe(confidence)
    logger.info(json.dumps({"event": "prediction", "prediction": pred,
                            "confidence": round(confidence, 4)}))
    return PredictionResponse(
        prediction=pred,
        label="heart disease" if pred == 1 else "no heart disease",
        confidence=round(confidence, 4),
        probability_disease=round(proba, 4),
        model_version=request.app.state.model_name,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
