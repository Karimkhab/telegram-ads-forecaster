from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

from src.app.predictor import Predictor
from src.app.schemas import PredictRequest, PredictResponse


app = FastAPI(title="Telegram Ads Forecaster", version="1.0.0")

_predictor: Predictor | None = None


def get_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    try:
        _predictor = Predictor.load(get_base_dir())
    except FileNotFoundError as exc:
        _predictor = None
        raise RuntimeError(str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    rows: List[dict] = []
    for item in payload.items:
        rows.append(
            {
                "CPM": float(item.cpm),
                "CHANNEL_NAME": item.channel_name,
                "DATE": item.date.isoformat(),
            }
        )

    preds = _predictor.predict(rows)
    return PredictResponse(views=preds)
