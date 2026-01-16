from datetime import date

from pydantic import BaseModel


class PredictItem(BaseModel):
    cpm: float
    channel_name: str
    date: date


class PredictRequest(BaseModel):
    items: list[PredictItem]


class PredictResponse(BaseModel):
    views: list[float]
