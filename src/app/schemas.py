"""Pydantic схемы для запросов и ответов API."""
from datetime import date
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Запрос на прогноз просмотров."""
    cpm: float = Field(..., gt=0, description="Стоимость за 1000 показов (CPM)")
    channel: str = Field(..., min_length=1, description="Название канала размещения")
    date: date = Field(..., description="Дата размещения в формате YYYY-MM-DD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cpm": 100.0,
                "channel": "example_channel",
                "date": "2024-01-15"
            }
        }


class PredictResponse(BaseModel):
    """Ответ с прогнозом просмотров."""
    predicted_views: int = Field(..., ge=0, description="Предсказанное количество просмотров")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_views": 15000
            }
        }


class HealthResponse(BaseModel):
    """Ответ health check endpoint."""
    status: str = "ok"
    message: str = "Service is running"
