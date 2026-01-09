"""FastAPI приложение для прогнозирования просмотров рекламы."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import date

from src.app.schemas import PredictRequest, PredictResponse, HealthResponse
from src.app.predictor import get_predictor
from src.core.config import settings

app = FastAPI(
    title="Telegram Ads Views Forecaster",
    description="API для прогнозирования количества просмотров рекламных объявлений в Telegram",
    version="1.0.0"
)

# CORS middleware для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения."""
    try:
        # Загружаем модель при старте
        get_predictor()
    except FileNotFoundError:
        # Если модель не найдена, приложение все равно запустится
        # но endpoint /predict будет возвращать ошибку
        pass


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Service is running")


@app.post("/predict", response_model=PredictResponse)
async def predict_views(request: PredictRequest) -> PredictResponse:
    """
    Прогнозирование количества просмотров рекламного объявления.
    
    Args:
        request: Запрос с параметрами CPM, канал и дата
        
    Returns:
        Предсказанное количество просмотров
    """
    try:
        predictor = get_predictor()
        predicted_views = predictor.predict(
            cpm=request.cpm,
            channel=request.channel,
            date_value=request.date
        )
        return PredictResponse(predicted_views=predicted_views)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/")
async def root():
    """Корневой endpoint с информацией об API."""
    return {
        "message": "Telegram Ads Views Forecaster API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
