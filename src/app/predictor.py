"""Модуль для загрузки модели и выполнения прогнозов."""
import pickle
from pathlib import Path
from typing import Any
import pandas as pd
from datetime import date

from src.core.config import settings


class Predictor:
    """Класс для загрузки модели и выполнения прогнозов."""
    
    def __init__(self, model_path: Path | None = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу модели. Если None, используется из настроек.
        """
        self.model_path = model_path or settings.model_path
        self.model: Any = None
        self.feature_stats: dict | None = None
        self._load_model()
        self._load_feature_stats()
    
    def _load_model(self) -> None:
        """Загрузка обученной модели."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please train the model first."
            )
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
    
    def _load_feature_stats(self) -> None:
        """Загрузка статистики признаков (если используется для нормализации)."""
        if settings.feature_stats_path.exists():
            import json
            with open(settings.feature_stats_path, "r") as f:
                self.feature_stats = json.load(f)
    
    def predict(self, cpm: float, channel: str, date_value: date) -> int:
        """
        Выполнение прогноза количества просмотров.
        
        Args:
            cpm: Стоимость за 1000 показов
            channel: Название канала
            date_value: Дата размещения
            
        Returns:
            Предсказанное количество просмотров (целое число)
        """
        # TODO: Реализовать преобразование входных данных в признаки
        # и вызов модели для предсказания
        
        # Временная заглушка
        features = self._prepare_features(cpm, channel, date_value)
        prediction = self.model.predict(features)
        
        # Округляем до целого числа и гарантируем неотрицательность
        return max(0, int(round(prediction[0])))
    
    def _prepare_features(self, cpm: float, channel: str, date_value: date) -> pd.DataFrame:
        """
        Подготовка признаков для модели.
        
        Args:
            cpm: Стоимость за 1000 показов
            channel: Название канала
            date_value: Дата размещения
            
        Returns:
            DataFrame с признаками для модели
        """
        # TODO: Реализовать полную подготовку признаков
        # - Извлечение признаков из даты (день недели, месяц, сезонность)
        # - Кодирование канала (one-hot, target encoding и т.д.)
        # - Нормализация CPM при необходимости
        
        # Временная заглушка
        features = pd.DataFrame({
            "cpm": [cpm],
            "channel": [channel],
            "date": [date_value]
        })
        
        return features


# Глобальный экземпляр предиктора (будет инициализирован при старте приложения)
predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Получить экземпляр предиктора (singleton)."""
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor
