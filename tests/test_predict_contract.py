"""Тесты контракта API для прогнозирования."""
import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta

from src.app.main import app

client = TestClient(app)


def test_predict_endpoint_exists():
    """Проверка существования endpoint /predict."""
    # Даже если модель не загружена, endpoint должен существовать
    response = client.post(
        "/predict",
        json={
            "cpm": 100.0,
            "channel": "test_channel",
            "date": "2024-01-15"
        }
    )
    # Может быть 503 (модель не загружена) или 200 (если модель есть)
    assert response.status_code in [200, 503]


def test_predict_request_validation():
    """Тест валидации запроса."""
    # Невалидный запрос - отрицательный CPM
    response = client.post(
        "/predict",
        json={
            "cpm": -10.0,
            "channel": "test_channel",
            "date": "2024-01-15"
        }
    )
    assert response.status_code == 422  # Validation error
    
    # Невалидный запрос - пустой канал
    response = client.post(
        "/predict",
        json={
            "cpm": 100.0,
            "channel": "",
            "date": "2024-01-15"
        }
    )
    assert response.status_code == 422
    
    # Невалидный запрос - отсутствует поле
    response = client.post(
        "/predict",
        json={
            "cpm": 100.0,
            "channel": "test_channel"
            # date отсутствует
        }
    )
    assert response.status_code == 422


def test_predict_response_format():
    """Проверка формата ответа (если модель загружена)."""
    response = client.post(
        "/predict",
        json={
            "cpm": 100.0,
            "channel": "test_channel",
            "date": "2024-01-15"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_views" in data
        assert isinstance(data["predicted_views"], int)
        assert data["predicted_views"] >= 0
