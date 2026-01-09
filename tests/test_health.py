"""Тесты для health check endpoint."""
import pytest
from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def test_health_check():
    """Тест health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


def test_root_endpoint():
    """Тест корневого endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
