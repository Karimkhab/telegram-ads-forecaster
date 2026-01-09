"""Конфигурация приложения."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # API Keys
    tgstat_api_key: str | None = None
    
    # Model paths
    model_path: Path = Path("artifacts/model.pkl")
    feature_stats_path: Path = Path("artifacts/feature_stats.json")
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Data paths
    data_path: Path = Path("data/train_data.csv")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
