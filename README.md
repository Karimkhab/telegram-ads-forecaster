# Telegram Ads Views Forecaster

ML-модель для прогнозирования количества просмотров (охвата) рекламных объявлений в Telegram на основе CPM, канала размещения и даты.

## Описание проекта

Веб-сервис с REST API для предсказания количества просмотров рекламного объявления в Telegram. Модель использует исторические данные для обучения и предсказывает VIEWS на основе:
- **CPM** (стоимость за 1000 показов)
- **CHANNEL_NAME** (канал размещения)
- **DATE** (дата размещения)

## Структура проекта

```
tg-ads-views-forecaster/
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── Хакатон_Постановка_Задачи_Трек_2.pdf
├── src/
│   ├── app/
│   │   ├── main.py            # FastAPI вход
│   │   ├── schemas.py         # Pydantic модели запрос/ответ
│   │   └── predictor.py       # загрузка модели + predict()
│   └── core/
│       └── config.py          # конфиг/пути/настройки
├── tests/
│   ├── test_health.py
│   └── test_predict_contract.py
└── artifacts/
    └── .gitkeep               # сюда модель/статы (НЕ коммитим реальные артефакты)
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd telegram-ads-forecaster
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Windows: venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл при необходимости
```

## Запуск API

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу: `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

### Прогноз просмотров
```
POST /predict
Body: {
    "cpm": 100.0,
    "channel": "channel_name",
    "date": "2024-01-15"
}
```

## Обучение модели

Инструкции по обучению модели будут добавлены после реализации.

## Тестирование

```bash
pytest tests/
```

## Лицензия

MIT
