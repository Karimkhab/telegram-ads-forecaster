# Telegram Ads Views Forecaster

ML-модель для прогнозирования количества просмотров (охвата) рекламных объявлений в Telegram на основе CPM, канала размещения и даты.

## Описание проекта

Веб-сервис с REST API для предсказания количества просмотров рекламного объявления в Telegram. Модель использует исторические данные для обучения и предсказывает VIEWS на основе:
- **CPM** (стоимость за 1000 показов)
- **CHANNEL_NAME** (канал размещения)
- **DATE** (дата размещения)

## Структура проекта

```
telegram-ads-forecaster/
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── Хакатон_Постановка_Задачи_Трек_2.pdf
├── notebooks/
│   ├── EDA.ipynb
│   └── model_v1.ipynb
├── reports/
│   └── report.tex
├── src/
│   ├── app/
│   │   ├── main.py            # FastAPI вход
│   │   ├── schemas.py         # Pydantic модели запрос/ответ
│   │   └── predictor.py       # загрузка модели + predict()
│   ├── core/
│   │   └── config.py          # конфиг/пути/настройки
│   ├── features/
│   │   └── build_features.py  # построение фичей из CPM/DATE/CHANNEL_NAME
│   └── train/
│       └── train_model.py     # обучение модели
├── scripts/
│   ├── train.py               # CLI для обучения модели
│   └── fill_test_dataset.py   # CLI для заполнения тестового датасета
├── tests/
│   ├── test_health.py
│   └── test_predict_contract.py
├── artifacts/                 # обученные модели и артефакты (не в git)
├── data/                      # данные для обучения (не в git)
└── outputs/                   # результаты предсказаний (не в git)
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

### Подготовка данных

Поместите файл `AllData.csv` в папку `data/`. Файл должен содержать колонки:
- `AD_ID` - уникальный идентификатор объявления
- `CPM` - стоимость за 1000 показов
- `VIEWS` - количество просмотров (целевая переменная)
- `CLICKS` - количество кликов
- `ACTIONS` - количество действий
- `CHANNEL_NAME` - название канала размещения
- `DATE` - дата размещения

**Важно:** В CSV файле названия колонок могут содержать ведущие пробелы (например `" CPM"`). Код автоматически обрабатывает это.

### Обучение

Запустите скрипт обучения:

```bash
python scripts/train.py --data data/AllData.csv --artifacts artifacts --holdout-days 30
```

Параметры:
- `--data` - путь к файлу AllData.csv (обязательно)
- `--artifacts` - директория для сохранения артефактов (по умолчанию: `artifacts`)
- `--holdout-days` - количество дней для holdout набора (по умолчанию: 30)
- `--backtesting` - включить backtesting на 3 фолда (опционально)

Скрипт:
1. Загружает данные и выполняет time-based split (последние N дней - holdout)
2. Строит фичи из CPM, DATE, CHANNEL_NAME
3. Обучает две модели:
   - **Модель A**: CatBoost на `log1p(VIEWS)` с loss RMSE
   - **Модель B**: CatBoost на `VIEWS` с loss MAE
4. Выбирает лучшую модель по MAE на holdout
5. Сохраняет:
   - `artifacts/model.cbm` - обученная модель
   - `artifacts/model_meta.json` - метаданные модели
   - `artifacts/preprocess.json` - параметры препроцессинга (клиппинг, биннинг)
   - `artifacts/channel_freq.json` - частоты каналов
   - `artifacts/metrics.json` - метрики качества (MAE, RMSE, RMSLE, SMAPE)

### Заполнение тестового датасета

Для заполнения тестового датасета предсказаниями:

```bash
python scripts/fill_test_dataset.py \
    --input data/TestDataset.csv \
    --output outputs/TestDataset_filled.csv \
    --artifacts artifacts
```

Параметры:
- `--input` - путь к TestDataset.csv (обязательно)
- `--output` - путь для сохранения результата (обязательно)
- `--artifacts` - директория с обученной моделью (по умолчанию: `artifacts`)

Тестовый датасет должен содержать колонки: `CPM`, `CHANNEL_NAME`, `DATE` (и опционально пустую `VIEWS`).

## Особенности реализации

### Фичи

Модель использует только `CPM`, `DATE`, `CHANNEL_NAME` (нельзя использовать `AD_ID`, `CLICKS`, `ACTIONS`):

1. **CPM фичи:**
   - `cpm` - базовое значение
   - `log_cpm` - `log1p(cpm)`
   - `cpm_clip` - клиппинг по 99.5 перцентилю
   - `cpm_bin` - квантильный биннинг (20 бинов)

2. **DATE фичи:**
   - `dow` - день недели (0-6)
   - `is_weekend` - выходной день
   - `day`, `month`, `weekofyear`, `dayofyear`
   - Циклические фичи: `dow_sin`, `dow_cos`, `dayofyear_sin`, `dayofyear_cos`

3. **CHANNEL_NAME фичи:**
   - `channel` - категориальный признак (обрабатывается CatBoost)
   - `channel_freq` - частота канала в обучающей выборке

### Метрики

Модель оценивается по следующим метрикам:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **RMSLE** (Root Mean Squared Log Error)
- **SMAPE** (Symmetric Mean Absolute Percentage Error)

## Тестирование

```bash
pytest tests/
```

## Лицензия

MIT
