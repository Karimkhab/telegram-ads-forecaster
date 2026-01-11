# Telegram Ads Views Forecaster

Forecasting Telegram ad reach (VIEWS) from CPM, channel, and date.

This repository is a notebook-first project. The current model is a
well-documented baseline aimed at being explainable rather than complex.

## Project status

- EDA and modeling live in notebooks.
- The API and CLI in `src/` and `scripts/` are scaffolding and not the
  primary entry point yet.
- The goal of this repo is clarity: how the data behaves, what signal is
  available, and why the model looks the way it does.

## Data

- `AllData.csv`: 7 columns, includes target `VIEWS`.
- `TestDataset.csv`: 3 input features only (CPM, CHANNEL_NAME, DATE),
  `VIEWS` is empty.

Important: the organizers published an updated TestDataset where there
are no overlapping keys (CPM, CHANNEL_NAME, DATE) with AllData. This
removes leakage and makes evaluation harder but fair.

## Notebooks (main work)

- `notebooks/EDA.ipynb`: data profiling, distributions, correlations,
  seasonality, and noise checks.
- `notebooks/model_v1.ipynb`: baseline model with clear feature logic,
  training, and submission export.

## Key findings from EDA

- `VIEWS` and `CPM` are heavy-tailed. Log transforms are usually better.
- Global CPM to VIEWS correlation is weak, but within-channel correlation
  is often positive.
- Channels are very sparse: most have fewer than 10 rows.
- There is label noise: identical (CPM, CHANNEL_NAME, DATE) can map to
  different VIEWS.
- Train dates start in 2024, while test includes 2023. Absolute time
  features can hurt generalization.

## Modeling approach (current)

- Target: `log1p(VIEWS)` to stabilize heavy tails.
- CPM features: `log_cpm` and relative CPM to channel median.
- Channel features: shrinked channel median and frequency.
- Date features: yearless seasonality (dow, month, day-of-year cycles).

The model is intentionally simple and explainable.

## Limitations and constraints

- At inference time only 3 features are available (CPM, channel, date).
- Many channels have too few examples to learn reliable behavior.
- Noise in labels creates a hard error floor.
- Unknown competition metric and public test distribution shift make
  offline validation noisy.
- Large improvements likely require external channel metadata
  (subscribers, ER, topic), which is not currently used.

## Project structure

```
telegram-ads-forecaster/
├── notebooks/
│   ├── EDA.ipynb
│   └── model_v1.ipynb
├── reports/
│   └── report.tex
├── src/                 # API/feature code (scaffold)
├── scripts/             # CLI helpers (scaffold)
├── artifacts/           # models and artifacts (not in git)
├── outputs/             # submission files (not in git)
├── data/                # local datasets (not in git)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
