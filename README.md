# Telegram Ads Views Forecaster

<p align="center">
  <img src="reports/figures/telegram_ads_picture.jpg" alt="Telegram Ads Forecasting" width="900" />
</p>

<p align="center">
  Forecasting Telegram ad reach (VIEWS) from CPM, channel, and date.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" /></a>
  <a href="notebooks/EDA.ipynb"><img src="https://img.shields.io/badge/EDA-notebook-blue.svg" alt="EDA Notebook" /></a>
  <a href="notebooks/model_v1.ipynb"><img src="https://img.shields.io/badge/Model-v1-blue.svg" alt="Model v1 Notebook" /></a>
  <a href="reports/report.pdf"><img src="https://img.shields.io/badge/Report-PDF-orange.svg" alt="Report PDF" /></a>
</p>

---

## Why this project

We want a **clear, explainable baseline** that predicts ad views using only the
three features available at inference time: `CPM`, `CHANNEL_NAME`, and `DATE`.
The goal is not a black box, but a model that can be **explained**, **debugged**,
and **reused** in a real product context.

Key ideas:
- Heavy tails in `VIEWS` and `CPM` require log transforms.
- CPM is weak globally but meaningful **within channels**.
- Channels are sparse, so **shrinkage** is essential.
- We prefer yearless seasonality to avoid temporal leakage.

---

## What is included

- **EDA notebook** with distributions, correlations, seasonality, and noise checks: `notebooks/EDA.ipynb`.
- **Model v1 notebook** with a clean feature pipeline and submission export: `notebooks/model_v1.ipynb`.
- **Python pipeline** in `src/` mirroring the notebook for clean reuse.
- **Report** in LaTeX + PDF with full analysis and limitations: `reports/report.tex`, `reports/report.pdf`.

---


## Model v1 (current)

Notebook: `notebooks/model_v1.ipynb`

**Target:** `log1p(VIEWS)`

**Features:**
- CPM: `log_cpm`, `cpm_to_ch_median`
- Channel stats (smoothed): median views, CTR, actions rate, CPM slope
- Date: day of week, month, day-of-year cycles

**Post-processing:**
- Blend with channel baseline
- Optional clipping by per-channel quantiles
- Optional scaling to align with the leaderboard metric

**Best public score:** ~0.927

---

## Project structure

```
telegram-ads-forecaster/
|-- notebooks/
|   |-- EDA.ipynb
|   `-- model_v1.ipynb
|-- reports/
|   |-- report.tex
|   `-- report.pdf
|-- src/
|   |-- core/
|   |-- features/
|   `-- train/
|-- scripts/
|   |-- train.py
|   `-- predict.py
|-- artifacts/           # models and artifacts (not in git)
|-- outputs/             # submission files (not in git)
|-- data/                # local datasets (not in git)
`-- requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place datasets in `data/`:
- `data/AllData.csv`
- `data/TestDataset.csv`

---

## How to run

### 1) Notebook flow (main exploration)
- `notebooks/EDA.ipynb`
- `notebooks/model_v1.ipynb`

### 2) Python pipeline (clean run)
```bash
python scripts/train.py
```

This will:
- build features,
- train Model v1,
- save artifacts to `artifacts/`,
- export the submission to `outputs/TestDataset_filled_model_v1.csv`.

### 3) Predict from saved artifacts
```bash
python scripts/predict.py
```

---

## Report

- Full report: `reports/report.pdf`
- Source (LaTeX): `reports/report.tex`

If you want the detailed reasoning, limitations, and future work, start there.

---

## Limitations

- Only three features are available at inference time.
- Channels are sparse; many have <10 rows.
- Label noise creates a hard error floor.
- The public metric is unknown and favors conservative predictions.
- Large gains likely require external metadata (subscribers, ER, topic).

---

## Roadmap

- Add optional external channel metadata (TgStat / TG API).
- Try monotonic or isotonic CPM constraints per channel.
- Add per-channel KNN smoothing for CPM neighborhoods.
- Provide a simple FastAPI inference endpoint.

---

## License

MIT. See `LICENSE`.
