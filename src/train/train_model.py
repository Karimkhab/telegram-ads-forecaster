from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.core.config import ModelConfig, PathsConfig
from src.features.build_features import (
    ChannelCpmStats,
    ChannelStats,
    FeatureBuilder,
    post_process_predictions,
)


def load_all_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # Collapse exact duplicates and keep counts as weights.
    key_cols = ["CHANNEL_NAME", "DATE", "CPM"]
    df = df.copy()
    df["dup_count"] = 1
    return (
        df.groupby(key_cols, as_index=False)
        .agg(
            VIEWS=("VIEWS", "median"),
            CLICKS=("CLICKS", "median"),
            ACTIONS=("ACTIONS", "median"),
            AD_ID=("AD_ID", "first"),
            dup_count=("dup_count", "sum"),
        )
    )


def split_last_days(data: pd.DataFrame, holdout_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    unique_dates = np.sort(data["DATE"].unique())
    if len(unique_dates) <= holdout_days:
        raise ValueError("Not enough unique dates for the holdout split.")
    cutoff = unique_dates[-holdout_days]
    train = data.loc[data["DATE"] < cutoff].copy()
    valid = data.loc[data["DATE"] >= cutoff].copy()
    return train, valid, cutoff, unique_dates[-1]


def build_pools(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: ModelConfig,
) -> Tuple[FeatureBuilder, Pool, Pool, np.ndarray, pd.DataFrame]:
    builder = FeatureBuilder(cfg).fit(train_df)

    train_fe = builder.transform(train_df)
    valid_fe = builder.transform(valid_df)

    feature_cols = builder.feature_columns()
    cat_features = builder.cat_features()

    X_train = train_fe[feature_cols]
    X_valid = valid_fe[feature_cols]

    y_train_raw = train_df["VIEWS"].astype(float).values
    y_valid_raw = valid_df["VIEWS"].astype(float).values

    y_train = np.log1p(np.clip(y_train_raw, 0, None))
    y_valid = np.log1p(np.clip(y_valid_raw, 0, None))

    train_weights = train_df["dup_count"].values if "dup_count" in train_df.columns else None
    valid_weights = valid_df["dup_count"].values if "dup_count" in valid_df.columns else None

    train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=train_weights)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features, weight=valid_weights)

    return builder, train_pool, valid_pool, y_valid_raw, valid_fe


def build_model(cfg: ModelConfig) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=cfg.loss_function,
        depth=8,
        learning_rate=0.05,
        iterations=4000,
        l2_leaf_reg=5,
        random_strength=1.0,
        bootstrap_type="Bayesian",
        bagging_temperature=0.8,
        random_seed=cfg.random_seed,
        eval_metric=cfg.resolved_eval_metric(),
        verbose=500,
        od_type="Iter",
        od_wait=200,
        task_type="CPU",
        devices="0",
    )


def train_model(train_pool: Pool, valid_pool: Pool, cfg: ModelConfig) -> CatBoostRegressor:
    model = build_model(cfg)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=500)
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    y_true_clip = np.clip(y_true, 0, None)
    y_pred_clip = np.clip(y_pred, 0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clip) - np.log1p(y_true_clip)) ** 2)))

    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

    return {"MAE": mae, "RMSE": rmse, "RMSLE": rmsle, "SMAPE": smape}


def predict_valid(
    model: CatBoostRegressor,
    valid_pool: Pool,
    valid_fe: pd.DataFrame,
    cfg: ModelConfig,
) -> np.ndarray:
    pred_valid_log = model.predict(valid_pool)
    pred_valid_raw = np.clip(np.expm1(pred_valid_log), 0, None)
    return post_process_predictions(pred_valid_raw, valid_fe, cfg)


def train_full_model(df: pd.DataFrame, cfg: ModelConfig) -> Tuple[FeatureBuilder, CatBoostRegressor]:
    builder = FeatureBuilder(cfg).fit(df)
    full_fe = builder.transform(df)

    feature_cols = builder.feature_columns()
    cat_features = builder.cat_features()

    X_full = full_fe[feature_cols]
    y_full_raw = df["VIEWS"].astype(float).values
    y_full_log = np.log1p(np.clip(y_full_raw, 0, None))

    full_weights = df["dup_count"].values if "dup_count" in df.columns else None
    full_pool = Pool(X_full, y_full_log, cat_features=cat_features, weight=full_weights)

    model = build_model(cfg)
    model.fit(full_pool, verbose=200)
    return builder, model


def make_submission(
    test_df: pd.DataFrame,
    model: CatBoostRegressor,
    builder: FeatureBuilder,
    cfg: ModelConfig,
) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df.columns = test_df.columns.str.strip()
    test_df["DATE"] = pd.to_datetime(test_df["DATE"], errors="coerce")

    test_fe = builder.transform(test_df)
    X_test = test_fe[builder.feature_columns()]
    test_pool = Pool(X_test, cat_features=builder.cat_features())

    pred_test_log = model.predict(test_pool)
    pred_test_raw = np.clip(np.expm1(pred_test_log), 0, None)
    pred_test = post_process_predictions(pred_test_raw, test_fe, cfg)

    out = test_df.copy()
    out["VIEWS"] = np.round(pred_test).astype(int)
    return out


def save_artifacts(
    paths: PathsConfig,
    model: CatBoostRegressor,
    builder: FeatureBuilder,
    cfg: ModelConfig,
    metrics: Dict[str, float],
) -> None:
    paths.artifacts_dir.mkdir(exist_ok=True)

    model.save_model(paths.artifacts_dir / "model_v1.cbm")
    builder.channel_stats.stats.to_csv(paths.artifacts_dir / "channel_stats_v1.csv", index=False)
    builder.channel_cpm_stats.stats.to_csv(
        paths.artifacts_dir / "channel_cpm_stats_v1.csv", index=False
    )

    meta = {
        "model": "CatBoostRegressor",
        "target": "log1p(VIEWS)",
        "feature_cols": builder.feature_columns(),
        "cat_features": builder.cat_features(),
        "holdout_days": cfg.holdout_days,
        "loss_function": cfg.loss_function,
        "eval_metric": cfg.resolved_eval_metric(),
        "pred_clip_q": cfg.pred_clip_q,
        "blend_alpha": cfg.blend_alpha,
        "blend_base": cfg.blend_base,
        "scale_factor": cfg.scale_factor,
        "config": asdict(cfg),
        "global_stats": {
            "med_log": builder.channel_stats.global_med_log,
            "ctr": builder.channel_stats.global_ctr,
            "actions_rate": builder.channel_stats.global_actions_rate,
            "slope": builder.channel_stats.global_slope,
            "cpm_log_med": builder.channel_stats.global_cpm_log_med,
            "clip": builder.channel_stats.global_clip,
            "cpm_median": builder.channel_cpm_stats.global_median,
        },
        "notes": "Model v1 baseline",
    }
    (paths.artifacts_dir / "meta_v1.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (paths.artifacts_dir / "metrics_v1.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_artifacts(
    paths: PathsConfig, cfg: ModelConfig
) -> Tuple[CatBoostRegressor, FeatureBuilder, Dict[str, float]]:
    model = CatBoostRegressor()
    model.load_model(paths.artifacts_dir / "model_v1.cbm")

    meta = json.loads((paths.artifacts_dir / "meta_v1.json").read_text(encoding="utf-8"))
    global_stats = meta.get("global_stats", {})

    ch_stats = pd.read_csv(paths.artifacts_dir / "channel_stats_v1.csv")
    ch_cpm_stats = pd.read_csv(paths.artifacts_dir / "channel_cpm_stats_v1.csv")

    builder = FeatureBuilder(cfg)
    builder.channel_stats = ChannelStats(
        stats=ch_stats,
        global_med_log=float(global_stats.get("med_log", 0.0)),
        global_ctr=float(global_stats.get("ctr", 0.0)),
        global_actions_rate=float(global_stats.get("actions_rate", 0.0)),
        global_slope=float(global_stats.get("slope", 0.0)),
        global_cpm_log_med=float(global_stats.get("cpm_log_med", 0.0)),
        global_clip=float(global_stats.get("clip", 0.0)),
    )
    builder.channel_cpm_stats = ChannelCpmStats(
        stats=ch_cpm_stats,
        global_median=float(global_stats.get("cpm_median", 0.0)),
    )

    return model, builder, meta
