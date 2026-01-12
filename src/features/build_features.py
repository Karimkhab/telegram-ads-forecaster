from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.core.config import ModelConfig


@dataclass
class ChannelStats:
    stats: pd.DataFrame
    global_med_log: float
    global_ctr: float
    global_actions_rate: float
    global_slope: float
    global_cpm_log_med: float
    global_clip: float


@dataclass
class ChannelCpmStats:
    stats: pd.DataFrame
    global_median: float


def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    d = out["DATE"]
    out["dow"] = d.dt.dayofweek.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = d.dt.month.astype(int)
    out["dayofyear"] = d.dt.dayofyear.astype(int)
    # Yearless seasonality (safe for 2023 in test).
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def apply_basic_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["cpm"] = out["CPM"].astype(float)
    out["log_cpm"] = np.log1p(out["cpm"].clip(lower=0))
    out = add_date_features(out)
    return out


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.var(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def fit_channel_stats(train: pd.DataFrame, cfg: ModelConfig) -> ChannelStats:
    y = np.log1p(train["VIEWS"].clip(lower=0))
    x = np.log1p(train["CPM"].clip(lower=0))

    global_med = float(np.median(y))
    views_sum = float(train["VIEWS"].sum())
    global_ctr = float(train["CLICKS"].sum() / views_sum) if views_sum else 0.0
    global_actions_rate = float(train["ACTIONS"].sum() / views_sum) if views_sum else 0.0
    global_slope = _safe_slope(x, y)
    global_cpm_log_med = float(np.median(x))
    global_clip = float(train["VIEWS"].quantile(cfg.pred_clip_q))

    stats = (
        train.assign(y=y, log_cpm=x)
        .groupby("CHANNEL_NAME")
        .agg(
            ch_count=("VIEWS", "size"),
            ch_med=("y", "median"),
            views_sum=("VIEWS", "sum"),
            clicks_sum=("CLICKS", "sum"),
            actions_sum=("ACTIONS", "sum"),
            ch_cpm_log_med=("log_cpm", "median"),
        )
        .reset_index()
    )

    stats["ch_ctr_raw"] = stats["clicks_sum"] / stats["views_sum"].replace(0, np.nan)
    stats["ch_actions_rate_raw"] = stats["actions_sum"] / stats["views_sum"].replace(0, np.nan)

    def _channel_slope(g: pd.DataFrame) -> float:
        if len(g) < cfg.min_slope_rows or g["CPM"].nunique() < 2:
            return float("nan")
        gx = np.log1p(g["CPM"].clip(lower=0))
        gy = np.log1p(g["VIEWS"].clip(lower=0))
        return _safe_slope(gx, gy)

    ch_slope = (
        train.groupby("CHANNEL_NAME")
        .apply(_channel_slope)
        .rename("ch_slope_raw")
        .reset_index()
    )

    stats = stats.merge(ch_slope, on="CHANNEL_NAME", how="left")

    w = stats["ch_count"] / (stats["ch_count"] + cfg.alpha_channel)
    stats["ch_med_smooth"] = w * stats["ch_med"] + (1 - w) * global_med
    stats["ch_ctr_smooth"] = w * stats["ch_ctr_raw"].fillna(global_ctr) + (1 - w) * global_ctr
    stats["ch_actions_rate_smooth"] = (
        w * stats["ch_actions_rate_raw"].fillna(global_actions_rate)
        + (1 - w) * global_actions_rate
    )

    w_slope = stats["ch_count"] / (stats["ch_count"] + cfg.alpha_slope)
    stats["ch_slope_smooth"] = (
        w_slope * stats["ch_slope_raw"].fillna(global_slope)
        + (1 - w_slope) * global_slope
    )

    ch_clip = (
        train.groupby("CHANNEL_NAME")["VIEWS"]
        .quantile(cfg.pred_clip_q)
        .rename("ch_views_clip")
        .reset_index()
    )
    stats = stats.merge(ch_clip, on="CHANNEL_NAME", how="left")
    stats.loc[stats["ch_count"] < cfg.min_clip_rows, "ch_views_clip"] = np.nan

    keep_cols = [
        "CHANNEL_NAME",
        "ch_count",
        "ch_med_smooth",
        "ch_ctr_smooth",
        "ch_actions_rate_smooth",
        "ch_cpm_log_med",
        "ch_slope_smooth",
        "ch_views_clip",
    ]

    return ChannelStats(
        stats=stats[keep_cols],
        global_med_log=global_med,
        global_ctr=global_ctr,
        global_actions_rate=global_actions_rate,
        global_slope=global_slope,
        global_cpm_log_med=global_cpm_log_med,
        global_clip=global_clip,
    )


def apply_channel_stats(data: pd.DataFrame, ch_stats: ChannelStats) -> pd.DataFrame:
    out = data.merge(ch_stats.stats, on="CHANNEL_NAME", how="left")
    out["ch_count"] = out["ch_count"].fillna(0).astype(int)
    out["ch_count_log"] = np.log1p(out["ch_count"])
    out["ch_med_smooth"] = out["ch_med_smooth"].fillna(ch_stats.global_med_log).astype(float)
    out["ch_ctr_smooth"] = out["ch_ctr_smooth"].fillna(ch_stats.global_ctr).astype(float)
    out["ch_actions_rate_smooth"] = out["ch_actions_rate_smooth"].fillna(
        ch_stats.global_actions_rate
    ).astype(float)
    out["ch_slope_smooth"] = out["ch_slope_smooth"].fillna(ch_stats.global_slope).astype(float)
    out["ch_cpm_log_med"] = out["ch_cpm_log_med"].fillna(ch_stats.global_cpm_log_med).astype(float)
    out["ch_views_clip"] = out["ch_views_clip"].fillna(ch_stats.global_clip).astype(float)

    # Expected log views from channel slope and CPM deviation.
    out["ch_log_pred"] = out["ch_med_smooth"] + out["ch_slope_smooth"] * (
        out["log_cpm"] - out["ch_cpm_log_med"]
    )
    return out


def fit_channel_cpm_stats(train: pd.DataFrame) -> ChannelCpmStats:
    stats = (
        train.groupby("CHANNEL_NAME")["CPM"]
        .median()
        .reset_index(name="ch_cpm_median")
    )
    global_median = float(train["CPM"].median())
    return ChannelCpmStats(stats=stats, global_median=global_median)


def apply_channel_cpm_stats(data: pd.DataFrame, ch_stats: ChannelCpmStats) -> pd.DataFrame:
    out = data.merge(ch_stats.stats, on="CHANNEL_NAME", how="left")
    out["ch_cpm_median"] = out["ch_cpm_median"].fillna(ch_stats.global_median)
    denom = out["ch_cpm_median"].replace(0, np.nan)
    out["cpm_to_ch_median"] = (out["cpm"] / denom).fillna(1.0)
    return out


def post_process_predictions(pred: np.ndarray, feats: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    out = np.clip(pred, 0, None)
    if cfg.scale_factor != 1.0:
        out = out * cfg.scale_factor
    if cfg.blend_alpha >= 0:
        if cfg.blend_base == "ch_log_pred":
            base = np.expm1(feats["ch_log_pred"].to_numpy())
        else:
            base = np.expm1(feats["ch_med_smooth"].to_numpy())
        out = cfg.blend_alpha * out + (1 - cfg.blend_alpha) * base
    if cfg.use_pred_clip:
        out = np.minimum(out, feats["ch_views_clip"].to_numpy())
    return out


class FeatureBuilder:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.channel_stats: ChannelStats | None = None
        self.channel_cpm_stats: ChannelCpmStats | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.channel_stats = fit_channel_stats(df, self.cfg)
        self.channel_cpm_stats = fit_channel_cpm_stats(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.channel_stats is None or self.channel_cpm_stats is None:
            raise ValueError("FeatureBuilder is not fitted.")
        out = apply_basic_preprocess(df)
        out = apply_channel_stats(out, self.channel_stats)
        out = apply_channel_cpm_stats(out, self.channel_cpm_stats)
        return out

    def feature_columns(self) -> List[str]:
        cols = [
            "log_cpm",
            "cpm_to_ch_median",
            "dow",
            "is_weekend",
            "month",
            "doy_sin",
            "doy_cos",
            "ch_count_log",
            "ch_med_smooth",
            "ch_ctr_smooth",
            "ch_actions_rate_smooth",
            "ch_slope_smooth",
            "ch_log_pred",
        ]
        if self.cfg.use_channel_id:
            cols.append("CHANNEL_NAME")
        return cols

    def cat_features(self) -> List[str]:
        return ["CHANNEL_NAME"] if self.cfg.use_channel_id else []
