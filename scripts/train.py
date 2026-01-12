from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ModelConfig, PathsConfig
from src.train.train_model import (
    build_pools,
    deduplicate,
    evaluate_predictions,
    load_all_data,
    make_submission,
    predict_valid,
    save_artifacts,
    split_last_days,
    train_full_model,
    train_model,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    paths = PathsConfig.from_base(base_dir)

    cfg = ModelConfig(
        random_seed=42,
        holdout_days=30,
        loss_function="MAE",
        eval_metric="MAE",
        use_channel_id=False,
        deduplicate=True,
        alpha_channel=10.0,
        alpha_slope=5.0,
        min_slope_rows=20,
        use_pred_clip=True,
        pred_clip_q=0.65,
        min_clip_rows=27,
        blend_alpha=0.4,
        blend_base="ch_med",
        scale_factor=0.85,
    )

    np.random.seed(cfg.random_seed)

    df = load_all_data(paths.all_data_path)
    if cfg.deduplicate:
        df = deduplicate(df)

    train_df, valid_df, cutoff, dmax = split_last_days(df, cfg.holdout_days)

    builder, train_pool, valid_pool, y_valid_raw, valid_fe = build_pools(
        train_df, valid_df, cfg
    )
    model = train_model(train_pool, valid_pool, cfg)

    pred_valid = predict_valid(model, valid_pool, valid_fe, cfg)
    metrics = evaluate_predictions(y_valid_raw, pred_valid)

    full_builder, full_model = train_full_model(df, cfg)
    save_artifacts(paths, full_model, full_builder, cfg, metrics)

    if paths.test_data_path.exists():
        test_df = pd.read_csv(paths.test_data_path)
        submission = make_submission(test_df, full_model, full_builder, cfg)
        paths.outputs_dir.mkdir(exist_ok=True)
        out_path = paths.outputs_dir / "TestDataset_filled_model_v1.csv"
        submission.to_csv(out_path, index=False)

    print("Holdout:", cutoff, "->", dmax)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
