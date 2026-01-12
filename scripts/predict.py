import json
from pathlib import Path

import pandas as pd

from src.core.config import ModelConfig, PathsConfig
from src.train.train_model import load_artifacts, make_submission


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    paths = PathsConfig.from_base(base_dir)

    meta_path = paths.artifacts_dir / "meta_v1.json"
    if not meta_path.exists():
        raise FileNotFoundError("Artifacts not found. Run scripts/train.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = ModelConfig(**meta.get("config", {}))

    model, builder, _ = load_artifacts(paths, cfg)

    test_df = pd.read_csv(paths.test_data_path)
    submission = make_submission(test_df, model, builder, cfg)

    paths.outputs_dir.mkdir(exist_ok=True)
    out_path = paths.outputs_dir / "TestDataset_filled_model_v1.csv"
    submission.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
