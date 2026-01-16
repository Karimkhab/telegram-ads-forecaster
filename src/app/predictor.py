from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.core.config import ModelConfig, PathsConfig
from src.train.train_model import load_artifacts, make_submission


@dataclass
class Predictor:
    base_dir: Path
    config: ModelConfig
    model: object
    builder: object

    @classmethod
    def load(cls, base_dir: Path) -> "Predictor":
        paths = PathsConfig.from_base(base_dir)
        meta_path = paths.artifacts_dir / "meta_v1.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "Artifacts not found. Run scripts/train.py to create model_v1 artifacts."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = ModelConfig(**meta.get("config", {}))
        model, builder, _ = load_artifacts(paths, cfg)
        return cls(base_dir=base_dir, config=cfg, model=model, builder=builder)

    def predict(self, rows: List[dict]) -> List[float]:
        df = pd.DataFrame(rows)
        submission = make_submission(df, self.model, self.builder, self.config)
        return submission["VIEWS"].astype(float).tolist()
