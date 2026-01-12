from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path
    all_data_path: Path
    test_data_path: Path
    artifacts_dir: Path
    outputs_dir: Path

    @classmethod
    def from_base(cls, base_dir: Path) -> "PathsConfig":
        base_dir = base_dir.resolve()
        return cls(
            base_dir=base_dir,
            all_data_path=base_dir / "data" / "AllData.csv",
            test_data_path=base_dir / "data" / "TestDataset.csv",
            artifacts_dir=base_dir / "artifacts",
            outputs_dir=base_dir / "outputs",
        )


@dataclass
class ModelConfig:
    random_seed: int = 42
    holdout_days: int = 30
    loss_function: str = "MAE"
    eval_metric: Optional[str] = None
    use_channel_id: bool = False
    deduplicate: bool = True

    alpha_channel: float = 10.0
    alpha_slope: float = 50.0
    min_slope_rows: int = 20
    use_pred_clip: bool = True
    pred_clip_q: float = 0.65
    min_clip_rows: int = 20
    blend_alpha: float = 0.4
    blend_base: str = "ch_med"
    scale_factor: float = 1.0

    def resolved_eval_metric(self) -> str:
        return self.eval_metric or self.loss_function
