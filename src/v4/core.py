from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


VERSION = "4.0.0"
MODEL_NAMES = [
    "LogisticRegression",
    "ExtraTrees",
    "HistGradientBoosting",
    "XGBoost",
    "CatBoost",
    "ClinicalLabExpert",
    "UltrasoundExpert",
    "MissingnessExpert",
]
VARIANTS = ["CatBoost", "GlobalBlend", "ExpertBlend", "CEMATStack"]


@dataclass(frozen=True)
class Config:
    seed: int = 2041
    outer_folds: int = 5
    repeats: int = 5
    inner_folds: int = 3
    meta_folds: int = 3
    calibration_folds: int = 3
    bootstrap_iterations: int = 3000

    # Clinical operating points selected using outer-training OOF predictions only.
    high_sensitivity_target: float = 0.90
    balanced_min_sensitivity: float = 0.82
    balanced_min_specificity: float = 0.80

    # Conservative model complexity for a small clinical cohort.
    cat_iterations: int = 500
    xgb_estimators: int = 500
    tree_estimators: int = 600
    min_category_frequency: int = 3
    meta_c: float = 0.15
    stack_l2: float = 0.015

    expected_n: int = 463
    expected_positive: int = 118
    expected_negative: int = 345

    drive_root: str = "/content/drive/MyDrive/MAT-Appendix/cemat_v4_runs"
    fallback_root: str = "/content/MAT-Appendix/cemat_v4_runs"
    force_restart: bool = False


CFG = Config(
    repeats=int(os.environ.get("CEMAT_REPEATS", "5")),
    bootstrap_iterations=int(os.environ.get("CEMAT_BOOTSTRAPS", "3000")),
    force_restart=os.environ.get("CEMAT_FORCE_RESTART", "0") == "1",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def canonical(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    exact = {canonical(c): str(c) for c in columns}
    for candidate in candidates:
        key = canonical(candidate)
        if key in exact:
            return exact[key]
    for column in columns:
        key = canonical(column)
        if any(canonical(candidate) in key for candidate in candidates):
            return str(column)
    return None


def safe_probability(probability: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(probability: np.ndarray | Iterable[float]) -> np.ndarray:
    probability = safe_probability(probability)
    return np.log(probability / (1.0 - probability))


def sigmoid(values: np.ndarray | Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def mount_output_root(config: Config) -> Path:
    drive_root = Path("/content/drive/MyDrive")
    try:
        from google.colab import drive

        if not drive_root.exists():
            drive.mount("/content/drive", force_remount=False)
    except Exception as exc:
        print(f"Drive unavailable; using local runtime storage: {exc}")

    root = Path(config.drive_root if drive_root.exists() else config.fallback_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def config_fingerprint(config: Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def save_environment(run_dir: Path, config: Config) -> None:
    metadata = {
        "version": VERSION,
        "created_utc": now_utc(),
        "config": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "source_commit": os.environ.get("CEMAT_SOURCE_COMMIT", "unrecorded"),
        "source_sha256": os.environ.get("CEMAT_SOURCE_SHA256", "unrecorded"),
    }
    save_json(metadata, run_dir / "config.json")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        (run_dir / "environment_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    except Exception as exc:
        print(f"Could not export pip freeze: {exc}")
