from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from core import (
    CFG,
    VERSION,
    Config,
    VARIANTS,
    config_fingerprint,
    mount_output_root,
    native,
    now_utc,
    save_environment,
    save_json,
    seed_everything,
)
from evaluation import select_thresholds, summarize_predictions
from experts import generate_level1_predictions
from features import availability_features, engineer_features, load_cohort
from stacking import build_variant_predictions


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config: Config) -> None:
    if config.outer_folds < 3 or config.inner_folds < 2 or config.meta_folds < 2:
        raise ValueError("Nested validation requires at least 3 outer folds and 2 inner/meta folds.")
    if config.repeats < 1:
        raise ValueError("At least one repeat is required.")
    synthetic_labels = np.r_[np.zeros(30, dtype=int), np.ones(15, dtype=int)]
    synthetic_probability = np.linspace(0.03, 0.97, len(synthetic_labels))
    selected = select_thresholds(synthetic_labels, synthetic_probability, config)
    if not 0.0 < selected["balanced"]["threshold"] < 1.0:
        raise RuntimeError("Threshold-selection preflight failed.")
    print("PRE-FLIGHT PASSED: configuration and threshold selection.")


def run_outer_fold(
    raw_features: pd.DataFrame,
    engineered_features: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    repeat: int,
    fold: int,
    config: Config,
    run_dir: Path,
) -> pd.DataFrame:
    seed = config.seed + repeat * 10000 + fold * 137
    train_frame = engineered_features.iloc[train_indices].reset_index(drop=True)
    test_frame = engineered_features.iloc[test_indices].reset_index(drop=True)
    train_raw = raw_features.iloc[train_indices].reset_index(drop=True)
    test_raw = raw_features.iloc[test_indices].reset_index(drop=True)
    train_y = labels[train_indices]
    test_y = labels[test_indices]

    level1_train, level1_test, level1_status = generate_level1_predictions(
        train_frame,
        train_y,
        test_frame,
        config,
        seed,
    )
    train_availability = availability_features(train_raw)
    test_availability = availability_features(test_raw)
    train_variants, test_variants, stacking_artifacts = build_variant_predictions(
        level1_train,
        train_y,
        level1_test,
        train_availability,
        test_availability,
        config,
        seed + 500,
    )

    rows = pd.DataFrame(
        {
            "Original_Index": test_indices,
            "Repeat": repeat,
            "Outer_Fold": fold,
            "y_true": test_y,
        }
    )
    threshold_artifacts: dict[str, Any] = {}
    for variant in VARIANTS:
        thresholds = select_thresholds(train_y, train_variants[variant], config)
        threshold_artifacts[variant] = thresholds
        probability = test_variants[variant]
        rows[f"{variant}_Probability"] = probability
        for operating_point, details in thresholds.items():
            threshold = float(details["threshold"])
            rows[f"{variant}_{operating_point}_Threshold"] = threshold
            rows[f"{variant}_{operating_point}_Prediction"] = (
                probability >= threshold
            ).astype(int)

    prediction_path = run_dir / f"fold_predictions_repeat{repeat}_fold{fold}.csv"
    rows.to_csv(prediction_path, index=False)
    fold_log = {
        "version": VERSION,
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "train_n": int(len(train_indices)),
        "test_n": int(len(test_indices)),
        "train_positive": int(train_y.sum()),
        "test_positive": int(test_y.sum()),
        "level1_status": level1_status,
        "stacking": stacking_artifacts,
        "thresholds": threshold_artifacts,
        "completed_utc": now_utc(),
    }
    save_json(fold_log, run_dir / f"fold_log_repeat{repeat}_fold{fold}.json")

    fallback_count = sum(
        str(row.get("status", "")).startswith("fallback") for row in level1_status
    )
    print(
        f"Completed repeat {repeat}, fold {fold}: "
        f"train={len(train_indices)}, test={len(test_indices)}, fallbacks={fallback_count}"
    )
    return rows


def build_manifest(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in {"manifest_sha256.csv"}
    ):
        rows.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(run_dir / "manifest_sha256.csv", index=False)
    return manifest


def main(config: Config = CFG) -> None:
    seed_everything(config.seed)
    preflight(config)
    output_root = mount_output_root(config)
    fingerprint = config_fingerprint(config)
    run_dir = output_root / f"cemat_stack_v4_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_environment(run_dir, config)

    print("=" * 88)
    print(f"CEMAT-STACK V4 | VERSION {VERSION}")
    print("Leakage-safe repeated nested CV on the complete prognostic cohort")
    print("=" * 88)
    print("Output directory:", run_dir)
    print("Configuration:", json.dumps(native(asdict(config)), indent=2))

    raw_features, label_series, leakage_audit = load_cohort(config)
    engineered_features = engineer_features(raw_features)
    labels = label_series.to_numpy(dtype=int)
    leakage_audit.to_csv(run_dir / "leakage_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "patients": len(raw_features),
                "complicated": int(labels.sum()),
                "uncomplicated": int((labels == 0).sum()),
                "raw_predictors": int(raw_features.shape[1]),
                "engineered_predictors": int(engineered_features.shape[1]),
            }
        ]
    ).to_csv(run_dir / "cohort_summary.csv", index=False)

    print(
        f"Validated cohort: N={len(raw_features)}, complicated={int(labels.sum())}, "
        f"uncomplicated={int((labels == 0).sum())}"
    )
    print(
        f"Predictors: raw={raw_features.shape[1]}, engineered={engineered_features.shape[1]}"
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=config.outer_folds,
        n_repeats=config.repeats,
        random_state=config.seed,
    )
    prediction_frames: list[pd.DataFrame] = []
    total_folds = config.outer_folds * config.repeats

    for index, (train_indices, test_indices) in enumerate(
        splitter.split(engineered_features, labels), start=0
    ):
        repeat = index // config.outer_folds + 1
        fold = index % config.outer_folds + 1
        prediction_path = run_dir / f"fold_predictions_repeat{repeat}_fold{fold}.csv"
        print("\n" + "-" * 88)
        print(f"REPEAT {repeat}/{config.repeats} | OUTER FOLD {fold}/{config.outer_folds}")
        print("-" * 88)

        if prediction_path.exists() and not config.force_restart:
            cached = pd.read_csv(prediction_path)
            required = {
                "Original_Index",
                "Repeat",
                "Outer_Fold",
                "y_true",
                *[f"{variant}_Probability" for variant in VARIANTS],
            }
            if len(cached) == len(test_indices) and required.issubset(cached.columns):
                print("Loading completed outer-fold checkpoint.")
                prediction_frames.append(cached)
                continue

        frame = run_outer_fold(
            raw_features,
            engineered_features,
            labels,
            train_indices,
            test_indices,
            repeat,
            fold,
            config,
            run_dir,
        )
        prediction_frames.append(frame)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            run_dir / "all_outer_predictions_partial.csv", index=False
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = len(raw_features) * config.repeats
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Repeated outer prediction count mismatch: {len(predictions)} != {expected_rows}"
        )
    coverage = predictions.groupby(["Repeat", "Original_Index"]).size()
    if not bool((coverage == 1).all()):
        raise RuntimeError("Each patient must have exactly one outer prediction per repeat.")

    predictions.to_csv(run_dir / "all_repeated_nested_predictions.csv", index=False)
    summary = summarize_predictions(predictions, config, run_dir)

    fallback_rows = []
    for log_path in sorted(run_dir.glob("fold_log_repeat*_fold*.json")):
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        for status in payload.get("level1_status", []):
            if str(status.get("status", "")).startswith("fallback"):
                fallback_rows.append({"fold_log": log_path.name, **status})
    pd.DataFrame(fallback_rows).to_csv(run_dir / "fallback_audit.csv", index=False)

    publication_audit = {
        "version": VERSION,
        "cohort_signature_passed": True,
        "repeated_nested_cv_complete": True,
        "outer_evaluations": total_folds,
        "patient_predictions": int(len(predictions)),
        "fallback_count": len(fallback_rows),
        "no_level1_fallbacks": len(fallback_rows) == 0,
        "leakage_features_dropped": leakage_audit.loc[
            leakage_audit["action"] == "DROP", "feature"
        ].tolist(),
        "primary_evidence": "repeat-level mean and standard deviation",
        "secondary_evidence": "per-patient consensus and stratified paired bootstrap",
        "decision": summary["decision"],
    }
    save_json(publication_audit, run_dir / "publication_audit.json")
    build_manifest(run_dir)
    archive = Path(shutil.make_archive(str(run_dir), "zip", root_dir=run_dir))

    print("\n" + "=" * 88)
    print("CEMAT-STACK V4 COMPLETE")
    print("=" * 88)
    print("Run directory:", run_dir)
    print("ZIP archive:", archive)
    print("Primary results: summary_mean_std.csv")
    print("Consensus results: consensus_metrics.csv")
    print("Paired uncertainty: paired_bootstrap.csv")
    print("Decision audit: final_decision.json")


if __name__ == "__main__":
    main()
