from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# -----------------------------------------------------------------------------
# Drive and run discovery
# -----------------------------------------------------------------------------
from google.colab import drive

MOUNT = Path("/content/drive")
DRIVE_ROOT = MOUNT / "MyDrive"

if not DRIVE_ROOT.exists():
    try:
        drive.mount(str(MOUNT), force_remount=False, timeout_ms=120000)
    except Exception as first_error:
        print("First Drive mount attempt failed:", first_error)
        try:
            drive.flush_and_unmount()
        except Exception:
            pass
        drive.mount(str(MOUNT), force_remount=True, timeout_ms=120000)

if not DRIVE_ROOT.exists():
    raise RuntimeError(
        "Google Drive could not be mounted. Enable browser pop-ups/cookies and rerun."
    )

RUNS_ROOT = DRIVE_ROOT / "MAT-Appendix" / "cgr_mat_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def discover_runs() -> list[Path]:
    runs = [
        path
        for path in RUNS_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("cgr_mat_")
    ]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def newest_full_run() -> Path | None:
    candidates = [path for path in discover_runs() if "_full_" in path.name]
    return candidates[0] if candidates else None


def run_is_complete(run_dir: Path | None) -> bool:
    return bool(
        run_dir
        and (run_dir / "official_holdout_metrics.csv").exists()
        and (run_dir / "official_holdout_predictions.csv").exists()
        and (run_dir / "final_cgr_mat_model.pt").exists()
    )


# -----------------------------------------------------------------------------
# Resume the scientific run only when required and possible
# -----------------------------------------------------------------------------
run_before = newest_full_run()
print("Newest full run before resume:", run_before)
print("Already complete:", run_is_complete(run_before))

if not run_is_complete(run_before):
    if not torch.cuda.is_available():
        print("\n⚠ No completed full run was found and no CUDA GPU is attached.")
        print("Smoke-mode metrics will not be substituted for scientific results.")
        print("Attach a T4/L4/A100 GPU and rerun this notebook to finish training.")
    else:
        print("\nResuming the leakage-safe full CGR-MAT run...")
        print("GPU:", torch.cuda.get_device_name(0))

        os.environ["CGR_MAT_RUN_MODE"] = "full"
        os.environ["CGR_MAT_USE_DRIVE"] = "1"
        os.environ["CGR_MAT_FORCE_RESTART"] = "0"
        os.environ["CGR_MAT_PRETRAINED"] = "1"

        SOURCE_COMMIT = "a072de95190d214177bbf3091cf98ab982e9ce5e"
        LOADER_URL = (
            "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/"
            f"{SOURCE_COMMIT}/src/cgr_mat/cgr_mat_verified_loader_v1_4.py"
        )
        loader_bytes = urllib.request.urlopen(LOADER_URL, timeout=120).read()
        print("Pinned loader SHA256:", hashlib.sha256(loader_bytes).hexdigest())
        exec(
            compile(loader_bytes.decode("utf-8"), LOADER_URL, "exec"),
            globals(),
            globals(),
        )

RUN_DIR = newest_full_run()
STATUS = {
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "run_directory": str(RUN_DIR) if RUN_DIR else None,
    "complete": run_is_complete(RUN_DIR),
}
status_path = DRIVE_ROOT / "MAT-Appendix" / "cgr_mat_finish_status.json"
status_path.write_text(json.dumps(STATUS, indent=2), encoding="utf-8")
print("\nRun status:")
print(json.dumps(STATUS, indent=2))


# -----------------------------------------------------------------------------
# Evaluate only a completed full run
# -----------------------------------------------------------------------------
if not run_is_complete(RUN_DIR):
    display(
        Markdown(
            "## Scientific evaluation unavailable\n"
            "No completed **full** CGR-MAT run is accessible yet. "
            "No metric has been fabricated or taken from smoke mode."
        )
    )
else:
    metrics = pd.read_csv(RUN_DIR / "official_holdout_metrics.csv")
    predictions = pd.read_csv(RUN_DIR / "official_holdout_predictions.csv")
    config_path = RUN_DIR / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )

    print("\nOFFICIAL HOLDOUT METRICS")
    display(metrics)

    mode = (
        config.get("config", {}).get("run_mode")
        if isinstance(config.get("config"), dict)
        else config.get("run_mode")
    )
    scientific_mode = mode == "full" and "_full_" in RUN_DIR.name

    base_row = metrics.loc[
        metrics["model"].str.contains("Tabular CatBoost base", case=False, na=False)
    ].iloc[0]
    selected_rows = metrics.loc[
        metrics["model"].str.contains("OOF-selected threshold", case=False, na=False)
    ]
    selected_row = selected_rows.iloc[0] if len(selected_rows) else metrics.iloc[-1]

    higher_better = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "npv",
        "f1",
        "f2",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    lower_better = ["brier", "log_loss"]
    delta_rows: list[dict[str, object]] = []

    for metric_name in higher_better:
        if metric_name in metrics.columns:
            delta_rows.append(
                {
                    "metric": metric_name,
                    "base": float(base_row[metric_name]),
                    "cgr_mat_selected": float(selected_row[metric_name]),
                    "improvement": float(
                        selected_row[metric_name] - base_row[metric_name]
                    ),
                    "direction": "higher is better",
                }
            )
    for metric_name in lower_better:
        if metric_name in metrics.columns:
            delta_rows.append(
                {
                    "metric": metric_name,
                    "base": float(base_row[metric_name]),
                    "cgr_mat_selected": float(selected_row[metric_name]),
                    "improvement": float(
                        base_row[metric_name] - selected_row[metric_name]
                    ),
                    "direction": "lower is better",
                }
            )

    delta_table = pd.DataFrame(delta_rows)
    delta_table.to_csv(RUN_DIR / "evaluation_metric_deltas.csv", index=False)
    print("\nCGR-MAT IMPROVEMENT OVER TABULAR BASE")
    display(delta_table)

    required_columns = {"label", "base_probability", "calibrated_probability"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise RuntimeError(
            f"Missing holdout prediction columns: {sorted(missing_columns)}"
        )

    y = predictions["label"].to_numpy(dtype=int)
    p_base = predictions["base_probability"].to_numpy(dtype=float)
    p_cgr = predictions["calibrated_probability"].to_numpy(dtype=float)

    point_deltas = {
        "roc_auc_delta": roc_auc_score(y, p_cgr) - roc_auc_score(y, p_base),
        "pr_auc_delta": average_precision_score(y, p_cgr)
        - average_precision_score(y, p_base),
        "brier_improvement": brier_score_loss(y, p_base)
        - brier_score_loss(y, p_cgr),
    }

    rng = np.random.default_rng(2029)
    bootstrap_rows: list[dict[str, float]] = []
    for replicate in range(2000):
        for _ in range(100):
            indices = rng.integers(0, len(y), len(y))
            if np.unique(y[indices]).size == 2:
                break
        yy = y[indices]
        bb = p_base[indices]
        cc = p_cgr[indices]
        bootstrap_rows.append(
            {
                "replicate": replicate + 1,
                "roc_auc_delta": roc_auc_score(yy, cc) - roc_auc_score(yy, bb),
                "pr_auc_delta": average_precision_score(yy, cc)
                - average_precision_score(yy, bb),
                "brier_improvement": brier_score_loss(yy, bb)
                - brier_score_loss(yy, cc),
            }
        )

    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(RUN_DIR / "holdout_paired_bootstrap_deltas.csv", index=False)

    summary_rows = []
    for column in ["roc_auc_delta", "pr_auc_delta", "brier_improvement"]:
        values = bootstrap[column].to_numpy()
        summary_rows.append(
            {
                "comparison": column,
                "point_estimate": float(point_deltas[column]),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.50)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "probability_improvement": float(np.mean(values > 0)),
            }
        )

    bootstrap_summary = pd.DataFrame(summary_rows)
    bootstrap_summary.to_csv(
        RUN_DIR / "holdout_paired_bootstrap_summary.csv", index=False
    )
    print("\nPAIRED PATIENT-LEVEL BOOTSTRAP")
    display(bootstrap_summary)

    indexed = bootstrap_summary.set_index("comparison")
    definite_gains = int((indexed["ci_2_5"] > 0).sum())
    definite_losses = int((indexed["ci_97_5"] < 0).sum())
    point_gains = sum(value > 0 for value in point_deltas.values())

    if not scientific_mode:
        verdict = "NOT A SCIENTIFIC FULL RUN"
        recommendation = "Do not report these metrics as final paper evidence."
    elif definite_gains >= 1 and definite_losses == 0:
        verdict = "SUPPORTIVE EVIDENCE FOR CGR-MAT"
        recommendation = (
            "At least one primary probability metric has a paired 95% bootstrap "
            "interval above zero, without a clearly significant loss in the others."
        )
    elif point_gains >= 2 and definite_losses == 0:
        verdict = "PROMISING BUT STATISTICALLY INCONCLUSIVE"
        recommendation = (
            "Most primary point estimates favor CGR-MAT, but uncertainty overlaps zero. "
            "Report effect sizes and intervals; do not claim superiority."
        )
    elif definite_losses >= 1 and point_gains <= 1:
        verdict = "NO RELIABLE PERFORMANCE SUPPORT FOR CGR-MAT"
        recommendation = (
            "The paired holdout evidence does not support CGR-MAT as the stronger "
            "performance model. Keep it only as an ablation or feasibility result."
        )
    else:
        verdict = "MIXED OR INCONCLUSIVE EVIDENCE"
        recommendation = (
            "The primary metrics disagree or uncertainty is wide. Avoid a superiority claim."
        )

    expected_artifacts = [
        "config.json",
        "dataset_audit.json",
        "cohort_manifest.csv",
        "image_manifest.csv",
        "split_summary.csv",
        "oof_predictions.csv",
        "fold_metrics.csv",
        "cv_summary_mean_std.csv",
        "development_oof_metrics.json",
        "temperature_scaler.pkl",
        "operating_threshold.json",
        "final_preprocessor.pkl",
        "final_tabular_base_model.pkl",
        "final_tabular_base_model.cbm",
        "final_cgr_mat_model.pt",
        "official_holdout_predictions.csv",
        "official_holdout_metrics.csv",
        "official_holdout_metrics.json",
        "reproducibility_bundle.pkl",
        "manifest_sha256.csv",
    ]
    artifact_audit = pd.DataFrame(
        [
            {
                "artifact": name,
                "exists": (RUN_DIR / name).exists(),
                "bytes": (RUN_DIR / name).stat().st_size
                if (RUN_DIR / name).exists()
                else 0,
            }
            for name in expected_artifacts
        ]
    )
    artifact_audit.to_csv(RUN_DIR / "evaluation_artifact_audit.csv", index=False)
    print("\nARTIFACT AUDIT")
    display(artifact_audit)

    chart_metrics = [
        metric
        for metric in [
            "roc_auc",
            "pr_auc",
            "brier",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
        ]
        if metric in metrics.columns
    ]
    axis = metrics.set_index("model")[chart_metrics].T.plot(
        kind="bar", figsize=(14, 7)
    )
    axis.set_title("Official holdout model comparison")
    axis.set_ylabel("Metric value")
    axis.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        RUN_DIR / "evaluation_holdout_metric_comparison.png",
        dpi=230,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    evaluation = {
        "run_directory": str(RUN_DIR),
        "run_mode": mode,
        "scientific_full_run": scientific_mode,
        "holdout_n": int(len(y)),
        "verdict": verdict,
        "recommendation": recommendation,
        "point_deltas": point_deltas,
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
        "missing_expected_artifacts": artifact_audit.loc[
            ~artifact_audit["exists"], "artifact"
        ].tolist(),
    }
    (RUN_DIR / "scientific_evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    (RUN_DIR / "scientific_verdict.md").write_text(
        "\n".join(
            [
                "# CGR-MAT Scientific Evaluation",
                "",
                f"**Run:** `{RUN_DIR.name}`  ",
                f"**Mode:** `{mode}`  ",
                f"**Official holdout N:** {len(y)}",
                "",
                "## Verdict",
                "",
                f"**{verdict}**",
                "",
                recommendation,
                "",
                "## Primary paired point differences",
                "",
                f"- ROC-AUC delta: {point_deltas['roc_auc_delta']:+.4f}",
                f"- PR-AUC delta: {point_deltas['pr_auc_delta']:+.4f}",
                f"- Brier improvement: {point_deltas['brier_improvement']:+.4f}",
                "",
                "Positive values favor CGR-MAT.",
            ]
        ),
        encoding="utf-8",
    )

    display(Markdown(f"## {verdict}\n\n{recommendation}"))
    print("\nEvaluation files saved in:", RUN_DIR)
