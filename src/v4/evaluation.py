from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core import Config, VARIANTS, safe_probability, save_json


METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision_PPV",
    "Sensitivity_TPR",
    "Specificity_TNR",
    "NPV",
    "F1",
    "F2",
    "MCC",
    "ROC_AUC",
    "PR_AUC",
    "Brier",
    "Log_Loss",
]


def metrics_from_prediction(
    labels: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probability = safe_probability(probability)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "N": int(len(labels)),
        "Positive_N": int(labels.sum()),
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(labels, prediction)),
        "Balanced_Accuracy": float(balanced_accuracy_score(labels, prediction)),
        "Precision_PPV": float(precision_score(labels, prediction, zero_division=0)),
        "Sensitivity_TPR": float(recall_score(labels, prediction, zero_division=0)),
        "Specificity_TNR": float(specificity),
        "NPV": float(npv),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "F2": float(fbeta_score(labels, prediction, beta=2, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, prediction)) if np.unique(prediction).size > 1 else 0.0,
        "ROC_AUC": float(roc_auc_score(labels, probability)),
        "PR_AUC": float(average_precision_score(labels, probability)),
        "Brier": float(brier_score_loss(labels, probability)),
        "Log_Loss": float(log_loss(labels, probability, labels=[0, 1])),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def metrics_at_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (safe_probability(probability) >= threshold).astype(int)
    return metrics_from_prediction(labels, probability, prediction, threshold)


def _threshold_candidates(probability: np.ndarray) -> np.ndarray:
    probability = safe_probability(probability)
    quantiles = np.quantile(probability, np.linspace(0.01, 0.99, 199))
    return np.unique(np.concatenate([np.linspace(0.02, 0.98, 481), quantiles]))


def select_thresholds(
    labels: np.ndarray,
    probability: np.ndarray,
    config: Config,
) -> dict[str, dict[str, float]]:
    rows = [metrics_at_threshold(labels, probability, float(t)) for t in _threshold_candidates(probability)]
    table = pd.DataFrame(rows)

    balanced_eligible = table.loc[
        (table["Sensitivity_TPR"] >= config.balanced_min_sensitivity)
        & (table["Specificity_TNR"] >= config.balanced_min_specificity)
    ].copy()
    if balanced_eligible.empty:
        balanced_eligible = table.copy()
    balanced_eligible["Selection"] = (
        balanced_eligible["Balanced_Accuracy"]
        + 0.05 * balanced_eligible["F1"]
        + 0.03 * balanced_eligible["MCC"]
    )
    balanced = balanced_eligible.sort_values(
        ["Selection", "Specificity_TNR", "Sensitivity_TPR"],
        ascending=False,
    ).iloc[0]

    high_sensitivity = table.loc[
        table["Sensitivity_TPR"] >= config.high_sensitivity_target
    ].copy()
    if high_sensitivity.empty:
        high_sensitivity = table.sort_values(
            ["Sensitivity_TPR", "Specificity_TNR", "Precision_PPV"],
            ascending=False,
        ).head(1)
    else:
        high_sensitivity = high_sensitivity.sort_values(
            ["Specificity_TNR", "Precision_PPV", "MCC"],
            ascending=False,
        ).head(1)
    high = high_sensitivity.iloc[0]

    return {
        "balanced": {
            "threshold": float(balanced["Threshold"]),
            "training_sensitivity": float(balanced["Sensitivity_TPR"]),
            "training_specificity": float(balanced["Specificity_TNR"]),
            "training_balanced_accuracy": float(balanced["Balanced_Accuracy"]),
        },
        "high_sensitivity": {
            "threshold": float(high["Threshold"]),
            "training_sensitivity": float(high["Sensitivity_TPR"]),
            "training_specificity": float(high["Specificity_TNR"]),
            "training_balanced_accuracy": float(high["Balanced_Accuracy"]),
        },
    }


def _stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    indices = np.concatenate(
        [
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
    )
    rng.shuffle(indices)
    return indices


def paired_bootstrap(
    labels: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    iterations: int,
    seed: int,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    probability_a = safe_probability(probability_a)
    probability_b = safe_probability(probability_b)
    rng = np.random.default_rng(seed)
    metric_names = ["ROC_AUC", "PR_AUC", "Brier_Improvement", "Balanced_Accuracy", "F1", "MCC"]
    values = {metric: [] for metric in metric_names}

    def point(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict[str, float]:
        ma = metrics_at_threshold(y, pa, threshold_a)
        mb = metrics_at_threshold(y, pb, threshold_b)
        return {
            "ROC_AUC": float(ma["ROC_AUC"] - mb["ROC_AUC"]),
            "PR_AUC": float(ma["PR_AUC"] - mb["PR_AUC"]),
            "Brier_Improvement": float(mb["Brier"] - ma["Brier"]),
            "Balanced_Accuracy": float(ma["Balanced_Accuracy"] - mb["Balanced_Accuracy"]),
            "F1": float(ma["F1"] - mb["F1"]),
            "MCC": float(ma["MCC"] - mb["MCC"]),
        }

    point_estimates = point(labels, probability_a, probability_b)
    for _ in range(iterations):
        indices = _stratified_bootstrap_indices(labels, rng)
        replicate = point(labels[indices], probability_a[indices], probability_b[indices])
        for metric in metric_names:
            values[metric].append(replicate[metric])

    rows = []
    for metric in metric_names:
        array = np.asarray(values[metric], dtype=float)
        rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Metric": metric,
                "Delta_A_minus_B": point_estimates[metric],
                "CI95_Lower": float(np.quantile(array, 0.025)),
                "CI95_Upper": float(np.quantile(array, 0.975)),
                "Probability_A_Better": float(np.mean(array > 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame, config: Config, output_dir: Path) -> dict[str, Any]:
    repeat_rows: list[dict[str, Any]] = []
    for repeat in sorted(predictions["Repeat"].unique()):
        repeat_frame = predictions.loc[predictions["Repeat"] == repeat].sort_values("Original_Index")
        labels = repeat_frame["y_true"].to_numpy(dtype=int)
        for variant in VARIANTS:
            probability = repeat_frame[f"{variant}_Probability"].to_numpy(dtype=float)
            for operating_point in ("balanced", "high_sensitivity"):
                threshold = float(repeat_frame[f"{variant}_{operating_point}_Threshold"].mean())
                prediction = repeat_frame[f"{variant}_{operating_point}_Prediction"].to_numpy(dtype=int)
                metrics = metrics_from_prediction(labels, probability, prediction, threshold)
                metrics.update(
                    {
                        "Repeat": int(repeat),
                        "Variant": variant,
                        "Operating_Point": operating_point,
                    }
                )
                repeat_rows.append(metrics)

    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_metrics.to_csv(output_dir / "repeat_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for operating_point in ("balanced", "high_sensitivity"):
            subset = repeat_metrics.loc[
                (repeat_metrics["Variant"] == variant)
                & (repeat_metrics["Operating_Point"] == operating_point)
            ]
            row: dict[str, Any] = {
                "Variant": variant,
                "Operating_Point": operating_point,
            }
            for metric in METRICS + ["Threshold"]:
                row[f"{metric}_Mean"] = float(subset[metric].mean())
                row[f"{metric}_SD"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    aggregations: dict[str, tuple[str, str]] = {"y_true": ("y_true", "first")}
    for variant in VARIANTS:
        aggregations[f"{variant}_Probability"] = (f"{variant}_Probability", "mean")
        for operating_point in ("balanced", "high_sensitivity"):
            aggregations[f"{variant}_{operating_point}_Threshold"] = (
                f"{variant}_{operating_point}_Threshold",
                "mean",
            )
    consensus = (
        predictions.groupby("Original_Index", as_index=False)
        .agg(**aggregations)
        .sort_values("Original_Index")
    )

    consensus_rows: list[dict[str, Any]] = []
    labels = consensus["y_true"].to_numpy(dtype=int)
    for variant in VARIANTS:
        probability = consensus[f"{variant}_Probability"].to_numpy(dtype=float)
        for operating_point in ("balanced", "high_sensitivity"):
            threshold = float(consensus[f"{variant}_{operating_point}_Threshold"].mean())
            prediction = (probability >= threshold).astype(int)
            metrics = metrics_from_prediction(labels, probability, prediction, threshold)
            metrics.update({"Variant": variant, "Operating_Point": operating_point})
            consensus_rows.append(metrics)
            consensus[f"{variant}_{operating_point}_Prediction"] = prediction
    consensus.to_csv(output_dir / "per_patient_consensus.csv", index=False)
    consensus_metrics = pd.DataFrame(consensus_rows)
    consensus_metrics.to_csv(output_dir / "consensus_metrics.csv", index=False)

    bootstrap_tables = []
    for baseline in ("CatBoost", "GlobalBlend", "ExpertBlend"):
        cemat_probability = consensus["CEMATStack_Probability"].to_numpy(dtype=float)
        baseline_probability = consensus[f"{baseline}_Probability"].to_numpy(dtype=float)
        cemat_threshold = float(consensus["CEMATStack_balanced_Threshold"].mean())
        baseline_threshold = float(consensus[f"{baseline}_balanced_Threshold"].mean())
        bootstrap_tables.append(
            paired_bootstrap(
                labels,
                cemat_probability,
                baseline_probability,
                cemat_threshold,
                baseline_threshold,
                config.bootstrap_iterations,
                config.seed + 900 + len(bootstrap_tables) * 31,
                "CEMATStack",
                baseline,
            )
        )
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)

    balanced_summary = summary.loc[summary["Operating_Point"] == "balanced"].copy()
    balanced_summary["Selection_Score"] = (
        0.35 * balanced_summary["PR_AUC_Mean"]
        + 0.30 * balanced_summary["ROC_AUC_Mean"]
        + 0.20 * balanced_summary["Balanced_Accuracy_Mean"]
        + 0.10 * balanced_summary["F1_Mean"]
        - 0.05 * balanced_summary["Brier_Mean"]
    )
    selected = balanced_summary.sort_values("Selection_Score", ascending=False).iloc[0]

    cemat = consensus_metrics.loc[
        (consensus_metrics["Variant"] == "CEMATStack")
        & (consensus_metrics["Operating_Point"] == "balanced")
    ].iloc[0]
    expert = consensus_metrics.loc[
        (consensus_metrics["Variant"] == "ExpertBlend")
        & (consensus_metrics["Operating_Point"] == "balanced")
    ].iloc[0]
    comparison = bootstrap.loc[
        (bootstrap["Model_A"] == "CEMATStack") & (bootstrap["Model_B"] == "ExpertBlend")
    ].set_index("Metric")
    supported_probability_gain = bool(
        comparison.loc["PR_AUC", "CI95_Lower"] > 0
        or comparison.loc["ROC_AUC", "CI95_Lower"] > 0
    )
    no_material_calibration_loss = bool(float(cemat["Brier"] - expert["Brier"]) <= 0.005)

    decision = {
        "exploratory_selected_variant": str(selected["Variant"]),
        "selection_score": float(selected["Selection_Score"]),
        "cemat_retained_as_performance_driver": bool(
            supported_probability_gain and no_material_calibration_loss
        ),
        "cemat_vs_expertblend": {
            "roc_auc_delta": float(cemat["ROC_AUC"] - expert["ROC_AUC"]),
            "pr_auc_delta": float(cemat["PR_AUC"] - expert["PR_AUC"]),
            "balanced_accuracy_delta": float(cemat["Balanced_Accuracy"] - expert["Balanced_Accuracy"]),
            "brier_delta": float(cemat["Brier"] - expert["Brier"]),
        },
        "performance_targets": {
            "roc_auc_at_least_0_93": bool(cemat["ROC_AUC"] >= 0.93),
            "accuracy_at_least_0_90": bool(cemat["Accuracy"] >= 0.90),
            "sensitivity_at_least_0_90": bool(cemat["Sensitivity_TPR"] >= 0.90),
            "specificity_at_least_0_85": bool(cemat["Specificity_TNR"] >= 0.85),
            "f1_at_least_0_85": bool(cemat["F1"] >= 0.85),
            "brier_at_most_0_12": bool(cemat["Brier"] <= 0.12),
        },
        "primary_reporting_rule": (
            "Use repeat-level mean±SD as primary evidence; consensus and paired bootstrap are secondary. "
            "Do not claim superiority unless paired uncertainty supports it."
        ),
    }
    save_json(decision, output_dir / "final_decision.json")

    create_figures(consensus, consensus_metrics, output_dir)
    return {
        "repeat_metrics": repeat_metrics,
        "summary": summary,
        "consensus": consensus,
        "consensus_metrics": consensus_metrics,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def create_figures(consensus: pd.DataFrame, consensus_metrics: pd.DataFrame, output_dir: Path) -> None:
    labels = consensus["y_true"].to_numpy(dtype=int)

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    for variant in VARIANTS:
        probability = consensus[f"{variant}_Probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probability)
        axis.plot(fpr, tpr, label=f"{variant} (AUC={roc_auc_score(labels, probability):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Repeated nested-CV consensus ROC curves")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "consensus_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    prevalence = labels.mean()
    for variant in VARIANTS:
        probability = consensus[f"{variant}_Probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(labels, probability)
        axis.plot(recall, precision, label=f"{variant} (AP={average_precision_score(labels, probability):.3f})")
    axis.axhline(prevalence, linestyle="--", label=f"Prevalence={prevalence:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Repeated nested-CV consensus precision-recall curves")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "consensus_pr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    for variant in VARIANTS:
        probability = consensus[f"{variant}_Probability"].to_numpy(dtype=float)
        observed, predicted = calibration_curve(labels, probability, n_bins=8, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=variant)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed complicated fraction")
    axis.set_title("Repeated nested-CV consensus calibration")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "consensus_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    balanced = consensus_metrics.loc[consensus_metrics["Operating_Point"] == "balanced"].copy()
    chart_metrics = ["ROC_AUC", "PR_AUC", "Accuracy", "Balanced_Accuracy", "Sensitivity_TPR", "Specificity_TNR", "F1", "MCC"]
    chart = balanced.set_index("Variant")[chart_metrics].T
    axis = chart.plot(kind="bar", figsize=(14, 7))
    axis.set_ylabel("Metric value")
    axis.set_title("Balanced operating-point comparison")
    axis.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "balanced_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
