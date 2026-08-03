from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import Config, MODEL_NAMES, VARIANTS, logit, safe_probability
from experts import GLOBAL_MODELS


@dataclass
class ConstrainedBlender:
    l2: float = 0.015

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "ConstrainedBlender":
        probabilities = safe_probability(probabilities)
        labels = np.asarray(labels, dtype=int)
        n_models = probabilities.shape[1]
        initial = np.full(n_models, 1.0 / n_models, dtype=float)

        def objective(weights: np.ndarray) -> float:
            blended = safe_probability(probabilities @ weights)
            return float(log_loss(labels, blended, labels=[0, 1]) + self.l2 * np.sum(weights**2))

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            self.weights_ = initial
            self.status_ = f"fallback_equal_weights:{result.message}"
        else:
            weights = np.clip(result.x, 0.0, 1.0)
            self.weights_ = weights / max(weights.sum(), 1e-12)
            self.status_ = "ok"
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return safe_probability(np.asarray(probabilities, dtype=float) @ self.weights_)


def _crossfit_blend(
    train_probabilities: pd.DataFrame,
    labels: np.ndarray,
    test_probabilities: pd.DataFrame,
    columns: Sequence[str],
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_matrix = train_probabilities[list(columns)].to_numpy(dtype=float)
    test_matrix = test_probabilities[list(columns)].to_numpy(dtype=float)
    splitter = StratifiedKFold(
        n_splits=config.meta_folds,
        shuffle=True,
        random_state=seed,
    )
    oof = np.zeros(len(labels), dtype=float)
    test = np.zeros(len(test_probabilities), dtype=float)
    weight_rows: list[np.ndarray] = []
    statuses: list[str] = []

    for meta_fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train_matrix, labels), start=1
    ):
        blender = ConstrainedBlender(config.stack_l2).fit(
            train_matrix[fit_indices], labels[fit_indices]
        )
        oof[validation_indices] = blender.predict(train_matrix[validation_indices])
        test += blender.predict(test_matrix) / config.meta_folds
        weight_rows.append(blender.weights_)
        statuses.append(f"fold_{meta_fold}:{blender.status_}")

    weights = np.mean(np.vstack(weight_rows), axis=0)
    weights = weights / max(weights.sum(), 1e-12)
    artifact = {
        "columns": list(columns),
        "mean_crossfit_weights": dict(zip(columns, weights.tolist())),
        "statuses": statuses,
    }
    return safe_probability(oof), safe_probability(test), artifact


def _meta_matrix(probabilities: pd.DataFrame, availability: np.ndarray) -> np.ndarray:
    probability_matrix = safe_probability(probabilities[MODEL_NAMES].to_numpy(dtype=float))
    logits = logit(probability_matrix)
    global_indices = [MODEL_NAMES.index(name) for name in GLOBAL_MODELS]
    global_logits = logits[:, global_indices]
    global_mean = global_logits.mean(axis=1)

    disagreement = np.column_stack(
        [
            probability_matrix.std(axis=1),
            probability_matrix.max(axis=1) - probability_matrix.min(axis=1),
            np.mean(np.abs(probability_matrix - probability_matrix.mean(axis=1, keepdims=True)), axis=1),
        ]
    )

    clinical_observed = availability[:, 0]
    laboratory_observed = availability[:, 1]
    ultrasound_observed = availability[:, 2]
    total_observed = availability[:, 3]

    clinical_lab_logit = logits[:, MODEL_NAMES.index("ClinicalLabExpert")]
    ultrasound_logit = logits[:, MODEL_NAMES.index("UltrasoundExpert")]
    missingness_logit = logits[:, MODEL_NAMES.index("MissingnessExpert")]

    gated_residuals = np.column_stack(
        [
            (clinical_lab_logit - global_mean) * clinical_observed,
            (clinical_lab_logit - global_mean) * laboratory_observed,
            (ultrasound_logit - global_mean) * ultrasound_observed,
            (missingness_logit - global_mean) * (1.0 - total_observed),
        ]
    )
    return np.column_stack([logits, availability, disagreement, gated_residuals]).astype(np.float32)


def _crossfit_cemat(
    train_probabilities: pd.DataFrame,
    labels: np.ndarray,
    test_probabilities: pd.DataFrame,
    train_availability: np.ndarray,
    test_availability: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_matrix = _meta_matrix(train_probabilities, train_availability)
    test_matrix = _meta_matrix(test_probabilities, test_availability)
    splitter = StratifiedKFold(
        n_splits=config.meta_folds,
        shuffle=True,
        random_state=seed,
    )
    oof = np.zeros(len(labels), dtype=float)
    test = np.zeros(len(test_probabilities), dtype=float)
    coefficient_rows: list[np.ndarray] = []

    for meta_fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train_matrix, labels), start=1
    ):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "meta",
                    LogisticRegression(
                        C=config.meta_c,
                        solver="liblinear",
                        max_iter=3000,
                        random_state=seed + meta_fold,
                    ),
                ),
            ]
        )
        model.fit(train_matrix[fit_indices], labels[fit_indices])
        oof[validation_indices] = model.predict_proba(train_matrix[validation_indices])[:, 1]
        test += model.predict_proba(test_matrix)[:, 1] / config.meta_folds
        coefficient_rows.append(model.named_steps["meta"].coef_[0])

    artifact = {
        "meta_feature_count": int(train_matrix.shape[1]),
        "mean_absolute_scaled_coefficient": np.mean(np.abs(np.vstack(coefficient_rows)), axis=0).tolist(),
    }
    return safe_probability(oof), safe_probability(test), artifact


def _crossfit_platt(
    train_probability: np.ndarray,
    labels: np.ndarray,
    test_probability: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_logits = logit(train_probability).reshape(-1, 1)
    test_logits = logit(test_probability).reshape(-1, 1)
    splitter = StratifiedKFold(
        n_splits=config.calibration_folds,
        shuffle=True,
        random_state=seed,
    )
    calibrated_oof = np.zeros(len(labels), dtype=float)
    fold_coefficients: list[dict[str, float]] = []

    for calibration_fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train_logits, labels), start=1
    ):
        calibrator = LogisticRegression(C=2.0, solver="lbfgs", max_iter=2000)
        calibrator.fit(train_logits[fit_indices], labels[fit_indices])
        calibrated_oof[validation_indices] = calibrator.predict_proba(
            train_logits[validation_indices]
        )[:, 1]
        fold_coefficients.append(
            {
                "fold": calibration_fold,
                "coefficient": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
            }
        )

    final_calibrator = LogisticRegression(C=2.0, solver="lbfgs", max_iter=2000)
    final_calibrator.fit(train_logits, labels)
    calibrated_test = final_calibrator.predict_proba(test_logits)[:, 1]
    artifact = {
        "crossfit_coefficients": fold_coefficients,
        "final_coefficient": float(final_calibrator.coef_[0, 0]),
        "final_intercept": float(final_calibrator.intercept_[0]),
    }
    return safe_probability(calibrated_oof), safe_probability(calibrated_test), artifact


def build_variant_predictions(
    level1_train: pd.DataFrame,
    labels: np.ndarray,
    level1_test: pd.DataFrame,
    train_availability: np.ndarray,
    test_availability: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    raw_train: dict[str, np.ndarray] = {}
    raw_test: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {}

    raw_train["CatBoost"] = level1_train["CatBoost"].to_numpy(dtype=float)
    raw_test["CatBoost"] = level1_test["CatBoost"].to_numpy(dtype=float)
    artifacts["CatBoost"] = {"source": "level1_cross_fitted_catboost"}

    raw_train["GlobalBlend"], raw_test["GlobalBlend"], artifacts["GlobalBlend"] = _crossfit_blend(
        level1_train,
        labels,
        level1_test,
        GLOBAL_MODELS,
        config,
        seed + 101,
    )
    raw_train["ExpertBlend"], raw_test["ExpertBlend"], artifacts["ExpertBlend"] = _crossfit_blend(
        level1_train,
        labels,
        level1_test,
        MODEL_NAMES,
        config,
        seed + 211,
    )
    raw_train["CEMATStack"], raw_test["CEMATStack"], artifacts["CEMATStack"] = _crossfit_cemat(
        level1_train,
        labels,
        level1_test,
        train_availability,
        test_availability,
        config,
        seed + 307,
    )

    calibrated_train: dict[str, np.ndarray] = {}
    calibrated_test: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        calibrated_train[variant], calibrated_test[variant], calibration = _crossfit_platt(
            raw_train[variant],
            labels,
            raw_test[variant],
            config,
            seed + 401 + VARIANTS.index(variant) * 17,
        )
        artifacts[variant]["calibration"] = calibration

    return calibrated_train, calibrated_test, artifacts
