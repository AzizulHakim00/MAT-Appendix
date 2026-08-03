from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from core import Config, MODEL_NAMES, safe_probability
from features import FoldPreprocessor, missingness_indices, modality_indices


GLOBAL_MODELS = [
    "LogisticRegression",
    "ExtraTrees",
    "HistGradientBoosting",
    "XGBoost",
    "CatBoost",
]
EXPERT_MODELS = ["ClinicalLabExpert", "UltrasoundExpert", "MissingnessExpert"]


def build_models(config: Config, seed: int, positive_weight: float) -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            C=0.30,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=config.tree_estimators,
            max_features=0.70,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=320,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=config.xgb_estimators,
            max_depth=3,
            learning_rate=0.025,
            min_child_weight=5,
            subsample=0.82,
            colsample_bytree=0.78,
            reg_alpha=0.30,
            reg_lambda=3.0,
            gamma=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            n_jobs=-1,
            random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=config.cat_iterations,
            depth=5,
            learning_rate=0.025,
            l2_leaf_reg=7.0,
            random_strength=0.35,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def _fit_predict(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.fit(train_x, train_y)
    validation_probability = model.predict_proba(validation_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    return safe_probability(validation_probability), safe_probability(test_probability)


def _constant_predictions(train_y: np.ndarray, n_validation: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    prevalence = float(np.mean(train_y))
    return (
        np.full(n_validation, prevalence, dtype=float),
        np.full(n_test, prevalence, dtype=float),
    )


def generate_level1_predictions(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    test_frame: pd.DataFrame,
    config: Config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Generate strictly cross-fitted level-1 predictions.

    Every preprocessing object and expert is fitted inside an inner-training fold.
    Outer-test predictions are averaged across the same inner models.
    """
    splitter = StratifiedKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=seed,
    )
    train_predictions = {
        name: np.full(len(train_frame), np.nan, dtype=float) for name in MODEL_NAMES
    }
    test_predictions = {
        name: np.zeros(len(test_frame), dtype=float) for name in MODEL_NAMES
    }
    status_rows: list[dict[str, Any]] = []

    for inner_fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train_frame, train_y), start=1
    ):
        fit_frame = train_frame.iloc[fit_indices]
        validation_frame = train_frame.iloc[validation_indices]
        fit_y = train_y[fit_indices]

        preprocessor = FoldPreprocessor(config.min_category_frequency).fit(fit_frame)
        fit_x = preprocessor.transform(fit_frame)
        validation_x = preprocessor.transform(validation_frame)
        test_x = preprocessor.transform(test_frame)

        positive_weight = float((len(fit_y) - fit_y.sum()) / max(fit_y.sum(), 1))
        models = build_models(config, seed + inner_fold * 101, positive_weight)

        for name in GLOBAL_MODELS:
            try:
                validation_probability, test_probability = _fit_predict(
                    models[name], fit_x, fit_y, validation_x, test_x
                )
                status = "ok"
            except Exception as exc:
                validation_probability, test_probability = _constant_predictions(
                    fit_y, len(validation_indices), len(test_frame)
                )
                status = f"fallback:{type(exc).__name__}:{exc}"
            train_predictions[name][validation_indices] = validation_probability
            test_predictions[name] += test_probability / config.inner_folds
            status_rows.append(
                {
                    "inner_fold": inner_fold,
                    "model": name,
                    "features": int(fit_x.shape[1]),
                    "status": status,
                }
            )

        expert_specs = [
            ("ClinicalLabExpert", ["clinical", "laboratory"], "CatBoost"),
            ("UltrasoundExpert", ["clinical", "ultrasound"], "CatBoost"),
            ("MissingnessExpert", [], "LogisticRegression"),
        ]
        for expert_name, groups, base_name in expert_specs:
            if expert_name == "MissingnessExpert":
                selected = missingness_indices(preprocessor.feature_names)
            else:
                selected = modality_indices(preprocessor.feature_names, groups)

            if len(selected) < 2:
                validation_probability, test_probability = _constant_predictions(
                    fit_y, len(validation_indices), len(test_frame)
                )
                status = "fallback:insufficient_features"
            else:
                expert_model = build_models(
                    config,
                    seed + inner_fold * 211 + len(selected),
                    positive_weight,
                )[base_name]
                try:
                    validation_probability, test_probability = _fit_predict(
                        expert_model,
                        fit_x[:, selected],
                        fit_y,
                        validation_x[:, selected],
                        test_x[:, selected],
                    )
                    status = "ok"
                except Exception as exc:
                    validation_probability, test_probability = _constant_predictions(
                        fit_y, len(validation_indices), len(test_frame)
                    )
                    status = f"fallback:{type(exc).__name__}:{exc}"

            train_predictions[expert_name][validation_indices] = validation_probability
            test_predictions[expert_name] += test_probability / config.inner_folds
            status_rows.append(
                {
                    "inner_fold": inner_fold,
                    "model": expert_name,
                    "features": int(len(selected)),
                    "status": status,
                }
            )

    train_table = pd.DataFrame(train_predictions)
    test_table = pd.DataFrame(test_predictions)
    if train_table.isna().any().any():
        missing = train_table.columns[train_table.isna().any()].tolist()
        raise RuntimeError(f"Incomplete level-1 OOF predictions: {missing}")
    return train_table, test_table, status_rows
