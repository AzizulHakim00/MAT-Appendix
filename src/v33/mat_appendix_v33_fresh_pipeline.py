# MAT-Appendix V3.3 — fresh leakage-safe high-performance pipeline
# Generated for Google Colab. All tuning, feature selection, calibration,
# blending and thresholds are learned inside outer-training data only.

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import hashlib
import pickle
import subprocess
import warnings
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

def _ensure(package: str, import_name: Optional[str] = None) -> None:
    name = import_name or package.replace("-", "_")
    try:
        __import__(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for _pkg, _imp in [
    ("ucimlrepo", "ucimlrepo"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
]:
    _ensure(_pkg, _imp)

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from ucimlrepo import fetch_ucirepo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    seed: int = 2029
    outer_folds: int = 5
    repeats: int = 1
    inner_folds: int = 3
    candidate_trials: int = 8
    selector_bootstraps: int = 10
    selector_top_k: int = 32
    selector_min_frequency: float = 0.35
    max_selected_features: int = 36
    blend_candidates: int = 5000
    min_sensitivity: float = 0.88
    preferred_specificity: float = 0.78
    mat_epochs: int = 60
    mat_patience: int = 12
    mat_batch_size: int = 64
    mat_d_model: int = 24
    mat_heads: int = 2
    mat_dropout: float = 0.40
    mat_weight_decay: float = 0.005
    mat_lr: float = 0.001
    mat_feature_corruption: float = 0.12
    output_root: str = "/content/drive/MyDrive/MAT-Appendix/v33_runs"
    strict_expected_cohort: bool = True
    expected_n: int = 463
    expected_positive: int = 118


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return native(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def mount_drive() -> None:
    try:
        from google.colab import drive
    except Exception:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def norm_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {str(c).lower().replace(" ", "_"): str(c) for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    for c in columns:
        cl = str(c).lower()
        if any(candidate.lower() in cl for candidate in candidates):
            return str(c)
    return None


LEAKAGE_EXACT = {
    "length_of_stay",
    "management",
    "severity",
    "diagnosis",
    "peritonitis",
    "perforation",
    "appendicular_abscess",
    "abscess_location",
}

LEAKAGE_SUBSTRINGS = (
    "histology",
    "histopath",
    "pathology_result",
    "operation",
    "operative",
    "surgery",
    "postoperative",
    "discharge",
    "complication_label",
    "gangren",
    "perforat",
    "abscess",
    "peritonitis",
)


def load_corrected_cohort(cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ds = fetch_ucirepo(id=938)
    X = ds.data.features.copy()
    targets = ds.data.targets.copy()

    diagnosis_col = find_col(targets.columns, ["Diagnosis"])
    severity_col = find_col(targets.columns, ["Severity"])
    if diagnosis_col is None or severity_col is None:
        raise RuntimeError(
            f"Could not locate Diagnosis/Severity targets. Target columns: {list(targets.columns)}"
        )

    diagnosis = norm_text(targets[diagnosis_col])
    severity = norm_text(targets[severity_col])

    diagnosed_appendicitis = diagnosis.str.contains("appendicitis") & ~diagnosis.str.contains(
        "no appendicitis"
    )
    valid_severity = severity.isin(["complicated", "uncomplicated"])
    keep = diagnosed_appendicitis & valid_severity

    X = X.loc[keep].reset_index(drop=True)
    severity = severity.loc[keep].reset_index(drop=True)
    y = (severity == "complicated").astype(int)

    dropped = []
    retained = []
    for col in list(X.columns):
        key = str(col).strip().lower().replace(" ", "_")
        reason = None
        if key in LEAKAGE_EXACT:
            reason = "identifier/target/post-decision/direct-endpoint"
        elif any(token in key for token in LEAKAGE_SUBSTRINGS):
            # Preserve legitimate pre-decision lymph-node finding.
            if "pathological_lymph_nodes" in key or "lymph_node" in key:
                reason = None
            else:
                reason = "post-decision/pathology/direct-complication proxy"
        if reason:
            dropped.append({"Feature": col, "Action": "DROP", "Reason": reason})
        else:
            retained.append(col)

    X = X[retained].copy()

    if cfg.strict_expected_cohort:
        if len(X) != cfg.expected_n or int(y.sum()) != cfg.expected_positive:
            raise RuntimeError(
                "Corrected cohort mismatch. "
                f"Observed N={len(X)}, positives={int(y.sum())}; "
                f"expected N={cfg.expected_n}, positives={cfg.expected_positive}."
            )

    audit = pd.DataFrame(dropped)
    return X, y, audit


def modality_of(name: str) -> str:
    n = name.lower()
    lab = (
        "wbc", "leuk", "crp", "neut", "lymph", "platelet", "hemoglobin",
        "haemoglobin", "hematocrit", "rbc", "eryth", "mcv", "mch", "rdw",
        "bilirubin", "creatin", "sodium", "potassium", "urine", "ketone",
    )
    us = (
        "appendix", "diameter", "ultrasound", "sonograph", "us_", "_us",
        "free_fluid", "fluid", "compress", "hyperemia", "echogenic", "coprostasis",
        "target_sign", "lymph_node", "bowel_wall", "meteorism",
    )
    clinical = (
        "age", "sex", "bmi", "height", "weight", "duration", "pain", "vomit",
        "nausea", "fever", "temperature", "rebound", "guarding", "tender",
        "migration", "anorexia", "diarr", "dysuria", "stool", "score", "pas",
        "alvarado", "psoas", "rovsing", "cough", "percussion",
    )
    if any(k in n for k in lab):
        return "laboratory"
    if any(k in n for k in us):
        return "ultrasound"
    if any(k in n for k in clinical):
        return "clinical"
    return "other"


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    groups = {"clinical": [], "laboratory": [], "ultrasound": [], "other": []}
    for col in out.columns:
        groups[modality_of(str(col))].append(col)

    out["Missing_Total"] = out.isna().sum(axis=1).astype(float)
    for group, cols in groups.items():
        if cols:
            out[f"Missing_{group.title()}"] = out[cols].isna().sum(axis=1).astype(float)

    def col_alias(aliases: Sequence[str]) -> Optional[str]:
        return find_col(out.columns, aliases)

    def ratio(new_name: str, numerator_aliases: Sequence[str], denominator_aliases: Sequence[str]):
        a = col_alias(numerator_aliases)
        b = col_alias(denominator_aliases)
        if a is not None and b is not None:
            av = safe_numeric(out[a])
            bv = safe_numeric(out[b]).replace(0, np.nan)
            out[new_name] = av / bv

    def product(new_name: str, a_aliases: Sequence[str], b_aliases: Sequence[str]):
        a = col_alias(a_aliases)
        b = col_alias(b_aliases)
        if a is not None and b is not None:
            out[new_name] = safe_numeric(out[a]) * safe_numeric(out[b])

    ratio("CRP_WBC_Ratio", ["CRP"], ["WBC", "Leukocytes"])
    ratio("Neutrophil_Lymphocyte_Ratio", ["Neutrophils", "Neutrophil_Percentage"], ["Lymphocytes", "Lymphocyte_Percentage"])
    ratio("Platelet_Lymphocyte_Ratio", ["Platelets"], ["Lymphocytes", "Lymphocyte_Percentage"])
    product("CRP_WBC_Product", ["CRP"], ["WBC", "Leukocytes"])
    product("CRP_AppendixDiameter_Product", ["CRP"], ["Appendix_Diameter", "Appendix Diameter"])
    product("WBC_AppendixDiameter_Product", ["WBC", "Leukocytes"], ["Appendix_Diameter", "Appendix Diameter"])
    product("Fever_Duration_Product", ["Body_Temperature", "Temperature"], ["Symptoms_Duration", "Duration_of_Symptoms"])

    age = col_alias(["Age"])
    diameter = col_alias(["Appendix_Diameter", "Appendix Diameter"])
    if age is not None and diameter is not None:
        out["AppendixDiameter_Age_Ratio"] = safe_numeric(out[diameter]) / safe_numeric(out[age]).replace(0, np.nan)

    return out


class FoldPreprocessor:
    def __init__(self):
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        self.numeric_cols = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) or safe_numeric(X[c]).notna().mean() >= 0.90
        ]
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler(with_centering=True, with_scaling=True)),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)),
        ])

        self.transformer = ColumnTransformer(
            [
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        )
        self.transformer.fit(X)
        self.feature_names_ = [str(x) for x in self.transformer.get_feature_names_out()]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Preprocessor is not fitted.")
        arr = self.transformer.transform(X)
        return np.asarray(arr, dtype=np.float32)


def stable_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    k = min(cfg.selector_top_k, n_features)
    counts = np.zeros(n_features, dtype=int)
    importance_sum = np.zeros(n_features, dtype=float)

    for b in range(cfg.selector_bootstraps):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        model = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + b,
        )
        model.fit(X[idx], y[idx])
        imp = model.feature_importances_
        top = np.argsort(imp)[-k:]
        counts[top] += 1
        importance_sum += imp

    freq = counts / max(cfg.selector_bootstraps, 1)
    mean_imp = importance_sum / max(cfg.selector_bootstraps, 1)

    selected = np.where(freq >= cfg.selector_min_frequency)[0]
    if len(selected) < min(18, n_features):
        selected = np.argsort(mean_imp)[-min(cfg.max_selected_features, n_features):]
    if len(selected) > cfg.max_selected_features:
        order = selected[np.argsort(mean_imp[selected])]
        selected = order[-cfg.max_selected_features:]

    selected = np.array(sorted(selected.tolist()), dtype=int)
    table = pd.DataFrame({
        "Feature": list(feature_names),
        "Selection_Frequency": freq,
        "Mean_Importance": mean_imp,
        "Selected": [i in set(selected.tolist()) for i in range(n_features)],
    }).sort_values(["Selected", "Selection_Frequency", "Mean_Importance"], ascending=False)
    return selected, table


def candidate_params(name: str, class_ratio: float, cfg: Config, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []

    if name == "Logistic Regression":
        for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            candidates.append({"C": C, "class_weight": "balanced"})
    elif name == "Random Forest":
        for _ in range(cfg.candidate_trials):
            candidates.append({
                "n_estimators": int(rng.choice([600, 900, 1300])),
                "max_depth": [None, 5, 7, 10][int(rng.integers(0, 4))],
                "min_samples_leaf": int(rng.choice([2, 3, 4, 6, 8])),
                "max_features": ["sqrt", 0.35, 0.50, 0.70][int(rng.integers(0, 4))],
                "class_weight": ["balanced", "balanced_subsample"][int(rng.integers(0, 2))],
            })
    elif name == "Extra Trees":
        for _ in range(cfg.candidate_trials):
            candidates.append({
                "n_estimators": int(rng.choice([700, 1000, 1400])),
                "max_depth": [None, 6, 9, 12][int(rng.integers(0, 4))],
                "min_samples_leaf": int(rng.choice([2, 3, 4, 6])),
                "max_features": ["sqrt", 0.40, 0.60, 0.80][int(rng.integers(0, 4))],
                "class_weight": "balanced",
            })
    elif name == "XGBoost":
        for _ in range(cfg.candidate_trials):
            candidates.append({
                "n_estimators": int(rng.choice([250, 400, 650])),
                "max_depth": int(rng.choice([2, 3, 4])),
                "learning_rate": float(rng.choice([0.015, 0.025, 0.04, 0.06])),
                "min_child_weight": float(rng.choice([2, 4, 6, 8])),
                "subsample": float(rng.choice([0.70, 0.82, 0.92])),
                "colsample_bytree": float(rng.choice([0.55, 0.70, 0.85])),
                "reg_alpha": float(rng.choice([0.0, 0.05, 0.2, 0.5])),
                "reg_lambda": float(rng.choice([1.0, 3.0, 7.0, 12.0])),
                "gamma": float(rng.choice([0.0, 0.1, 0.3])),
                "scale_pos_weight": float(class_ratio * rng.choice([0.85, 1.0, 1.15])),
            })
    elif name == "CatBoost":
        for _ in range(cfg.candidate_trials):
            candidates.append({
                "iterations": int(rng.choice([350, 550, 800])),
                "depth": int(rng.choice([3, 4, 5, 6])),
                "learning_rate": float(rng.choice([0.015, 0.025, 0.04, 0.06])),
                "l2_leaf_reg": float(rng.choice([3, 6, 10, 16])),
                "random_strength": float(rng.choice([0.2, 0.5, 1.0])),
                "bagging_temperature": float(rng.choice([0.0, 0.5, 1.0])),
                "class_weights": [1.0, float(class_ratio)],
            })
    return candidates


def make_model(name: str, params: Mapping[str, Any], seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(
            solver="liblinear",
            max_iter=3000,
            random_state=seed,
            **params,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            loss_function="Logloss",
            **params,
        )
    raise KeyError(name)


def threshold_stats(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p[:, None] >= thresholds[None, :]
    positive = y[:, None] == 1
    negative = ~positive
    tp = np.sum(pred & positive, axis=0).astype(float)
    fn = np.sum((~pred) & positive, axis=0).astype(float)
    tn = np.sum((~pred) & negative, axis=0).astype(float)
    fp = np.sum(pred & negative, axis=0).astype(float)
    sens = tp / np.maximum(tp + fn, 1.0)
    spec = tn / np.maximum(tn + fp, 1.0)
    bal = 0.5 * (sens + spec)
    denom = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    mcc = ((tp * tn) - (fp * fn)) / denom
    f1 = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1.0)
    return pd.DataFrame({
        "Threshold": thresholds,
        "Sensitivity": sens,
        "Specificity": spec,
        "Balanced_Accuracy": bal,
        "MCC": mcc,
        "F1": f1,
    })


def choose_threshold(y: np.ndarray, p: np.ndarray, cfg: Config) -> Tuple[float, Dict[str, float]]:
    table = threshold_stats(y, p, np.linspace(0.03, 0.90, 436))
    feasible = table[
        (table["Sensitivity"] >= cfg.min_sensitivity)
        & (table["Specificity"] >= cfg.preferred_specificity)
    ]
    if len(feasible) == 0:
        feasible = table[table["Sensitivity"] >= cfg.min_sensitivity]
    if len(feasible) == 0:
        feasible = table.copy()
    feasible = feasible.assign(
        Objective=0.55 * feasible["Balanced_Accuracy"]
        + 0.25 * feasible["MCC"]
        + 0.20 * feasible["F1"]
    )
    row = feasible.sort_values(
        ["Objective", "Sensitivity", "Specificity"],
        ascending=False,
    ).iloc[0]
    return float(row["Threshold"]), {k: float(v) for k, v in row.items()}


def probabilistic_score(y: np.ndarray, p: np.ndarray) -> float:
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    return 0.52 * pr + 0.30 * roc + 0.18 * (1.0 - brier)


def tune_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, float]]:
    class_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    candidates = candidate_params(name, class_ratio, cfg, seed)
    inner = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params = candidates[0]
    best_oof = None
    candidate_log = []

    for ci, params in enumerate(candidates):
        oof = np.zeros(len(y), dtype=float)
        valid = True
        try:
            for fold, (tr, va) in enumerate(inner.split(X, y)):
                model = make_model(name, params, seed + ci * 17 + fold)
                model.fit(X[tr], y[tr])
                oof[va] = model.predict_proba(X[va])[:, 1]
            score = probabilistic_score(y, oof)
        except Exception as exc:
            valid = False
            score = -np.inf
        candidate_log.append({"Candidate": ci, "Score": score, "Valid": valid, "Params": native(params)})
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_oof = oof.copy()

    if best_oof is None:
        raise RuntimeError(f"No valid {name} candidate.")
    return best_params, best_oof, {"Inner_Objective": float(best_score), "Candidates": candidate_log}


class MATScalarPreprocessor:
    def __init__(self):
        self.columns: List[str] = []
        self.numeric: Dict[str, bool] = {}
        self.medians: Dict[str, float] = {}
        self.iqrs: Dict[str, float] = {}
        self.categories: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame) -> "MATScalarPreprocessor":
        self.columns = list(X.columns)
        for col in self.columns:
            numeric_ratio = safe_numeric(X[col]).notna().mean()
            is_num = pd.api.types.is_numeric_dtype(X[col]) or numeric_ratio >= 0.90
            self.numeric[col] = bool(is_num)
            if is_num:
                s = safe_numeric(X[col])
                med = float(s.median()) if s.notna().any() else 0.0
                q1 = float(s.quantile(0.25)) if s.notna().any() else 0.0
                q3 = float(s.quantile(0.75)) if s.notna().any() else 1.0
                self.medians[col] = med
                self.iqrs[col] = max(q3 - q1, 1e-3)
            else:
                vals = X[col].astype(str).fillna("MISS").str.strip().str.lower()
                cats = sorted(vals.dropna().unique().tolist())
                self.categories[col] = {v: i + 1 for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(X), len(self.columns)), dtype=np.float32)
        masks = np.zeros_like(values)
        for j, col in enumerate(self.columns):
            raw = X[col]
            if self.numeric[col]:
                s = safe_numeric(raw)
                miss = s.isna().to_numpy()
                v = ((s.fillna(self.medians[col]) - self.medians[col]) / self.iqrs[col]).clip(-8, 8)
                values[:, j] = v.to_numpy(np.float32)
                masks[:, j] = miss.astype(np.float32)
            else:
                text = raw.astype(str).where(~raw.isna(), "MISS").str.strip().str.lower()
                mapping = self.categories[col]
                encoded = text.map(mapping).fillna(0).astype(float)
                denom = max(len(mapping), 1)
                values[:, j] = (encoded / denom).to_numpy(np.float32)
                masks[:, j] = raw.isna().to_numpy(np.float32)
        return values, masks


class MATV2(nn.Module):
    def __init__(self, n_features: int, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.03)
        self.value_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.missing_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.03)
        self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.03)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        token = (
            x.unsqueeze(-1) * self.value_weight.unsqueeze(0)
            + self.value_bias.unsqueeze(0)
            + m.unsqueeze(-1) * self.missing_embedding.unsqueeze(0)
            + self.feature_embedding.unsqueeze(0)
        )
        cls = self.cls.expand(x.shape[0], -1, -1)
        z = torch.cat([cls, token], dim=1)
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0])).squeeze(-1)


def focal_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: float, gamma: float = 1.5):
    prob = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    pt = torch.where(y > 0.5, prob, 1 - prob)
    return ((1 - pt) ** gamma * bce).mean()


def train_mat_once(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, MATV2, MATScalarPreprocessor]:
    set_seed(seed)
    y_train = np.asarray(y_train, dtype=int)
    y_valid = np.asarray(y_valid, dtype=int)
    pre = MATScalarPreprocessor().fit(X_train)
    xv, xm = pre.transform(X_train)
    vv, vm = pre.transform(X_valid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MATV2(xv.shape[1], cfg.mat_d_model, cfg.mat_heads, cfg.mat_dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.mat_lr, weight_decay=cfg.mat_weight_decay
    )
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    dataset = TensorDataset(
        torch.tensor(xv), torch.tensor(xm), torch.tensor(y_train.astype(np.float32))
    )
    loader = DataLoader(dataset, batch_size=cfg.mat_batch_size, shuffle=True)

    valid_x = torch.tensor(vv, device=device)
    valid_m = torch.tensor(vm, device=device)

    best_state = None
    best_pr = -np.inf
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(cfg.mat_epochs):
        model.train()
        for bx, bm, by in loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            if cfg.mat_feature_corruption > 0:
                corruption = torch.rand_like(bx) < cfg.mat_feature_corruption
                bx = bx.masked_fill(corruption, 0.0)
                bm = torch.maximum(bm, corruption.float())
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, bm)
            loss = focal_bce(logits, by, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(valid_x, valid_m)).cpu().numpy()
        pr = average_precision_score(y_valid, val_prob)
        if pr > best_pr + 1e-4:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= cfg.mat_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(valid_x, valid_m)).cpu().numpy()
    return pred, model, pre


def mat_oof_and_test(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y_train = np.asarray(y_train, dtype=int)
    inner = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_train), dtype=float)
    fold_logs = []

    for fold, (tr, va) in enumerate(inner.split(X_train, y_train)):
        pred, _, _ = train_mat_once(
            X_train.iloc[tr], y_train[tr], X_train.iloc[va], y_train[va],
            cfg, seed + fold * 101
        )
        oof[va] = pred
        fold_logs.append({
            "Fold": fold,
            "PR_AUC": float(average_precision_score(y_train[va], pred)),
            "ROC_AUC": float(roc_auc_score(y_train[va], pred)),
        })

    # Fit full outer-training model using a small internal early-stopping split,
    # then predict untouched outer test.
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 777)
    fit_idx, val_idx = next(split.split(X_train, y_train))
    _, model, pre = train_mat_once(
        X_train.iloc[fit_idx], y_train[fit_idx],
        X_train.iloc[val_idx], y_train[val_idx],
        cfg, seed + 991
    )
    tv, tm = pre.transform(X_test)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(
            model(torch.tensor(tv, device=device), torch.tensor(tm, device=device))
        ).cpu().numpy()
    return oof, test_prob, {"Inner_Folds": fold_logs}


def group_indices(feature_names: Sequence[str], group: str) -> np.ndarray:
    ids = []
    for i, name in enumerate(feature_names):
        if modality_of(name) == group:
            ids.append(i)
    return np.asarray(ids, dtype=int)


def fit_branch_model(
    branch: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(indices) < 2:
        return np.full(len(y_train), y_train.mean()), np.full(len(X_test), y_train.mean()), {
            "Skipped": True, "Reason": f"{branch} has <2 encoded features"
        }

    Xt = X_train[:, indices]
    Xs = X_test[:, indices]
    if branch == "clinical":
        name = "Random Forest"
    elif branch == "laboratory":
        name = "XGBoost"
    elif branch == "ultrasound":
        name = "CatBoost"
    else:
        name = "Extra Trees"

    branch_cfg = replace(cfg, candidate_trials=min(3, cfg.candidate_trials))
    params, oof, log = tune_model(name, Xt, y_train, branch_cfg, seed)
    model = make_model(name, params, seed + 55)
    model.fit(Xt, y_train)
    test = model.predict_proba(Xs)[:, 1]
    return oof, test, {"Model": name, "Params": native(params), **log}


def optimize_blend(
    y: np.ndarray,
    oof_matrix: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Two-pass blend search.

    Pass 1 ranks convex weights by PR-AUC, ROC-AUC and Brier without using
    thresholds. Pass 2 applies the locked sensitivity/specificity objective
    only to a small shortlist. This is much faster and avoids target chasing
    over tens of thousands of thresholds.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    n_models = oof_matrix.shape[1]
    weights = np.vstack([
        np.eye(n_models),
        np.full((1, n_models), 1.0 / n_models),
        rng.dirichlet(np.full(n_models, 0.65), size=cfg.blend_candidates),
    ])

    shortlist: List[Tuple[float, np.ndarray, np.ndarray, float, float, float]] = []
    for start in range(0, len(weights), 256):
        batch = weights[start:start + 256]
        probs = np.clip(oof_matrix @ batch.T, 1e-6, 1 - 1e-6)
        for j in range(probs.shape[1]):
            p = probs[:, j]
            pr = float(average_precision_score(y, p))
            roc = float(roc_auc_score(y, p))
            brier = float(brier_score_loss(y, p))
            score = 0.52 * pr + 0.30 * roc + 0.18 * (1.0 - brier)
            shortlist.append((score, batch[j].copy(), p.copy(), pr, roc, brier))

    shortlist.sort(key=lambda z: z[0], reverse=True)
    shortlist = shortlist[: min(250, len(shortlist))]

    best = None
    for _, w, p, pr, roc, brier in shortlist:
        threshold, tlog = choose_threshold(y, p, cfg)
        objective = (
            0.32 * pr
            + 0.20 * roc
            + 0.27 * tlog["Balanced_Accuracy"]
            + 0.14 * tlog["MCC"]
            + 0.07 * (1.0 - brier)
        )
        if tlog["Sensitivity"] >= cfg.min_sensitivity:
            objective += 0.01
        if tlog["Specificity"] >= cfg.preferred_specificity:
            objective += 0.005
        record = (
            objective,
            w,
            p,
            threshold,
            {
                "PR_AUC": pr,
                "ROC_AUC": roc,
                "Brier": brier,
                **tlog,
            },
        )
        if best is None or record[0] > best[0]:
            best = record

    assert best is not None
    return best[1], best[2], best[3], best[4]


def platt_fit_apply(y: np.ndarray, train_p: np.ndarray, test_p: np.ndarray):
    train_p = np.clip(train_p, 1e-6, 1 - 1e-6)
    test_p = np.clip(test_p, 1e-6, 1 - 1e-6)
    tr_logit = np.log(train_p / (1 - train_p)).reshape(-1, 1)
    te_logit = np.log(test_p / (1 - test_p)).reshape(-1, 1)
    model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000)
    model.fit(tr_logit, y)
    return model.predict_proba(tr_logit)[:, 1], model.predict_proba(te_logit)[:, 1], model


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "N": int(len(y)),
        "Positive_N": int(y.sum()),
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Precision_PPV": float(precision_score(y, pred, zero_division=0)),
        "Recall_Sensitivity_TPR": float(sens),
        "Specificity_TNR": float(spec),
        "NPV": float(npv),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "F2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Log_Loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def run_outer_fold(
    X_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    repeat_id: int,
    fold_id: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    seed = cfg.seed + repeat_id * 10000 + fold_id * 500
    set_seed(seed)

    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    pre = FoldPreprocessor().fit(X_train_df)
    train_encoded = pre.transform(X_train_df)
    test_encoded = pre.transform(X_test_df)

    selected, selection_table = stable_feature_selection(
        train_encoded, y_train, pre.feature_names_, cfg, seed
    )
    train_selected = train_encoded[:, selected]
    test_selected = test_encoded[:, selected]
    selected_names = [pre.feature_names_[i] for i in selected]

    model_names = ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost", "CatBoost"]
    oof_parts = []
    test_parts = []
    model_logs = {}

    for mi, name in enumerate(model_names):
        params, oof, log = tune_model(name, train_selected, y_train, cfg, seed + mi * 71)
        model = make_model(name, params, seed + mi * 71 + 33)
        model.fit(train_selected, y_train)
        test_prob = model.predict_proba(test_selected)[:, 1]
        oof_parts.append(oof)
        test_parts.append(test_prob)
        model_logs[name] = {"Params": native(params), **log}

    # Modality branches on the fold-safe encoded matrix.
    for bi, branch in enumerate(["clinical", "laboratory", "ultrasound", "other"]):
        idx = group_indices(pre.feature_names_, branch)
        oof, test_prob, log = fit_branch_model(
            branch, train_encoded, y_train, test_encoded, idx, cfg, seed + 1000 + bi * 97
        )
        oof_parts.append(oof)
        test_parts.append(test_prob)
        model_logs[f"{branch.title()} Branch"] = log
        model_names.append(f"{branch.title()} Branch")

    # Proposed compact missingness-aware transformer.
    mat_oof, mat_test, mat_log = mat_oof_and_test(
        X_train_df, y_train, X_test_df, cfg, seed + 2000
    )
    oof_parts.append(mat_oof)
    test_parts.append(mat_test)
    model_names.append("MAT-V2")
    model_logs["MAT-V2"] = mat_log

    oof_matrix = np.column_stack(oof_parts)
    test_matrix = np.column_stack(test_parts)

    weights, blend_oof, _, blend_log = optimize_blend(
        y_train, oof_matrix, cfg, seed + 3000
    )
    blend_test = np.clip(test_matrix @ weights, 1e-6, 1 - 1e-6)

    calibrated_oof, calibrated_test, calibrator = platt_fit_apply(
        y_train, blend_oof, blend_test
    )
    threshold, threshold_log = choose_threshold(y_train, calibrated_oof, cfg)
    fold_metrics = metrics(y_test, calibrated_test, threshold)

    pred_df = pd.DataFrame({
        "Original_Index": test_idx,
        "Repeat": repeat_id + 1,
        "Outer_Fold": fold_id + 1,
        "Truth": y_test,
        "Probability": calibrated_test,
        "Threshold": threshold,
        "Prediction": (calibrated_test >= threshold).astype(int),
    })
    for name, probs in zip(model_names, test_parts):
        pred_df[f"{name}_Probability"] = probs

    selection_table.to_csv(
        output_dir / f"feature_selection_repeat{repeat_id+1}_fold{fold_id+1}.csv",
        index=False,
    )

    log = {
        "Repeat": repeat_id + 1,
        "Outer_Fold": fold_id + 1,
        "Selected_Feature_Count": int(len(selected)),
        "Selected_Features": selected_names,
        "Models": model_logs,
        "Blend_Weights": {name: float(w) for name, w in zip(model_names, weights)},
        "Blend_Training": blend_log,
        "Threshold_Training": threshold_log,
        "Outer_Test_Metrics": fold_metrics,
    }
    with open(output_dir / f"fold_log_repeat{repeat_id+1}_fold{fold_id+1}.json", "w") as f:
        json.dump(native(log), f, indent=2)

    return pred_df, log


def preflight_self_test(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(size=(72, 10)).astype(np.float32)
    y = np.array([0] * 48 + [1] * 24, dtype=int)
    rng.shuffle(y)
    ratio = float((y == 0).sum() / (y == 1).sum())

    tiny_models = {
        "Logistic Regression": {"C": 0.1, "class_weight": "balanced"},
        "Random Forest": {
            "n_estimators": 20, "max_depth": 4, "min_samples_leaf": 2,
            "max_features": "sqrt", "class_weight": "balanced",
        },
        "Extra Trees": {
            "n_estimators": 20, "max_depth": 4, "min_samples_leaf": 2,
            "max_features": "sqrt", "class_weight": "balanced",
        },
        "XGBoost": {
            "n_estimators": 15, "max_depth": 2, "learning_rate": 0.05,
            "min_child_weight": 2.0, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 2.0,
            "gamma": 0.0, "scale_pos_weight": ratio,
        },
        "CatBoost": {
            "iterations": 15, "depth": 3, "learning_rate": 0.05,
            "l2_leaf_reg": 3.0, "random_strength": 0.2,
            "bagging_temperature": 0.0, "class_weights": [1.0, ratio],
        },
    }
    for name, params in tiny_models.items():
        model = make_model(name, params, cfg.seed)
        model.fit(X, y)
        prob = model.predict_proba(X[:5])[:, 1]
        if prob.shape != (5,) or not np.isfinite(prob).all():
            raise RuntimeError(f"Preflight failed for {name}.")

    mat = MATV2(n_features=10, d_model=cfg.mat_d_model, heads=cfg.mat_heads, dropout=cfg.mat_dropout)
    logits = mat(torch.tensor(X[:4]), torch.zeros((4, 10), dtype=torch.float32))
    if logits.shape != (4,) or not torch.isfinite(logits).all():
        raise RuntimeError("Preflight failed for MAT-V2.")

    _ = threshold_stats(y, rng.random(len(y)), np.linspace(0.1, 0.9, 9))
    json.dumps(native({"x": np.int64(1), "y": np.float64(0.5)}))
    print("PRE-FLIGHT SELF-TEST PASSED: classical APIs, MAT-V2, thresholds and JSON.")


def main(cfg: Config = CFG):
    set_seed(cfg.seed)
    mount_drive()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("V3.3 targets are evaluation goals, never guaranteed settings.")
    preflight_self_test(cfg)

    X_raw, y_series, leakage_audit = load_corrected_cohort(cfg)
    X = add_engineered_features(X_raw)
    y = y_series.to_numpy(int)

    cohort_signature = hashlib.sha256(
        ("|".join(map(str, X.columns)) + "|" + "".join(map(str, y.tolist())) + json.dumps(native(asdict(cfg)), sort_keys=True)).encode()
    ).hexdigest()[:16]
    run_id = f"v33_{cohort_signature}"
    out_root = Path(cfg.output_root)
    if not Path("/content/drive/MyDrive").exists():
        out_root = Path("/content/MAT-Appendix/v33_runs")
    output_dir = out_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    leakage_audit.to_csv(output_dir / "leakage_audit.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(native(asdict(cfg)), f, indent=2)

    print("\nCORRECTED COHORT")
    print(f"N={len(X)}, complicated={int(y.sum())}, uncomplicated={int((1-y).sum())}")
    print(f"Features after engineering={X.shape[1]}")
    print("\nSTRICT LEAKAGE AUDIT")
    print(leakage_audit.to_string(index=False) if len(leakage_audit) else "No named leakage features present in X.")

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.outer_folds,
        n_repeats=cfg.repeats,
        random_state=cfg.seed,
    )

    all_predictions = []
    fold_logs = []
    for split_number, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_id = split_number // cfg.outer_folds
        fold_id = split_number % cfg.outer_folds
        print(f"\n===== REPEAT {repeat_id+1}/{cfg.repeats}, OUTER FOLD {fold_id+1}/{cfg.outer_folds} =====")
        checkpoint_csv = output_dir / f"fold_predictions_repeat{repeat_id+1}_fold{fold_id+1}.csv"
        checkpoint_json = output_dir / f"fold_log_repeat{repeat_id+1}_fold{fold_id+1}.json"
        if checkpoint_csv.exists() and checkpoint_json.exists():
            print("Loading completed fold checkpoint.")
            pred_df = pd.read_csv(checkpoint_csv)
            with open(checkpoint_json) as f:
                log = json.load(f)
        else:
            pred_df, log = run_outer_fold(
                X, y, train_idx, test_idx, cfg, repeat_id, fold_id, output_dir
            )
            pred_df.to_csv(checkpoint_csv, index=False)
        all_predictions.append(pred_df)
        fold_logs.append(log)
        print("Fold metrics:", log["Outer_Test_Metrics"])
        pd.concat(all_predictions, ignore_index=True).to_csv(
            output_dir / "all_outer_predictions_partial.csv", index=False
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "all_repeated_nested_predictions.csv", index=False)

    repeat_rows = []
    for repeat in sorted(predictions["Repeat"].unique()):
        sub = predictions[predictions["Repeat"] == repeat].sort_values("Original_Index")
        # Fold-specific thresholds were learned without each test sample.
        pred = sub["Prediction"].to_numpy(int)
        truth = sub["Truth"].to_numpy(int)
        prob = sub["Probability"].to_numpy(float)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        repeat_rows.append({
            "Repeat": int(repeat),
            "Accuracy": float(accuracy_score(truth, pred)),
            "Balanced_Accuracy": float(balanced_accuracy_score(truth, pred)),
            "Precision_PPV": float(precision_score(truth, pred, zero_division=0)),
            "Recall_Sensitivity_TPR": float(sens),
            "Specificity_TNR": float(spec),
            "F1": float(f1_score(truth, pred, zero_division=0)),
            "F2": float(fbeta_score(truth, pred, beta=2, zero_division=0)),
            "MCC": float(matthews_corrcoef(truth, pred)),
            "ROC_AUC": float(roc_auc_score(truth, prob)),
            "PR_AUC": float(average_precision_score(truth, prob)),
            "Brier": float(brier_score_loss(truth, prob)),
            "Log_Loss": float(log_loss(truth, np.clip(prob, 1e-6, 1 - 1e-6))),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        })

    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_metrics.to_csv(output_dir / "repeat_metrics.csv", index=False)

    summary_rows = []
    for col in repeat_metrics.columns:
        if col == "Repeat":
            continue
        summary_rows.append({
            "Metric": col,
            "Mean": float(repeat_metrics[col].mean()),
            "Std": float(repeat_metrics[col].std(ddof=1)) if len(repeat_metrics) > 1 else 0.0,
            "Min": float(repeat_metrics[col].min()),
            "Max": float(repeat_metrics[col].max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    # Cross-fitted per-patient consensus across repeats.
    consensus = predictions.groupby("Original_Index", as_index=False).agg(
        Truth=("Truth", "first"),
        Mean_Probability=("Probability", "mean"),
        Positive_Vote_Rate=("Prediction", "mean"),
    )
    consensus["Consensus_Prediction"] = (consensus["Positive_Vote_Rate"] >= 0.5).astype(int)
    consensus.to_csv(output_dir / "per_patient_consensus.csv", index=False)

    truth = consensus["Truth"].to_numpy(int)
    prob = consensus["Mean_Probability"].to_numpy(float)
    pred = consensus["Consensus_Prediction"].to_numpy(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    consensus_metrics = {
        "Accuracy": float(accuracy_score(truth, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(truth, pred)),
        "Precision_PPV": float(precision_score(truth, pred, zero_division=0)),
        "Recall_Sensitivity_TPR": float(tp / max(tp + fn, 1)),
        "Specificity_TNR": float(tn / max(tn + fp, 1)),
        "F1": float(f1_score(truth, pred, zero_division=0)),
        "F2": float(fbeta_score(truth, pred, beta=2, zero_division=0)),
        "MCC": float(matthews_corrcoef(truth, pred)),
        "ROC_AUC": float(roc_auc_score(truth, prob)),
        "PR_AUC": float(average_precision_score(truth, prob)),
        "Brier": float(brier_score_loss(truth, prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    with open(output_dir / "consensus_metrics.json", "w") as f:
        json.dump(native(consensus_metrics), f, indent=2)

    # MAT contribution audit from actual fold weights.
    mat_weights = []
    for log in fold_logs:
        mat_weights.append(float(log["Blend_Weights"].get("MAT-V2", 0.0)))
    mat_audit = {
        "Mean_MAT_Weight": float(np.mean(mat_weights)),
        "Median_MAT_Weight": float(np.median(mat_weights)),
        "Zero_or_NearZero_Fraction": float(np.mean(np.asarray(mat_weights) < 0.01)),
        "All_Weights": mat_weights,
        "Decision": (
            "Retain MAT-V2 as a contributing proposed branch"
            if np.mean(mat_weights) >= 0.05
            else "MAT-V2 contribution is weak; do not claim it as the performance driver"
        ),
    }
    with open(output_dir / "mat_contribution_audit.json", "w") as f:
        json.dump(native(mat_audit), f, indent=2)

    bundle = {
        "config": asdict(cfg),
        "repeat_metrics": repeat_metrics,
        "summary": summary,
        "consensus": consensus,
        "consensus_metrics": consensus_metrics,
        "fold_logs": fold_logs,
        "mat_contribution_audit": mat_audit,
    }
    with open(output_dir / "mat_appendix_v33_reproducibility.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nREPEATED NESTED-CV METRICS")
    print(repeat_metrics.to_string(index=False))
    print("\nMEAN ± STD")
    for _, row in summary.iterrows():
        print(f"{row['Metric']:28s}: {row['Mean']:.6f} ± {row['Std']:.6f}")
    print("\nCONSENSUS METRICS")
    for k, v in consensus_metrics.items():
        print(f"{k:28s}: {v:.6f}" if isinstance(v, float) else f"{k:28s}: {v}")
    print("\nMAT CONTRIBUTION AUDIT")
    print(json.dumps(native(mat_audit), indent=2))
    print("\nSaved to:", output_dir)
    print(
        "\nSCIENTIFIC VERDICT: Report achieved metrics exactly. "
        "Do not restore direct complication labels or inspect outer-test folds to force targets."
    )
    return bundle


V33_RESULT = main(CFG)
