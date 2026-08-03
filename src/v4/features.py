from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from ucimlrepo import fetch_ucirepo

from core import Config, canonical, find_column, normalize_text


LEAKAGE_EXACT = {
    "diagnosis",
    "severity",
    "management",
    "length_of_stay",
    "peritonitis",
    "perforation",
    "appendicular_abscess",
    "abscess_location",
    "us_number",
}
LEAKAGE_PARTIAL = (
    "histology",
    "histopath",
    "pathology",
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


def modality(name: str) -> str:
    key = canonical(name)
    laboratory = (
        "wbc",
        "leuk",
        "crp",
        "neut",
        "lymph",
        "platelet",
        "hemoglobin",
        "haemoglobin",
        "hematocrit",
        "rbc",
        "eryth",
        "mcv",
        "mch",
        "rdw",
        "bilirubin",
        "creatin",
        "sodium",
        "potassium",
        "urine",
        "ketone",
    )
    ultrasound = (
        "appendix",
        "diameter",
        "ultrasound",
        "sonograph",
        "us_",
        "_us",
        "free_fluid",
        "fluid",
        "compress",
        "hyperemia",
        "perfusion",
        "echogenic",
        "coprostasis",
        "target_sign",
        "lymph_node",
        "bowel_wall",
        "meteorism",
        "ileus",
        "conglomerate",
        "gynecological",
    )
    clinical = (
        "age",
        "sex",
        "gender",
        "bmi",
        "height",
        "weight",
        "duration",
        "pain",
        "vomit",
        "nausea",
        "fever",
        "temperature",
        "rebound",
        "guarding",
        "tender",
        "migration",
        "anorexia",
        "appetite",
        "diarr",
        "dysuria",
        "stool",
        "score",
        "pas",
        "alvarado",
        "psoas",
        "rovsing",
        "cough",
        "percussion",
    )
    if any(token in key for token in laboratory):
        return "laboratory"
    if any(token in key for token in ultrasound):
        return "ultrasound"
    if any(token in key for token in clinical):
        return "clinical"
    return "other"


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_cohort(config: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dataset = fetch_ucirepo(id=938)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    diagnosis_column = find_column(targets.columns, ["Diagnosis"])
    severity_column = find_column(targets.columns, ["Severity"])
    if diagnosis_column is None or severity_column is None:
        raise RuntimeError(f"Required target columns were not found: {list(targets.columns)}")

    diagnosis = normalize_text(targets[diagnosis_column])
    severity = normalize_text(targets[severity_column])
    keep = (
        diagnosis.eq("appendicitis")
        & severity.isin({"complicated", "uncomplicated"})
    )
    features = features.loc[keep].reset_index(drop=True)
    labels = (
        severity.loc[keep].reset_index(drop=True).eq("complicated").astype(int)
    )

    retained: list[str] = []
    audit_rows: list[dict[str, str]] = []
    for column in features.columns:
        key = canonical(column)
        reason: str | None = None
        if key in LEAKAGE_EXACT:
            reason = "identifier, target, post-decision, or direct endpoint component"
        elif any(token in key for token in LEAKAGE_PARTIAL) and "lymph_node" not in key:
            reason = "pathology, treatment, or direct complication proxy"
        if reason is None:
            retained.append(str(column))
            audit_rows.append({"feature": str(column), "action": "KEEP", "reason": "pre-decision candidate"})
        else:
            audit_rows.append({"feature": str(column), "action": "DROP", "reason": reason})

    features = features[retained].copy()
    observed = (len(features), int(labels.sum()), int((1 - labels).sum()))
    expected = (config.expected_n, config.expected_positive, config.expected_negative)
    if observed != expected:
        raise RuntimeError(f"Corrected cohort signature changed: {observed} != {expected}")

    return features, labels, pd.DataFrame(audit_rows)


def _product(frame: pd.DataFrame, name: str, left: Sequence[str], right: Sequence[str]) -> None:
    left_column = find_column(frame.columns, left)
    right_column = find_column(frame.columns, right)
    if left_column and right_column:
        frame[name] = numeric(frame[left_column]) * numeric(frame[right_column])


def _ratio(frame: pd.DataFrame, name: str, numerator: Sequence[str], denominator: Sequence[str]) -> None:
    numerator_column = find_column(frame.columns, numerator)
    denominator_column = find_column(frame.columns, denominator)
    if numerator_column and denominator_column:
        denominator_values = numeric(frame[denominator_column]).replace(0, np.nan)
        frame[name] = numeric(frame[numerator_column]) / denominator_values


def _log_feature(frame: pd.DataFrame, candidates: Sequence[str]) -> None:
    column = find_column(frame.columns, candidates)
    if column:
        values = numeric(frame[column]).clip(lower=0)
        frame[f"Log1p_{canonical(column)}"] = np.log1p(values)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, label-free features.

    Distribution-dependent transforms are deliberately left to the fold-fitted
    preprocessing pipeline to prevent information leakage.
    """
    frame = raw.copy()
    groups = {
        group: [column for column in frame.columns if modality(str(column)) == group]
        for group in ("clinical", "laboratory", "ultrasound", "other")
    }

    frame["Missing_Total"] = frame.isna().sum(axis=1).astype(float)
    frame["Observed_Total"] = frame.notna().sum(axis=1).astype(float)
    frame["Missing_Fraction"] = frame.isna().mean(axis=1).astype(float)
    for group, columns in groups.items():
        if columns:
            frame[f"Missing_{group.title()}"] = frame[columns].isna().sum(axis=1).astype(float)
            frame[f"Observed_{group.title()}"] = frame[columns].notna().sum(axis=1).astype(float)
            frame[f"MissingFraction_{group.title()}"] = frame[columns].isna().mean(axis=1).astype(float)
        else:
            frame[f"Missing_{group.title()}"] = 0.0
            frame[f"Observed_{group.title()}"] = 0.0
            frame[f"MissingFraction_{group.title()}"] = 1.0

    # Explicit indicators let models exploit informative data availability without
    # interpreting an imputed value as an observed measurement.
    for column in list(raw.columns):
        if raw[column].isna().any():
            frame[f"{column}__Missing"] = raw[column].isna().astype(np.int8)

    _product(frame, "CRP_x_WBC", ["CRP"], ["WBC", "Leukocytes", "WBC_Count"])
    _product(
        frame,
        "CRP_x_Neutrophils",
        ["CRP"],
        ["Neutrophil_Percentage", "Neutrophils"],
    )
    _product(frame, "CRP_x_AppendixDiameter", ["CRP"], ["Appendix_Diameter"])
    _product(
        frame,
        "WBC_x_AppendixDiameter",
        ["WBC", "Leukocytes", "WBC_Count"],
        ["Appendix_Diameter"],
    )
    _product(
        frame,
        "Duration_x_CRP",
        ["Symptoms_Duration", "Duration_of_Symptoms", "Duration"],
        ["CRP"],
    )
    _product(
        frame,
        "Duration_x_WBC",
        ["Symptoms_Duration", "Duration_of_Symptoms", "Duration"],
        ["WBC", "Leukocytes", "WBC_Count"],
    )
    _product(
        frame,
        "Alvarado_x_PAS",
        ["Alvarado_Score"],
        ["Paedriatic_Appendicitis_Score", "Pediatric_Appendicitis_Score"],
    )
    _product(
        frame,
        "ClinicalScore_x_AppendixDiameter",
        ["Paedriatic_Appendicitis_Score", "Pediatric_Appendicitis_Score", "Alvarado_Score"],
        ["Appendix_Diameter"],
    )
    _ratio(frame, "CRP_WBC_Ratio", ["CRP"], ["WBC", "Leukocytes", "WBC_Count"])
    _ratio(
        frame,
        "Neutrophil_WBC_Ratio",
        ["Neutrophil_Percentage", "Neutrophils"],
        ["WBC", "Leukocytes", "WBC_Count"],
    )
    _ratio(frame, "AppendixDiameter_Age_Ratio", ["Appendix_Diameter"], ["Age"])
    _ratio(frame, "CRP_Age_Ratio", ["CRP"], ["Age"])

    for candidates in (
        ["CRP"],
        ["WBC", "Leukocytes", "WBC_Count"],
        ["Neutrophil_Percentage", "Neutrophils"],
        ["Appendix_Diameter"],
        ["Symptoms_Duration", "Duration_of_Symptoms", "Duration"],
    ):
        _log_feature(frame, candidates)

    return frame.replace([np.inf, -np.inf], np.nan)


def availability_features(raw: pd.DataFrame) -> np.ndarray:
    ultrasound_columns = [c for c in raw.columns if modality(str(c)) == "ultrasound"]
    laboratory_columns = [c for c in raw.columns if modality(str(c)) == "laboratory"]
    clinical_columns = [c for c in raw.columns if modality(str(c)) == "clinical"]

    def observed_fraction(columns: list[str]) -> np.ndarray:
        if not columns:
            return np.zeros(len(raw), dtype=float)
        return raw[columns].notna().mean(axis=1).to_numpy(dtype=float)

    appendix_visible = find_column(raw.columns, ["Appendix_on_US", "Appendix_Visible"])
    ultrasound_performed = find_column(raw.columns, ["US_Performed", "Ultrasound_Performed"])

    def binary_availability(column: str | None, fallback: np.ndarray) -> np.ndarray:
        if column is None:
            return fallback
        values = normalize_text(raw[column])
        output = np.full(len(raw), 0.5, dtype=float)
        positive = {"yes", "true", "1", "performed", "present", "visible"}
        negative = {"no", "false", "0", "not performed", "absent", "not visible"}
        output[values.isin(positive).to_numpy()] = 1.0
        output[values.isin(negative).to_numpy()] = 0.0
        return output

    ultrasound_observed = observed_fraction(ultrasound_columns)
    laboratory_observed = observed_fraction(laboratory_columns)
    clinical_observed = observed_fraction(clinical_columns)
    performed = binary_availability(ultrasound_performed, (ultrasound_observed > 0).astype(float))
    visible = binary_availability(appendix_visible, (ultrasound_observed > 0).astype(float))

    return np.column_stack(
        [
            clinical_observed,
            laboratory_observed,
            ultrasound_observed,
            raw.notna().mean(axis=1).to_numpy(dtype=float),
            performed,
            visible,
        ]
    ).astype(np.float32)


@dataclass
class FoldPreprocessor:
    min_category_frequency: int = 3

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        self.numeric_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            or numeric(frame[column]).notna().mean() >= 0.90
        ]
        self.categorical_columns = [
            column for column in frame.columns if column not in self.numeric_columns
        ]

        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
            ]
        )
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=self.min_category_frequency,
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", encoder),
            ]
        )

        self.transformer = ColumnTransformer(
            [
                ("numeric", numeric_pipe, self.numeric_columns),
                ("categorical", categorical_pipe, self.categorical_columns),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        ).fit(frame)
        self.feature_names = [str(name) for name in self.transformer.get_feature_names_out()]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.transformer.transform(frame), dtype=np.float32)


def modality_indices(feature_names: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    selected = [
        index
        for index, name in enumerate(feature_names)
        if modality(name) in set(groups)
    ]
    return np.asarray(selected, dtype=int)


def missingness_indices(feature_names: Sequence[str]) -> np.ndarray:
    selected = [
        index
        for index, name in enumerate(feature_names)
        if "missing" in canonical(name) or "indicator" in canonical(name)
    ]
    return np.asarray(selected, dtype=int)
