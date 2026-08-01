"""MAT-Appendix V3.2 Drive/GitHub autoload runner.

Loads the latest completed V3.1 OOF predictions from Google Drive or /content,
reconstructs a wide OOF table from CSV/PKL artifacts, and then executes the
GitHub-only V3.2 constrained blend. No model retraining is performed.
"""
from __future__ import annotations
from pathlib import Path
import pickle, re, urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/main"
V32_URL = REPO_RAW + "/src/v32/v32_github_only_plain.py?version=20260801_v32_autoload3"
EXPECTED_N = 463
OUTER_SEED_FALLBACK = 2026
TARGET_PATTERNS = [r"^y_true$", r"^true_y$", r"^true_label$", r"^target$", r"^label$", r"^outcome$", r"^severity$", r"^complicated$", r"^class$", r"^y$"]
FOLD_PATTERNS = [r"^outer_fold$", r"^outerfold$", r"^outer_fold_id$", r"^fold$", r"^fold_id$", r"^cv_fold$", r"^test_fold$", r"^outer_cv_fold$"]
MODEL_PATTERNS = [r"^model$", r"^model_name$", r"^learner$", r"^estimator$"]
ID_PATTERNS = [r"^row_id$", r"^row_index$", r"^sample_id$", r"^patient_id$", r"^subject_id$", r"^source_row_id$", r"^index$", r"^id$"]
PROB_HINTS = ("prob", "score", "oof", "pred", "logistic", "random", "forest", "extra", "xgb", "xgboost", "catboost", "mat", "ensemble", "stack", "histgradient", "hist_gradient", "histgb")


def first_match(columns: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        regex = re.compile(pattern, re.I)
        for col in columns:
            if regex.search(str(col).strip()):
                return str(col)
    return None


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def binary(s: pd.Series) -> bool:
    x = numeric(s).dropna()
    return len(x) >= 50 and set(np.unique(x).tolist()) == {0, 1}


def fold_like(s: pd.Series) -> bool:
    x = numeric(s).dropna()
    return len(x) >= 50 and np.allclose(x, np.round(x)) and 3 <= x.nunique() <= 20


def probability(s: pd.Series, name: str = "") -> bool:
    x = numeric(s).dropna()
    return len(x) >= 50 and x.nunique() >= 8 and ((x >= 0) & (x <= 1)).mean() >= 0.99 and any(h in str(name).lower() for h in PROB_HINTS)


def normalize_target(s: pd.Series) -> pd.Series:
    x = numeric(s)
    vals = sorted(x.dropna().unique().tolist())
    if vals == [0, 1]:
        return x.astype(int)
    if len(vals) == 2:
        return x.map({vals[0]: 0, vals[1]: 1}).astype(int)
    text = s.astype(str).str.strip().str.lower()
    pos = text.str.contains("complicated|positive|yes|true|class.?1", regex=True)
    neg = text.str.contains("uncomplicated|negative|no|false|class.?0", regex=True)
    if (pos | neg).all() and pos.any() and neg.any():
        return pos.astype(int)
    raise ValueError("Could not normalize target to 0/1.")


def model_name(name: str) -> str:
    n = str(name).strip()
    n = re.sub(r"(?i)(?:^|[_\-\s])(oof|calibrated|raw|test|val|validation|probability|prob|score|prediction|pred)(?:$|[_\-\s])", " ", n)
    n = re.sub(r"[_\-]+", " ", n)
    return re.sub(r"\s+", " ", n).strip() or str(name)


def wide_candidate(df: pd.DataFrame, source: str):
    if len(df) < 50:
        return None
    cols = [str(c) for c in df.columns]
    target = first_match(cols, TARGET_PATTERNS)
    if target is None:
        bins = [c for c in cols if binary(df[c])]
        preferred = [c for c in bins if any(k in c.lower() for k in ("true", "target", "label", "outcome", "complicated"))]
        target = (preferred or bins or [None])[0]
    if target is None:
        return None
    fold = first_match(cols, FOLD_PATTERNS)
    if fold is None:
        folds = [c for c in cols if c != target and fold_like(df[c])]
        preferred = [c for c in folds if "fold" in c.lower()]
        fold = (preferred or folds or [None])[0]
    probs = [c for c in cols if c not in {target, fold} and probability(df[c], c)]
    if len(probs) < 2:
        for c in cols:
            if c in {target, fold} or c in probs:
                continue
            x = numeric(df[c]).dropna()
            if len(x) >= 50 and x.nunique() >= 8 and ((x >= 0) & (x <= 1)).mean() >= 0.99 and any(h in c.lower() for h in ("logistic", "random", "forest", "extra", "xgb", "cat", "mat", "stack", "ensemble")):
                probs.append(c)
    if len(probs) < 2:
        return None
    out = pd.DataFrame({"y_true": normalize_target(df[target])})
    if fold is not None:
        out["outer_fold"] = numeric(df[fold]).astype(int)
    used = {}
    for col in probs:
        name = model_name(col)
        used[name] = used.get(name, 0) + 1
        final = name if used[name] == 1 else f"{name} {used[name]}"
        out[final] = np.clip(numeric(df[col]).to_numpy(float), 0, 1)
    if out.isna().any().any():
        return None
    score = 100 + 10 * len(probs) + max(0, 20 - abs(len(out) - EXPECTED_N) / 10) + (30 if fold else 0) + (20 if "oof" in source.lower() or "prediction" in source.lower() else 0)
    return score, out, f"wide:{source}"


def long_candidate(df: pd.DataFrame, source: str):
    if len(df) < 100:
        return None
    cols = [str(c) for c in df.columns]
    mcol, tcol, fcol = first_match(cols, MODEL_PATTERNS), first_match(cols, TARGET_PATTERNS), first_match(cols, FOLD_PATTERNS)
    if mcol is None or tcol is None:
        return None
    pcols = [c for c in cols if c not in {mcol, tcol, fcol} and probability(df[c], c)]
    if not pcols:
        pcols = [c for c in cols if c not in {mcol, tcol, fcol} and len(numeric(df[c]).dropna()) >= 50 and numeric(df[c]).dropna().nunique() >= 8 and ((numeric(df[c]).dropna() >= 0) & (numeric(df[c]).dropna() <= 1)).mean() >= 0.99]
    if not pcols:
        return None
    pcol, idcol = pcols[0], first_match(cols, ID_PATTERNS)
    keep = [mcol, tcol, pcol] + ([fcol] if fcol else []) + ([idcol] if idcol else [])
    temp = df[keep].copy()
    temp["_model"], temp["_prob"], temp["_y"] = temp[mcol].map(model_name), numeric(temp[pcol]), normalize_target(temp[tcol])
    if fcol:
        temp["_fold"] = numeric(temp[fcol]).astype(int)
    temp["_sample"] = temp[idcol].astype(str) if idcol else temp.groupby("_model", sort=False).cumcount().astype(str)
    idx = ["_sample", "_y"] + (["_fold"] if fcol else [])
    pivot = temp.pivot_table(index=idx, columns="_model", values="_prob", aggfunc="first").reset_index().rename(columns={"_y": "y_true", "_fold": "outer_fold"})
    pivot = pivot.drop(columns=["_sample"])
    models = [c for c in pivot.columns if c not in {"y_true", "outer_fold"}]
    if len(models) < 2 or len(pivot) < 50 or pivot[["y_true"] + models].isna().any().any():
        return None
    score = 120 + 10 * len(models) + max(0, 20 - abs(len(pivot) - EXPECTED_N) / 10) + (30 if fcol else 0)
    return score, pivot, f"long:{source}"


def dataframe_candidates(df: pd.DataFrame, source: str):
    out = []
    for fn in (wide_candidate, long_candidate):
        try:
            c = fn(df, source)
            if c is not None:
                out.append(c)
        except Exception:
            pass
    return out


def walk_frames(obj: Any, path="root", depth=0, seen=None):
    if seen is None:
        seen = set()
    if depth > 9 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, pd.DataFrame):
        yield path, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from walk_frames(v, f"{path}.{k}", depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_frames(v, f"{path}[{i}]", depth + 1, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, bytearray, np.ndarray)):
        try:
            yield from walk_frames(vars(obj), f"{path}.__dict__", depth + 1, seen)
        except Exception:
            pass


def collect_arrays(obj: Any, path="root", depth=0, seen=None, out=None):
    if out is None:
        out = []
    if seen is None:
        seen = set()
    if depth > 10 or id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, pd.Series):
        arr = obj.to_numpy()
        if arr.ndim == 1 and len(arr) >= 50:
            out.append((path, arr))
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 1 and len(obj) >= 50:
            out.append((path, obj))
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            collect_arrays(v, f"{path}.{k}", depth + 1, seen, out)
    elif isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if arr.ndim == 1 and len(arr) >= 50 and arr.dtype != object:
                out.append((path, arr))
                return out
        except Exception:
            pass
        for i, v in enumerate(obj):
            collect_arrays(v, f"{path}[{i}]", depth + 1, seen, out)
    return out


def array_candidate(obj: Any, source: str):
    arrays = collect_arrays(obj)
    counts = {}
    for _, arr in arrays:
        counts[len(arr)] = counts.get(len(arr), 0) + 1
    for n in sorted(counts, key=lambda x: (counts[x] + (10 if x == EXPECTED_N else 0), -abs(x - EXPECTED_N)), reverse=True):
        group = [(name, np.asarray(arr).ravel()) for name, arr in arrays if len(arr) == n]
        targets, folds, probs = [], [], []
        for name, arr in group:
            x = numeric(pd.Series(arr)).dropna()
            if len(x) != n:
                continue
            lname = name.lower()
            vals = set(np.unique(x).tolist())
            if vals == {0, 1}:
                targets.append((10 if any(k in lname for k in ("y_true", "target", "label", "outcome", "severity", "complicated")) else 0, name, x.to_numpy(int)))
            if np.allclose(x, np.round(x)) and 3 <= x.nunique() <= 20:
                folds.append((10 if "fold" in lname else 0, name, x.to_numpy(int)))
            if ((x >= 0) & (x <= 1)).all() and x.nunique() >= 8 and any(h in lname for h in PROB_HINTS):
                probs.append((sum(3 for h in PROB_HINTS if h in lname), name, x.to_numpy(float)))
        if not targets or len(probs) < 2:
            continue
        targets.sort(reverse=True, key=lambda z: z[0]); probs.sort(reverse=True, key=lambda z: z[0]); folds.sort(reverse=True, key=lambda z: z[0])
        data = {"y_true": targets[0][2]}
        if folds:
            data["outer_fold"] = folds[0][2]
        used = {}
        for _, name, p in probs:
            m = model_name(name.split(".")[-1])
            used[m] = used.get(m, 0) + 1
            data[m if used[m] == 1 else f"{m} {used[m]}"] = np.clip(p, 0, 1)
        out = pd.DataFrame(data)
        models = [c for c in out.columns if c not in {"y_true", "outer_fold"}]
        if len(models) >= 2:
            score = 80 + 8 * len(models) + (25 if folds else 0) + max(0, 20 - abs(n - EXPECTED_N) / 10)
            return score, out, f"arrays:{source}"
    return None


def add_fallback_fold(df: pd.DataFrame):
    if "outer_fold" in df.columns and df["outer_fold"].nunique() >= 3:
        return df, "artifact"
    from sklearn.model_selection import StratifiedKFold
    y = df["y_true"].to_numpy(int)
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED_FALLBACK)
    for fold_id, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[test_idx] = fold_id
    out = df.copy(); out.insert(1, "outer_fold", folds)
    return out, f"reconstructed_stratified_seed_{OUTER_SEED_FALLBACK}"


def mount_drive():
    try:
        from google.colab import drive
    except Exception:
        return
    if Path("/content/drive/MyDrive").exists():
        print("Google Drive already mounted.")
    else:
        print("Mounting Google Drive to load V3.1 artifacts...")
        drive.mount("/content/drive")


def artifact_files():
    roots = [Path("/content/drive/MyDrive/MAT-Appendix"), Path("/content/MAT-Appendix"), Path("/content/drive/MyDrive")]
    patterns = ("*oof*.csv", "*prediction*.csv", "*predictions*.csv", "*optimized*v31*.pkl", "*reproducibility*.pkl", "*.pkl")
    found = {}
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() and "v32_results" not in str(path):
                    found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: (p.stat().st_mtime, p.suffix.lower() == ".csv"), reverse=True)


def main(namespace: Dict[str, Any]):
    mount_drive()
    files = artifact_files()
    if not files:
        raise RuntimeError("No V3.1 CSV/PKL artifact found under MyDrive/MAT-Appendix or /content/MAT-Appendix. The completed V3.1 run must have saved its results to Drive.")
    print(f"Found {len(files)} possible artifact files. Inspecting newest candidates...")
    for p in files[:20]:
        print(" -", p)
    candidates = []
    for path in files[:80]:
        try:
            if path.suffix.lower() == ".csv":
                candidates.extend(dataframe_candidates(pd.read_csv(path), str(path)))
            elif path.suffix.lower() in {".pkl", ".pickle"}:
                with path.open("rb") as f:
                    obj = pickle.load(f)
                for obj_path, df in walk_frames(obj):
                    candidates.extend(dataframe_candidates(df, f"{path}:{obj_path}"))
                c = array_candidate(obj, str(path))
                if c is not None:
                    candidates.append(c)
        except Exception as exc:
            print(f"Skipping {path.name}: {type(exc).__name__}: {exc}")
    if not candidates:
        raise RuntimeError("V3.1 files were found, but no valid OOF prediction table could be reconstructed. Required: binary target plus at least two model probability arrays.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, df, source = candidates[0]
    df, fold_source = add_fallback_fold(df)
    base = ["y_true", "outer_fold"]
    models, signatures = [], set()
    for col in df.columns:
        if col in base:
            continue
        x = numeric(df[col])
        if x.isna().any() or ((x < 0) | (x > 1)).any() or x.nunique() < 8:
            continue
        sig = np.round(x.to_numpy(float), 12).tobytes()
        if sig not in signatures:
            signatures.add(sig); models.append(col)
    df = df[base + models].copy()
    if len(models) < 2:
        raise RuntimeError("Reconstructed artifact has fewer than two distinct probability columns.")
    print("\nSelected artifact:", source)
    print("OOF shape:", df.shape)
    print("Target counts:", df["y_true"].value_counts().sort_index().to_dict())
    print("Outer folds:", sorted(df["outer_fold"].unique().tolist()))
    print("Fold source:", fold_source)
    print("Probability columns:", models)
    out = Path("/content/drive/MyDrive/MAT-Appendix/v32_results") if Path("/content/drive/MyDrive").exists() else Path("/content/MAT-Appendix/v32_results")
    out.mkdir(parents=True, exist_ok=True)
    reconstructed = out / "v31_oof_reconstructed_for_v32.csv"
    df.to_csv(reconstructed, index=False)
    print("Saved reconstructed OOF table:", reconstructed)
    namespace["v31_oof_df"] = df
    source_v32 = urllib.request.urlopen(V32_URL).read().decode("utf-8")
    print("\nStarting V3.2 constrained blend without retraining...\n")
    exec(compile(source_v32, V32_URL, "exec"), namespace, namespace)


main(globals())
