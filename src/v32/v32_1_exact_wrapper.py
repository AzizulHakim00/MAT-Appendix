"""MAT-Appendix V3.2.1 exact Drive wrapper.

Loads only the MAT-Appendix V3.1 OOF table (463 patients, 118 positives),
reconstructs the original outer folds with seed 2026, patches NumPy JSON
serialization, and executes the verified V3.2 blend core. No retraining.
"""
from pathlib import Path
import hashlib
import json
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

EXPECTED_N = 463
EXPECTED_POSITIVES = 118
OUTER_SEED = 2026
CORE_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/"
    "main/src/v32/v32_github_only_plain.py?version=20260801_v32_exact1"
)
CORE_SHA256 = "591eb16a5c593be09fae17a0cea7f992aff99a3e1b49a5288d42464947790f4f"


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_native(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _mount_drive():
    try:
        from google.colab import drive
    except Exception:
        return
    if Path('/content/drive/MyDrive').exists():
        print('Google Drive already mounted.')
    else:
        print('Mounting Google Drive...')
        drive.mount('/content/drive')


def _find_exact_csv():
    roots = [
        Path('/content/drive/MyDrive/MAT-Appendix/optimization_v31_runs'),
        Path('/content/drive/MyDrive/MAT-Appendix/optimization_runs'),
        Path('/content/MAT-Appendix/optimization_v31_runs'),
        Path('/content/MAT-Appendix/optimization_runs'),
    ]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob('all_nested_oof_predictions.csv'))
    paths.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)

    rejected = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            if 'Truth' not in df.columns:
                rejected.append(f'{path}: missing Truth')
                continue
            y = pd.to_numeric(df['Truth'], errors='coerce')
            probs = [c for c in df.columns if c.endswith('_Probability')]
            if len(df) != EXPECTED_N:
                rejected.append(f'{path}: N={len(df)}')
                continue
            if y.isna().any() or set(y.astype(int).unique()) != {0, 1}:
                rejected.append(f'{path}: invalid Truth')
                continue
            if int(y.sum()) != EXPECTED_POSITIVES:
                rejected.append(f'{path}: positives={int(y.sum())}')
                continue
            if len(probs) < 6:
                rejected.append(f'{path}: probability columns={len(probs)}')
                continue
            print('Selected exact MAT-Appendix artifact:')
            print(path)
            return path, df
        except Exception as exc:
            rejected.append(f'{path}: {type(exc).__name__}: {exc}')

    raise RuntimeError(
        'No exact MAT-Appendix V3.1 artifact passed validation.\n'
        + '\n'.join(rejected[:20])
    )


def _clean_name(column):
    name = column[:-len('_Probability')].replace('_', ' ')
    return re.sub(r'\s+', ' ', name).strip()


def _build_oof(df):
    y = pd.to_numeric(df['Truth'], errors='raise').astype(int).to_numpy()
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        folds[test_idx] = fold

    out = pd.DataFrame({'y_true': y, 'outer_fold': folds})
    model_names = []
    for column in [c for c in df.columns if c.endswith('_Probability')]:
        name = _clean_name(column)
        values = pd.to_numeric(df[column], errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise RuntimeError(f'Invalid probability vector: {column}')
        out[name] = np.clip(values, 1e-6, 1 - 1e-6)
        model_names.append(name)

    print(f'Validated cohort: N={len(out)}, positives={int(out.y_true.sum())}, negatives={int((1-out.y_true).sum())}')
    print('Reconstructed outer folds:', sorted(out.outer_fold.unique().tolist()))
    print('Probability models:', model_names)
    return out


def main():
    _mount_drive()
    path, raw = _find_exact_csv()
    exact_oof = _build_oof(raw)

    core_source = urllib.request.urlopen(CORE_URL).read().decode('utf-8')
    actual_sha = hashlib.sha256(core_source.encode('utf-8')).hexdigest()
    if actual_sha != CORE_SHA256:
        raise RuntimeError(
            'V3.2 core checksum mismatch.\n'
            f'Expected: {CORE_SHA256}\nActual:   {actual_sha}'
        )

    original_dump = json.dump

    def safe_dump(obj, fp, *args, **kwargs):
        kwargs.setdefault('default', _native)
        return original_dump(obj, fp, *args, **kwargs)

    json.dump = safe_dump
    env = {
        '__name__': '__main__',
        'v31_oof_df': exact_oof,
        'json': json,
    }
    try:
        print('\nStarting exact MAT-Appendix V3.2.1 blend...\n')
        exec(compile(core_source, CORE_URL, 'exec'), env, env)
    finally:
        json.dump = original_dump

    globals()['v31_oof_exact'] = exact_oof
    for key in ('v32_result', 'result', 'aggregate', 'comparison', 'predictions'):
        if key in env:
            globals()[f'v32_1_{key}'] = env[key]

    print('\nV3.2.1 completed using only the validated 463-patient MAT-Appendix cohort.')
    print('Source artifact:', path)
    return env


v32_1_environment = main()
