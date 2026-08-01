from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path("/content/MAT_Appendix_Multimodal_Data")
REG_DIR = ROOT / "regensburg"
EXT_DIR = ROOT / "duesseldorf_external"
REPORT_DIR = ROOT / "audit"
for p in (REG_DIR, EXT_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

DOWNLOAD_IMAGES = True
DOWNLOAD_EXTERNAL = True
FORCE_REDOWNLOAD = False

ZENODO_RECORD_ID = "7711412"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
FILES = {
    "app_data.xlsx": {
        "url": f"{ZENODO_BASE}/app_data.xlsx?download=1",
        "md5": "d17a803f5e27532e518676a38f588b59",
    },
    "README.md": {
        "url": f"{ZENODO_BASE}/README.md?download=1",
        "md5": "0de3b9ed4479d0b977beb2d2780cef72",
    },
    "test_set_codes.csv": {
        "url": f"{ZENODO_BASE}/test_set_codes.csv?download=1",
        "md5": "4154f6b2df56bfc8c90993532c5e0132",
    },
    "US_Pictures.zip": {
        "url": f"{ZENODO_BASE}/US_Pictures.zip?download=1",
        "md5": "38b81fd3dddaeed458132be2ec1732ed",
    },
}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_verified(name: str, spec: dict, dest_dir: Path) -> Path:
    path = dest_dir / name
    expected = spec["md5"]
    if path.exists() and not FORCE_REDOWNLOAD:
        actual = md5sum(path)
        if actual == expected:
            print(f"✓ Existing verified file: {name}")
            return path
        print(f"Checksum mismatch for existing {name}; re-downloading.")
        path.unlink()

    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with requests.get(spec["url"], stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {name}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    actual = md5sum(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {name}\nExpected: {expected}\nActual:   {actual}"
        )
    tmp.replace(path)
    print(f"✓ Downloaded and verified: {name}")
    return path


def main() -> None:
    print("=" * 72)
    print("OFFICIAL REGENSBURG MULTIMODAL DATASET DOWNLOAD")
    print("=" * 72)

    required = ["app_data.xlsx", "README.md", "test_set_codes.csv"]
    if DOWNLOAD_IMAGES:
        required.append("US_Pictures.zip")

    downloaded = {
        name: download_verified(name, FILES[name], REG_DIR) for name in required
    }

    if DOWNLOAD_IMAGES:
        archive = downloaded["US_Pictures.zip"]
        existing_images = [
            p
            for p in REG_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
        ]
        if not existing_images:
            print("Extracting ultrasound archive...")
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(REG_DIR)
            print("✓ Extraction complete")
        else:
            print(f"✓ Ultrasound images already extracted ({len(existing_images)} found)")

    readme_text = downloaded["README.md"].read_text(
        encoding="utf-8", errors="replace"
    )
    print("\nREADME preview:\n", readme_text[:1200])

    xlsx_path = downloaded["app_data.xlsx"]
    xls = pd.ExcelFile(xlsx_path)
    print("\nWorkbook sheets:", xls.sheet_names)

    frames = {}
    sheet_summary = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        frames[sheet] = df
        sheet_summary.append(
            {
                "sheet": sheet,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_fraction": float(df.isna().mean().mean()),
            }
        )
    sheet_summary_df = pd.DataFrame(sheet_summary)
    print("\nSheet audit:")
    print(sheet_summary_df.to_string(index=False))
    sheet_summary_df.to_csv(REPORT_DIR / "workbook_sheet_audit.csv", index=False)

    primary_sheet = max(frames, key=lambda s: len(frames[s]))
    patient_df = frames[primary_sheet].copy()
    print(f"\nPrimary audit sheet: {primary_sheet} | shape={patient_df.shape}")
    print("Columns:")
    print(patient_df.columns.tolist())

    image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_paths = [
        p
        for p in REG_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    image_rows = []
    corrupt = []
    for p in tqdm(image_paths, desc="Auditing images"):
        try:
            with Image.open(p) as im:
                width, height = im.size
                mode = im.mode
            image_rows.append(
                {
                    "relative_path": str(p.relative_to(REG_DIR)),
                    "filename": p.name,
                    "stem": p.stem,
                    "extension": p.suffix.lower(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": p.stat().st_size,
                }
            )
        except Exception as exc:
            corrupt.append({"path": str(p), "error": repr(exc)})

    image_manifest = pd.DataFrame(image_rows)
    if not image_manifest.empty:
        image_manifest.to_csv(
            REPORT_DIR / "regensburg_image_manifest.csv", index=False
        )
    pd.DataFrame(corrupt).to_csv(REPORT_DIR / "corrupt_images.csv", index=False)
    print(f"\nImages discovered: {len(image_manifest)}")
    print(f"Unreadable/corrupt images: {len(corrupt)}")

    candidate_id_columns = [
        c
        for c in patient_df.columns
        if any(
            token in str(c).lower()
            for token in ("id", "code", "patient", "number", "us_")
        )
    ]
    print("Candidate ID columns:", candidate_id_columns)

    external_repo = "https://github.com/i6092467/pediatric-appendicitis-ml-ext.git"
    if DOWNLOAD_EXTERNAL:
        if not (EXT_DIR / ".git").exists():
            if any(EXT_DIR.iterdir()):
                shutil.rmtree(EXT_DIR)
                EXT_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", external_repo, str(EXT_DIR)],
                check=True,
            )
        else:
            print("✓ External-validation repository already cloned")

    external_csvs = sorted(EXT_DIR.rglob("*.csv")) if DOWNLOAD_EXTERNAL else []
    print("\nExternal CSV files:")
    external_shapes = []
    for p in external_csvs:
        try:
            df = pd.read_csv(p)
            shape = list(df.shape)
            external_shapes.append(
                {"path": str(p.relative_to(EXT_DIR)), "rows": shape[0], "columns": shape[1]}
            )
            print(f"  {p.relative_to(EXT_DIR)} -> {tuple(shape)}")
        except Exception as exc:
            external_shapes.append(
                {"path": str(p.relative_to(EXT_DIR)), "error": repr(exc)}
            )
            print(f"  {p.relative_to(EXT_DIR)} -> read error: {exc}")

    source_audit = pd.DataFrame(
        [
            {
                "dataset": "Regensburg Pediatric Appendicitis",
                "role": "Primary multimodal development dataset",
                "official_source": "Zenodo 7711412 / UCI 938",
                "repository_release": "2023-02-23",
                "repository_modified": "2024-02-01",
                "clinical_period": "Reported as 2016-2021",
                "patients_or_instances": 782,
                "raw_ultrasound_images": True,
                "tabular_data": True,
                "automatic_download": True,
                "authentication_required": False,
                "notes": "1-15 B-mode views per subject; severity subset is smaller.",
            },
            {
                "dataset": "Düsseldorf external-validation cohort",
                "role": "External tabular/site-shift validation",
                "official_source": "Authors' GitHub + Frontiers in Pediatrics 2025",
                "repository_release": "2024-2025",
                "repository_modified": "live GitHub repository",
                "clinical_period": "2015-01-01 to 2022-02-01",
                "patients_or_instances": 301,
                "raw_ultrasound_images": False,
                "tabular_data": True,
                "automatic_download": True,
                "authentication_required": False,
                "notes": "US parameters are present, but no raw image archive.",
            },
        ]
    )
    source_audit.to_csv(REPORT_DIR / "dataset_source_audit.csv", index=False)
    print("\nSource audit:")
    print(source_audit.to_string(index=False))

    audit = {
        "root": str(ROOT),
        "regensburg_record_id": ZENODO_RECORD_ID,
        "verified_files": {
            name: {
                "path": str(path),
                "md5": md5sum(path),
                "expected_md5": FILES[name]["md5"],
            }
            for name, path in downloaded.items()
        },
        "workbook_sheets": sheet_summary,
        "primary_audit_sheet": primary_sheet,
        "primary_shape": list(patient_df.shape),
        "candidate_id_columns": [str(c) for c in candidate_id_columns],
        "image_count": int(len(image_manifest)),
        "corrupt_image_count": int(len(corrupt)),
        "external_csv_count": int(len(external_csvs)),
        "external_csv_shapes": external_shapes,
        "date_warning": (
            "The repository release is recent, but patient acquisition is older. "
            "Do not describe it as a 2023-2026 clinical cohort."
        ),
        "kaggle_warning": (
            "Kaggle mirrors exist, but a Kaggle usability score is platform metadata, "
            "not scientific quality evidence. Mirrors also show inconsistent license "
            "labels. This pipeline uses the official checksum-verified Zenodo source."
        ),
    }
    (REPORT_DIR / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 72)
    print("Primary multimodal root:", REG_DIR)
    print("External validation root:", EXT_DIR)
    print("Audit reports:", REPORT_DIR)
    print("No manual upload was required.")
    print("\nSCIENTIFIC DECISION:")
    print("- Use Regensburg for raw-image + tabular multimodal development.")
    print("- Use Düsseldorf only for external tabular/missing-image validation.")
    print("- Do not claim the patient records were collected within the last 3 years.")


if __name__ == "__main__":
    main()
