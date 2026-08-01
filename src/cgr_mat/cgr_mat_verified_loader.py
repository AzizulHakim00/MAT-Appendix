from __future__ import annotations

import hashlib
import os
import urllib.request

REPOSITORY = "AzizulHakim00/MAT-Appendix"
PART_NAMES = [
    "cgr_mat_multimodal_pipeline.py.part01",
    "cgr_mat_multimodal_pipeline.py.part02",
    "cgr_mat_multimodal_pipeline.py.part03",
    "cgr_mat_multimodal_pipeline.py.part04",
    "cgr_mat_multimodal_pipeline.py.part05",
    "cgr_mat_multimodal_pipeline.py.part06",
]
PART_SHA256 = {
    "cgr_mat_multimodal_pipeline.py.part01": "7ecd6b0d816a9549d23def5f67e54286ef5865e4bc6a0b74bce020d9a2d2a713",
    "cgr_mat_multimodal_pipeline.py.part02": "042a022fe7652ac4fab86b50ed1aae301d88865c92464188a027d044a32b07b7",
    "cgr_mat_multimodal_pipeline.py.part03": "658a91d1713a304d29e2af914e6b4a8ca4549cf06bf887e6c20d84ca5a3ba41f",
    "cgr_mat_multimodal_pipeline.py.part04": "149f9d6b85bceb743c3dc2f00d68bc1971a683daf4ee9d275115c0e9fade1170",
    "cgr_mat_multimodal_pipeline.py.part05": "4f7cc2cf2f1fda7f830d9c3a3ee7d620f2d3f81abccb76346fef038eea4af0be",
    "cgr_mat_multimodal_pipeline.py.part06": "4cd732ef9067805c43aac9f44650317c8dee2fc19f8ce3945cb149a243860d5e",
}
COMBINED_SHA256 = "bf7a8b0c4fbb6020411bb73ae1769d2e89ea068685ac3a905546df904e32b864"

SOURCE_COMMIT = globals().get("SOURCE_COMMIT") or os.environ.get("CGR_MAT_SOURCE_COMMIT")
if not SOURCE_COMMIT:
    raise RuntimeError("SOURCE_COMMIT must be pinned by the notebook before loading the pipeline.")

print("Loading checksum-pinned CGR-MAT source parts...")
print("Source commit:", SOURCE_COMMIT)
chunks: list[str] = []
for index, name in enumerate(PART_NAMES, start=1):
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/"
        f"src/cgr_mat/parts/{name}"
    )
    raw = urllib.request.urlopen(url, timeout=120).read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = PART_SHA256[name]
    if actual != expected:
        raise RuntimeError(
            f"Source part integrity failure for {name}.\n"
            f"Expected: {expected}\nActual:   {actual}"
        )
    chunks.append(raw.decode("utf-8"))
    print(f"  ✓ part {index}/{len(PART_NAMES)} verified: {name}")

source = "".join(chunks)
actual_combined = hashlib.sha256(source.encode("utf-8")).hexdigest()
if actual_combined != COMBINED_SHA256:
    raise RuntimeError(
        "Combined CGR-MAT source integrity failure.\n"
        f"Expected: {COMBINED_SHA256}\nActual:   {actual_combined}"
    )

os.environ["CGR_MAT_SOURCE_COMMIT"] = SOURCE_COMMIT
os.environ["CGR_MAT_SOURCE_SHA256"] = COMBINED_SHA256
print("✓ Combined CGR-MAT source integrity verified.")
print("Launching the pipeline...")
exec(compile(source, f"cgr_mat_pipeline@{SOURCE_COMMIT}", "exec"), globals(), globals())
