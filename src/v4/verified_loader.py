from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "AzizulHakim00/MAT-Appendix"
SOURCE_COMMIT = "b115ac27bf953e25e900defd68c03e608a15b06b"
SOURCE_FILES = {
    "core.py": "02a83838b864a5471de26094eb58fb07fde46b395813757e69a371ef1f20b3b3",
    "features.py": "28ac63ba53752cf63152d841e1af3d24b9bb33468be0ef6d59a9f85aec19e401",
    "experts.py": "b25dbb678baeccfa3629c73ac9c66aff9ef5d86422bc705268f62c3026e73d68",
    "stacking.py": "ee3a60d689a5a657af8d79b175bd62738a3f547a59202a7323bd7dbf28b4fe74",
    "evaluation.py": "7a436da6a5580bf20a345efbdc785b9555b86d8c18d81ca8581af1d94c4d923d",
    "runner.py": "f5fb89331e5ba086ffd9017f928e999c88dc6f91265a4918af26e8e773af0f88",
}

source_root = Path("/content/cemat_stack_v4_source")
source_root.mkdir(parents=True, exist_ok=True)
combined = hashlib.sha256()
print("Loading checksum-pinned CEMAT-Stack V4 source...")
print("Source commit:", SOURCE_COMMIT)

for filename, expected_sha256 in SOURCE_FILES.items():
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/"
        f"src/v4/{filename}"
    )
    raw = urllib.request.urlopen(url, timeout=120).read()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Integrity failure for {filename}: expected {expected_sha256}, got {actual_sha256}"
        )
    compile(raw.decode("utf-8"), f"cemat_v4@{SOURCE_COMMIT}/{filename}", "exec")
    (source_root / filename).write_bytes(raw)
    combined.update(filename.encode("utf-8"))
    combined.update(raw)
    print(f"  ✓ {filename} verified")

combined_sha256 = combined.hexdigest()
os.environ["CEMAT_SOURCE_COMMIT"] = SOURCE_COMMIT
os.environ["CEMAT_SOURCE_SHA256"] = combined_sha256

for module_name in ("runner", "evaluation", "stacking", "experts", "features", "core"):
    sys.modules.pop(module_name, None)
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

print("Combined source SHA256:", combined_sha256)
print("Launching CEMAT-Stack V4...")
from runner import main

main()
