"""Checksum-pinned loader for MAT-Appendix V3.2 Locked Revalidation."""
from __future__ import annotations

import base64
import hashlib
import importlib
import os
import sys
import tempfile
import urllib.request
import zlib
from pathlib import Path

REPOSITORY = "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix"
SOURCE_COMMIT = "d56313c50767d4ed4067d4fefa909f9aaaba5ec7"
FILES = {
    "core.py": (
        f"{REPOSITORY}/{SOURCE_COMMIT}/src/v35/core.py",
        "1de44294ecbf755e81485f86013a3dd2c2cac7320a25204d27a3e481c4909e83",
    ),
    "models.py": (
        f"{REPOSITORY}/{SOURCE_COMMIT}/src/v35/models.py",
        "a0798f161c3d5f4708ecd0428838f9cf39f0c2201736e3fcb50678cf11767bb5",
    ),
}
PAYLOAD_URL = f"{REPOSITORY}/{SOURCE_COMMIT}/src/v32_locked/payload.txt"
EXPECTED_PAYLOAD_SHA256 = "86d52e1d97a9f1f1e03f846f615eb73acb9e7ce7235f55f57225f725ff09107a"
EXPECTED_RUNNER_SHA256 = "b05f5e45e1398d05f34bf3fb012c3caf33ddca1b931471a600f983e685d3ded5"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MAT-Appendix-v32-locked"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _verify(name: str, content: bytes, expected: str) -> None:
    actual = hashlib.sha256(content).hexdigest()
    print(f"  {name}: sha256={actual} bytes={len(content)}")
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {name}. Expected {expected}, got {actual}."
        )


def load_and_run():
    print("Loading MAT-Appendix V3.2 Locked Revalidation...")
    print("Pinned source commit:", SOURCE_COMMIT)
    work = Path(tempfile.mkdtemp(prefix="mat_v32_locked_"))

    for filename, (url, expected_sha) in FILES.items():
        content = _download(url)
        _verify(filename, content, expected_sha)
        (work / filename).write_bytes(content)

    payload_bytes = _download(PAYLOAD_URL).strip()
    _verify("payload.txt", payload_bytes, EXPECTED_PAYLOAD_SHA256)
    try:
        runner_bytes = zlib.decompress(base64.b64decode(payload_bytes))
    except Exception as exc:
        raise RuntimeError(f"Could not decode locked runner payload: {exc}") from exc
    _verify("runner.py", runner_bytes, EXPECTED_RUNNER_SHA256)
    compile(runner_bytes.decode("utf-8"), "runner.py", "exec")
    (work / "runner.py").write_bytes(runner_bytes)

    sys.path.insert(0, str(work))
    for module_name in ("runner", "models", "core"):
        sys.modules.pop(module_name, None)
    runner = importlib.import_module("runner")
    print("Source integrity and syntax verified.")
    return runner.main()


V32_LOCKED_OUTPUT_DIR = load_and_run()
