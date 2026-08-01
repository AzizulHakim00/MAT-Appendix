"""Verified loader for MAT-Appendix V3.4 0.93-AUROC challenge.

The full audited source is stored as a compressed text payload to keep the
Colab notebook small. The SHA-256 check is performed before execution.
"""

import base64
import hashlib
import urllib.request
import zlib

_REPOSITORY = "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix"
_SOURCE_REF = globals().get("V34_SOURCE_REF", "v34-093-challenge")
_PAYLOAD_URL = f"{_REPOSITORY}/{_SOURCE_REF}/src/v34/payload_01.txt"
_EXPECTED_SOURCE_SHA256 = "caa82a054f412b3fa0c37b4ccc53eb4accdc37423329eb86431a20cec0652d35"

print("Loading verified MAT-Appendix V3.4 challenge source...")
print("Source ref:", _SOURCE_REF)

encoded = urllib.request.urlopen(_PAYLOAD_URL).read().decode("utf-8").strip()
source = zlib.decompress(base64.b64decode(encoded)).decode("utf-8")
actual_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

if actual_sha256 != _EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        "V3.4 source checksum mismatch.\n"
        f"Expected: {_EXPECTED_SOURCE_SHA256}\n"
        f"Actual:   {actual_sha256}"
    )

print("V3.4 source integrity verified.")
exec(compile(source, "MAT_Appendix_V3_4_093_Challenge.py", "exec"), globals(), globals())
