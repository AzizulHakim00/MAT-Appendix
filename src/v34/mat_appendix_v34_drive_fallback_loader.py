"""MAT-Appendix V3.4 loader with a narrow Google Drive fallback patch.

The verified V3.4 source is loaded from pinned commit
86171faaaa47e3df939754420a3b71bcfe233073. Its SHA-256 is checked before a
single operational patch is applied: a failing google.colab.drive.mount call
is caught so the source can continue using its /content fallback.
"""

import base64
import hashlib
import re
import urllib.request
import zlib

SOURCE_REF = "86171faaaa47e3df939754420a3b71bcfe233073"
REPOSITORY = "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix"
PAYLOAD_URL = f"{REPOSITORY}/{SOURCE_REF}/src/v34/payload_01.txt"
EXPECTED_ORIGINAL_SHA256 = "caa82a054f412b3fa0c37b4ccc53eb4accdc37423329eb86431a20cec0652d35"

print("Loading verified MAT-Appendix V3.4 source with Drive fallback...")
print("Source ref:", SOURCE_REF)

encoded = urllib.request.urlopen(PAYLOAD_URL).read().decode("utf-8").strip()
source = zlib.decompress(base64.b64decode(encoded)).decode("utf-8")
actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
if actual != EXPECTED_ORIGINAL_SHA256:
    raise RuntimeError(
        "Original V3.4 source checksum mismatch.\n"
        f"Expected: {EXPECTED_ORIGINAL_SHA256}\nActual:   {actual}"
    )
print("Original V3.4 source integrity verified.")

# Replace exactly one direct drive.mount(...) statement while preserving its
# indentation. No model, data, validation, target, or metric logic is changed.
pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)drive\.mount\((?P<args>[^\n]*)\)\s*$")


def replacement(match):
    indent = match.group("indent")
    args = match.group("args")
    return (
        f"{indent}try:\n"
        f"{indent}    drive.mount({args})\n"
        f"{indent}except Exception as drive_exc:\n"
        f"{indent}    print('Google Drive mount failed; continuing with /content fallback:', drive_exc)"
    )

patched_source, count = pattern.subn(replacement, source, count=1)
if count != 1:
    raise RuntimeError(
        f"Drive fallback patch expected exactly one drive.mount call; found {count}."
    )

print("Drive fallback patch verified: 1 operational call patched.")
exec(
    compile(patched_source, "MAT_Appendix_V3_4_093_Challenge_DriveFallback.py", "exec"),
    globals(),
    globals(),
)
