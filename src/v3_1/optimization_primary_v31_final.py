# MAT-Appendix Optimization V3.1 final integrity wrapper
import ast
import base64
import hashlib
import json
import re
import urllib.request
import zlib

_REPO = "https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/main"
_PATCH_LOADER_URL = _REPO + "/src/v3_1/optimization_primary_v31.py?v=20260801-integrity"
_BASE = _REPO + "/src/v3"
_BASE_PARTS = [f"{_BASE}/payload_{i:02d}.txt?v=20260801-integrity" for i in range(1, 5)]
_ORIGINAL_SHA = "d0cba186ae89a29452d8b80c780f5f7cb7d3218d672fe1b203c24f087593b12d"
_TYPED_SHA = "83c2b3a2b5f6a21857d6c3cfbfa4a64b3550175e7d197c22e3d975b39df653ac"
_FINAL_SHA = "3f93c91dbc8b27316b74ec85bb40998fed6d3997bfc8509e421ccf37b44ee53f"

patch_loader = urllib.request.urlopen(_PATCH_LOADER_URL).read().decode("utf-8")
match = re.search(r'_PATCH_B64\s*=\s*("(?:[^"\\]|\\.)*")', patch_loader)
if not match:
    raise RuntimeError("Could not extract V3.1 patch payload from repository loader.")
patch_b64 = ast.literal_eval(match.group(1))

encoded = "".join(urllib.request.urlopen(url).read().decode("utf-8").strip() for url in _BASE_PARTS)
source = zlib.decompress(base64.b64decode(encoded)).decode("utf-8")
if hashlib.sha256(source.encode()).hexdigest() != _ORIGINAL_SHA:
    raise RuntimeError("Original optimization source checksum mismatch.")

anchor = "def sample_candidates(name, rng, class_ratio):"
helper = '''def _typed_choice(rng, options):
    """Return an original Python object without NumPy mixed-dtype coercion."""
    return options[int(rng.integers(0, len(options)))]


'''
source = source.replace(anchor, helper + anchor, 1)
replacements = {
    "rng.choice([500,800,1200])": "_typed_choice(rng, [500,800,1200])",
    "rng.choice([None,4,6,8,10])": "_typed_choice(rng, [None,4,6,8,10])",
    "rng.choice([1,2,4,6,8])": "_typed_choice(rng, [1,2,4,6,8])",
    "rng.choice(['sqrt',0.35,0.55,0.75])": "_typed_choice(rng, ['sqrt',0.35,0.55,0.75])",
    "rng.choice([None,5,8,12])": "_typed_choice(rng, [None,5,8,12])",
    "rng.choice(['sqrt',0.4,0.6,0.8])": "_typed_choice(rng, ['sqrt',0.4,0.6,0.8])",
    "class_weight=rng.choice(['balanced','balanced_subsample'])": "class_weight=_typed_choice(rng, ['balanced','balanced_subsample'])",
}
source = source.replace(
    "min_samples_leaf=int(rng.choice([1,2,4,6]))",
    "min_samples_leaf=int(_typed_choice(rng, [1,2,4,6]))",
)
for old, new in replacements.items():
    source = source.replace(old, new)
if hashlib.sha256(source.encode()).hexdigest() != _TYPED_SHA:
    raise RuntimeError("Type-safe base source checksum mismatch.")

old_lines = source.splitlines(keepends=True)
ops = json.loads(zlib.decompress(base64.b64decode(patch_b64)).decode("utf-8"))
new_lines, cursor = [], 0
for i1, i2, replacement in ops:
    new_lines.extend(old_lines[cursor:i1])
    new_lines.extend(replacement)
    cursor = i2
new_lines.extend(old_lines[cursor:])
source = "".join(new_lines)
if hashlib.sha256(source.encode()).hexdigest() != _FINAL_SHA:
    raise RuntimeError("Final V3.1 source checksum mismatch.")

print("Loaded verified MAT-Appendix Optimization V3.1 final source.")
exec(compile(source, "MAT_Appendix_Optimization_V3_1_Final.py", "exec"))
