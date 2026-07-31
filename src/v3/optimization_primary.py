# Bootstrap loader for the validated MAT-Appendix nested optimization source.
# The full source is zlib-compressed and split into four text payloads to keep
# GitHub/Colab loading reliable. The decoded source SHA-256 is verified before execution.
import base64
import hashlib
import urllib.request
import zlib

_BASE = 'https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/main/src/v3'
_PARTS = [f'{_BASE}/payload_{index:02d}.txt' for index in range(1, 5)]
_EXPECTED_SHA256 = 'd0cba186ae89a29452d8b80c780f5f7cb7d3218d672fe1b203c24f087593b12d'

encoded = ''.join(
    urllib.request.urlopen(url).read().decode('utf-8').strip()
    for url in _PARTS
)
source = zlib.decompress(base64.b64decode(encoded)).decode('utf-8')
actual_sha256 = hashlib.sha256(source.encode('utf-8')).hexdigest()
if actual_sha256 != _EXPECTED_SHA256:
    raise RuntimeError(
        f'Optimization source checksum mismatch: expected {_EXPECTED_SHA256}, got {actual_sha256}'
    )

# Runtime hotfix (2026-08-01): NumPy coerces mixed string/float choices to strings.
# This made RandomForest/ExtraTrees receive np.str_('0.55') instead of float 0.55.
def _apply_parameter_type_hotfix(text):
    anchor = 'def sample_candidates(name, rng, class_ratio):'
    helper = '''def _typed_choice(rng, options):
    """Return an original Python object without NumPy mixed-dtype coercion."""
    return options[int(rng.integers(0, len(options)))]


'''
    if 'def _typed_choice(rng, options):' not in text:
        if text.count(anchor) != 1:
            raise RuntimeError('Optimization hotfix failed: sample_candidates anchor mismatch.')
        text = text.replace(anchor, helper + anchor, 1)

    replacements = {
        "rng.choice([None,4,6,8,10])": "_typed_choice(rng, [None,4,6,8,10])",
        "rng.choice(['sqrt',0.35,0.55,0.75])": "_typed_choice(rng, ['sqrt',0.35,0.55,0.75])",
        "rng.choice([None,5,8,12])": "_typed_choice(rng, [None,5,8,12])",
        "rng.choice(['sqrt',0.4,0.6,0.8])": "_typed_choice(rng, ['sqrt',0.4,0.6,0.8])",
        "class_weight=rng.choice(['balanced','balanced_subsample'])": "class_weight=_typed_choice(rng, ['balanced','balanced_subsample'])",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f'Optimization hotfix target missing: {old}')
        text = text.replace(old, new)
    return text

source = _apply_parameter_type_hotfix(source)
compile(source, 'MAT_Appendix_Optimization_Primary_FullSource.py', 'exec')
print('Applied RandomForest/ExtraTrees parameter-type hotfix.')
exec(compile(source, 'MAT_Appendix_Optimization_Primary_FullSource.py', 'exec'))
