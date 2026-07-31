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
exec(compile(source, 'MAT_Appendix_Optimization_Primary_FullSource.py', 'exec'))
