from __future__ import annotations
import hashlib, importlib, sys, urllib.request
from pathlib import Path

SOURCE_REF=globals().get('V35_SOURCE_REF')
if not SOURCE_REF:
    raise RuntimeError('V35_SOURCE_REF must be set to a pinned GitHub commit.')
BASE=f'https://raw.githubusercontent.com/AzizulHakim00/MAT-Appendix/{SOURCE_REF}/src/v35'
WORK=Path('/content/mat_appendix_v35'); WORK.mkdir(parents=True,exist_ok=True)
print('Loading verified MAT-Appendix V3.5 modules...')
print('Source commit:',SOURCE_REF)
for name in ['core.py','models.py','runner.py']:
    data=urllib.request.urlopen(f'{BASE}/{name}').read()
    if len(data)<500:
        raise RuntimeError(f'V3.5 module is unexpectedly small: {name}')
    (WORK/name).write_bytes(data)
    print(f' {name}: sha256={hashlib.sha256(data).hexdigest()} bytes={len(data)}')
if str(WORK) not in sys.path: sys.path.insert(0,str(WORK))
for name in ['runner','models','core']:
    sys.modules.pop(name,None)
runner=importlib.import_module('runner')
runner.main()
