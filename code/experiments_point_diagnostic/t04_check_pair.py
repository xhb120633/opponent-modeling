"""Read-only check that prospective arms differ only in initialization/output path."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
a=json.loads((root/'base/preflight.json').read_text())
b=json.loads((root/'parent/preflight.json').read_text())
assert a['arm']=='base' and b['arm']=='parent'
assert a['init_adapter_path'] is None and b['init_adapter_path']
for key in ('arm','init_adapter_path','job_id'):
    a.pop(key);b.pop(key)
a['settings'].pop('output_dir');b['settings'].pop('output_dir')
for item in (a,b):
    item['hashes']={str(Path(path).resolve()):value for path,value in item['hashes'].items()}
assert a==b,'Non-initialization config difference'
print('MATCHED_PAIR_PREFLIGHT_PASSED')
