"""Recalculate Figure 3c and offline RL round-to-match aggregation (no inference)."""
from pathlib import Path
import csv,gzip,hashlib,io,json
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
def load(source):
    items=[r for r in manifest['files'] if r.get('source_relative_path')==source]
    assert len(items)==1,source
    r=items[0]; packed=(ROOT/r['path']).read_bytes()
    assert hashlib.sha256(packed).hexdigest()==r['release_sha256']
    raw=gzip.decompress(packed)
    assert hashlib.sha256(raw).hexdigest()==r['release_uncompressed_sha256']
    return raw.decode('utf-8-sig')
plotted=list(csv.DictReader((ROOT/'data/plotted_values.csv').open(encoding='utf-8')))
panel=[r for r in plotted if r['figure']=='3' and r['panel']=='c']
# Labels are read from the frozen table, not recreated from memory.
assert len(panel)==5
for file,index in [('initial_A',0),('trained_A',1),('trained_B',2)]:
    data=json.loads(load('fixed_history/r3_common_eval_job13269743/common_eval/'+file+'.json'))
    rows=data['rows']; assert len(rows)==600
    value=sum((r['p']-r['q'])**2 if r['valid'] else 1.0 for r in rows)/len(rows)
    assert abs(value-float(panel[index]['value']))<1e-12
    print('Figure 3c:',panel[index].get('condition',panel[index].get('label',file)),value,'PASS')
constant=sum((.5-r['q'])**2 for r in rows)/len(rows)
assert abs(constant-float(panel[3]['value']))<1e-12
print('Figure 3c constant:',constant,'PASS')
base='rl/figures_grpo_k6_600_eval_ckpts_20260706/'
rounds=list(csv.DictReader(io.StringIO(load(base+'grpo_online_rows_light.csv'))))
summary=list(csv.DictReader(io.StringIO(load(base+'grpo_online_match_summary.csv'))))
key=lambda r:(r['eval_set'],r['checkpoint_step'],r['condition'],r['llm_role'],r['opponent_name'],r['match_id'],r['repeat_idx'])
scores=defaultdict(float);counts=defaultdict(int)
for r in rounds:scores[key(r)]+=float(r['llm_payoff']);counts[key(r)]+=1
assert len(scores)==len(summary)
for r in summary:
    assert counts[key(r)]==50
    assert abs(scores[key(r)]-float(r['final_payoff']))<1e-12
print('Offline RL:',len(rounds),'rounds,',len(summary),'match scores PASS')
cells=0
for record in manifest['files']:
    source=record.get('source_relative_path','')
    if not (source.startswith('standard_live/') and source.endswith('/audit.json')):continue
    audit=json.loads(load(source));assert audit['audit_passed']
    for cell in audit['cells']:
        data=list(csv.DictReader(io.StringIO(load(source.rsplit('/',1)[0]+'/'+cell['csv'].replace('\\','/')))))
        assert len(data)==cell['n_rows']
        matches=defaultdict(float)
        for row in data:matches[row['repeat_idx']]+=float(row['llm_payoff'])
        assert len(matches)==cell['n_matches']
        assert abs(sum(matches.values())/len(matches)-cell['mean_final_payoff'])<1e-12
        cells+=1
assert cells==53
print('Standard live-play:',cells,'audit-bound conditions PASS')
print('Checks archived arithmetic only, not model inference or simulator target generation.')
