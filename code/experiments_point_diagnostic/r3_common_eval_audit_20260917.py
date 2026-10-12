"""Independent CPU-only audit of archived R3 evaluation; no inference."""
import ast, copy, hashlib, json, math, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'r3_common_eval_job13269743'/'common_eval'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
runtime=read(OUT/'runtime.json'); completed=read(OUT/'completed.json')
for name,h in completed['hashes'].items(): assert sha(OUT/name)==h,(name,'output hash')
for remote,h in runtime['hashes'].items():
    local=ROOT/('r3_20260916_package/'+Path(remote).name if '/r3_20260916_package/' in remote else Path(remote).name)
    assert sha(local)==h,(local,'source hash')
assert runtime['job']=='13269743'
assert runtime['selected']['A']['selected_hash']=='fa6d83dba94e7142455d232b32a262723a51d73308d7dc6d8e1f432be53e78c7'
assert runtime['selected']['B']['selected_hash']=='bc1b644da15b7b564e803e3b74f292cc0f636384799be8063ebadf28da381286'
matches=read(ROOT/'r3_20260916_package/test.json')
ids={m['id'] for m in matches}; assert len(ids)==60
for split in ['train','val']: assert not ids & {m['id'] for m in read(ROOT/f'r3_20260916_package/{split}.json')}
truth={(m['id'],r['round']):(m,r) for m in matches for r in m['rounds']}; assert len(truth)==600
# Execute only the exact hash-verified pure renderer/parser functions, no simulator imports.
source=(ROOT/'r3_20260916_data.py').read_text()
tree=ast.parse(source); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prompt','parse')]
env={'json':json,'math':math,'re':re,'PARAMS':{'volatility':-2,'b_temp':-1,'bias':0}}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'verified_renderer','exec'),env)
parse,prompt=env['parse'],env['prompt']
def measure(rows):
    valid=[r for r in rows if r['valid']]
    return {'n':len(rows),'valid':len(valid),'penalized_mse':float(np.mean([(r['p']-r['q'])**2 if r['valid'] else 1 for r in rows])),
      'valid_mae':float(np.mean([abs(r['p']-r['q']) for r in valid])) if valid else None,
      'valid_mse':float(np.mean([(r['p']-r['q'])**2 for r in valid])) if valid else None,
      'valid_brier':float(np.mean([(r['p']-r['y'])**2 for r in valid])) if valid else None}
conditions={}; accounting={}; state={}; prompt_digests={}
reported=read(OUT/'metrics.json')
for name in ('initial_A','initial_B','trained_A','trained_B'):
    obj=read(OUT/(name+'.json')); rows=obj['rows']; arm=name[-1]
    assert len(rows)==600 and {(r['match_id'],r['round']) for r in rows}==set(truth)
    priors={m['id']:m['initial'] for m in matches}; stopped=set(); counts={'invalid_output':0,'prior_chain_stopped':0,'normalized_fence':0}; coords={}; ph=hashlib.sha256()
    for r in sorted(rows,key=lambda r:(r['round'],r['match_id'])):
        mid=r['match_id'];m,t=truth[mid,r['round']]
        assert (r['q'],r['y'],r['a_role'],r['level'])==(t['p'],t['y'],m['a_role'],m['level'])
        if r['raw'] is None:
            assert mid in stopped and r['missing_reason']=='prior_chain_stopped' and not r['valid']
            counts['prior_chain_stopped']+=1;continue
        assert mid not in stopped
        text=prompt(m,t,arm,priors[mid]); altered=copy.deepcopy(t)
        altered.update(state=[999]*len(t['state']),previous=[888]*len(t['previous']),p=.987654321,y=1-t['y'])
        assert prompt(m,altered,arm,priors[mid])==text,'target leakage'
        ph.update((mid+str(r['round'])+text).encode())
        parsed=parse(r['raw'],m,arm); assert (parsed is not None)==r['valid']
        if parsed:
            assert parsed['p_opponent_1']==r['p'] and r['missing_reason'] is None
            if parse(r['raw'],m,arm,allow_fence=False) is None:counts['normalized_fence']+=1
            if arm=='B':
                priors[mid]=parsed['state'];errors=np.abs(np.array(parsed['state'])-t['state'])
                assert abs(float(errors.mean())-r['state_coordinate_mae'])<1e-10
                for key,e in zip(m['keys'],errors):coords.setdefault(f"ToM{m['level']}/{key}",[]).append(float(e))
        else:
            assert r['p'] is None and r['missing_reason']=='invalid_output';counts['invalid_output']+=1
            if arm=='B':stopped.add(mid)
    metrics=measure(rows)
    assert abs(metrics['penalized_mse']-reported['conditions'][name]['penalized_probability_mse'])<1e-12
    assert metrics['valid']==reported['conditions'][name]['valid']
    conditions[name]=rows;accounting[name]={**counts,'complete_chains':60-len(stopped),'elapsed_seconds':obj['elapsed_seconds']}
    state[name]={k:{'n':len(v),'mae':float(np.mean(v))} for k,v in coords.items()}
    prompt_digests[name]=ph.hexdigest()
for name in ('constant','frequency'):
    rows=[]
    for (mid,roundn),(m,t) in truth.items():
        p=.5 if name=='constant' else (1+sum(b for a,b in t['history']))/(2+len(t['history']))
        rows.append({'match_id':mid,'round':roundn,'a_role':m['a_role'],'level':m['level'],'q':t['p'],'y':t['y'],'p':p,'valid':True})
    conditions[name]=rows
    assert abs(measure(rows)['penalized_mse']-reported['baselines'][name]['penalized_probability_mse'])<1e-12
groups=[[m['id'] for m in matches if m['a_role']==role and m['level']==level] for role in (0,1) for level in (0,1,2)]
assert all(len(g)==10 for g in groups)
by_match={name:{m['id']:measure([r for r in rows if r['match_id']==m['id']])['penalized_mse'] for m in matches} for name,rows in conditions.items()}
rng=np.random.default_rng(20260916); samples=[[mid for g in groups for mid in rng.choice(g,len(g))] for _ in range(5000)]
contrasts={}
for a,b in [('trained_B','trained_A'),('trained_A','initial_A'),('trained_A','constant'),('trained_A','frequency'),('trained_B','initial_B'),('trained_B','constant')]:
    delta={mid:by_match[a][mid]-by_match[b][mid] for mid in ids}
    boot=[np.mean([delta[mid] for mid in sample]) for sample in samples]
    contrasts[a+' minus '+b]={'mean':float(np.mean(list(delta.values()))),'match_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
assert np.allclose(contrasts['trained_B minus trained_A']['match_bootstrap_95'],reported['B_minus_A_penalized_mse']['match_bootstrap_95'])
result={'status':'passed','source_and_output_hashes':True,'all_keys_targets_parser_and_chain_accounting':True,
 'no_oracle_boundary':'Verified hash-bound rollout and renderer; reconstructed own-state prompts and target-mutation invariance. Actual serialized inference prompts were not archived, so reconstruction is code-level provenance, not captured-input attestation.',
 'conditions':{n:measure(rows) for n,rows in conditions.items()},'accounting':accounting,'contrasts':contrasts,
 'by_cell':{n:{f"{'seeker' if role==0 else 'hider'}_ToM{level}":measure([r for r in rows if r['a_role']==role and r['level']==level]) for role in (0,1) for level in (0,1,2)} for n,rows in conditions.items()},
 'by_round':{n:{str(k):measure([r for r in rows if r['round']==k]) for k in range(1,11)} for n,rows in conditions.items()},
 'state_coordinates':state,'reconstructed_prompt_stream_sha256':prompt_digests,
 'limitations':['One training seed and development distribution only','A and B have unequal output burden and compute','Full history permits bypass of own-state recurrence','State-coordinate units differ and errors are conditional-valid','No captured prompt streams or per-step adapter tensors; runtime hashes plus source audit','No live decision or unknown-opponent transfer test']}
dest=ROOT/'r3_common_eval_job13269743/audit_20260917.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ['status','conditions','accounting','contrasts','by_cell','by_round']},indent=2))
