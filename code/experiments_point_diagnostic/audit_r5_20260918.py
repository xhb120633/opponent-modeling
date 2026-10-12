"""Independent post-run R5 audit. No model calls or training; local frozen inputs only."""
import copy, hashlib, json, math, re
from pathlib import Path
import numpy as np
import r3_20260916_data as d
from r4_transfer_20260917 import unknown

ROOT=Path(__file__).resolve().parent
P=ROOT/'r5_package_20260918'
OUT=ROOT/'r5_results_20260918'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def close(a,b): assert abs(a-b)<1e-12,(a,b)
def parse(s):
    try:
        s=s.strip()
        if s.startswith('```'):
            m=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',s,re.S)
            if m:s=m[1]
        x=json.loads(s)
        if not isinstance(x,dict) or set(x)!={'p_opponent_1'}: return None
        v=x['p_opponent_1']
        return v if type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 else None
    except (ValueError,TypeError): return None
def key(r): return (r['information'],r['match_id'],r['round'])
def stats(rr):
    vv=[r for r in rr if r['valid']]
    q=np.array([r['q'] for r in vv]); p=np.array([r['p'] for r in vv])
    def corr(x,y): return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
    return dict(n=len(rr),valid=len(vv),mse=float(np.mean([(r['p']-r['q'])**2 if r['valid'] else 1 for r in rr])),
        valid_mse=float(np.mean((p-q)**2)) if vv else None,mae=float(np.mean(abs(p-q))) if vv else None,r=corr(p,q),
        constant=float(np.mean([(.5-r['q'])**2 for r in rr])),constant_mae=float(np.mean([abs(.5-r['q']) for r in rr])),
        frequency=float(np.mean([(r['frequency']-r['q'])**2 for r in rr])),frequency_mae=float(np.mean([abs(r['frequency']-r['q']) for r in rr])),
        frequency_r=corr(np.array([r['frequency'] for r in rr]),np.array([r['q'] for r in rr])))
def grouped(rows):
    out={}
    for info in sorted({r['information'] for r in rows}):
        rr=[r for r in rows if r['information']==info and r['round']>1]
        groups={'all':rr}
        for role in (0,1):
            groups[f'A{role}']=[r for r in rr if r['role']==role]
            for level in range(3):groups[f'A{role}_ToM{level}']=[r for r in rr if r['role']==role and r['level']==level]
        for level in range(3):groups[f'ToM{level}']=[r for r in rr if r['level']==level]
        for name,lo,hi in [('early2_5',2,5),('late6_10',6,10),('early2_10',2,10),('late11_30',11,30)]:
            sub=[r for r in rr if lo<=r['round']<=hi]
            if sub:groups[name]=sub
        out[info]={k:stats(v) for k,v in groups.items()}
    return out

manifest=read(P/'manifest.json')
assert sha(P/'manifest.json')=='b375a9e93871a88c00596554dc48f1a36db70ea6f67ce6d43332ce7973748ed6'
for n,h in manifest['files'].items():assert sha(P/n)==h
for n,h in manifest['sources'].items():assert sha(ROOT/n)==h
assert sha(Path(d.kt.__file__))==manifest['simulator_sha256']
data={s:read(P/f'{s}.json') for s in ('train','val','test','fresh','long')}
seen_ids=set(); seen_seeds=set();replayed=0;max_target_delta=0.
for split,mm in data.items():
    for m in mm:
        assert m['id'] not in seen_ids and m['seed'] not in seen_seeds
        seen_ids.add(m['id']);seen_seeds.add(m['seed'])
        state=d.kt.init_k_tom(d.PARAMS,m['level'])
        for i,r in enumerate(m['rounds']):
            assert r['round']==i+1 and len(r['history'])==i
            if i:
                a,b=r['history'][-1]
                assert b==m['rounds'][i-1]['y']
                assert r['history'][:-1]==m['rounds'][i-1]['history']
                state=d.kt.learning_function(state,d.PARAMS,b,a,m['level'],m['b_role'],d.MATRIX)
            logit,_=d.kt.decision_function(copy.deepcopy(state),d.PARAMS,m['b_role'],m['level'],d.MATRIX)
            delta=abs(float(d.kt.inv_logit(logit))-r['p']);max_target_delta=max(max_target_delta,delta)
            assert delta<1e-8,(split,m['id'],r['round'],delta)
            replayed+=1
a,b=[read(P/f'train_{x}.json') for x in ('known','mixed')]
assert len(a)==len(b)==2400
for i,(x,y) in enumerate(zip(a,b)):
    assert (x['match_id'],x['round'],x['target'])==(y['match_id'],y['round'],y['target'])
    m=data['train'][i//10]; r=m['rounds'][i%10]
    assert x['information']=='known' and x['prompt']==d.prompt(m,r,'A')
    expected='unknown' if (i//10+i%10)%2 else 'known'
    assert y['information']==expected
    assert y['prompt']==(unknown(m,r) if expected=='unknown' else d.prompt(m,r,'A'))
    close(json.loads(x['target'])['p_opponent_1'],float(f"{r['p']:.9g}"))

invalid=[];audited=0
def audit_file(path,split,infos):
    global audited
    rr=lines(path);expected={(info,m['id'],r['round']):(m,r) for info in infos for m in data[split] for r in m['rounds']}
    assert len(rr)==len(expected) and len({key(r) for r in rr})==len(rr) and {key(r) for r in rr}==set(expected)
    for z in rr:
        m,r=expected[key(z)];info=z['information']
        assert z['role']==m['a_role'] and z['level']==m['level']
        close(z['q'],r['p']);close(z['frequency'],(1+sum(b for a,b in r['history']))/(2+len(r['history'])))
        renderer=unknown if info=='unknown' else lambda m,r:d.prompt(m,r,'A')
        assert z['prompt']==renderer(m,r)
        mutate={**r,'p':-99,'y':-99,'state':[999],'previous':[999]}
        assert renderer(m,mutate)==z['prompt']
        if info=='unknown':assert renderer({**m,'level':999,'keys':[],'initial':[999],'seed':-1,'a_policy':{}},mutate)==z['prompt']
        p=parse(z['raw']);assert z['valid']==(p is not None) and z['p']==p
        if p is None:invalid.append(dict(file=str(path.relative_to(OUT)),key=key(z),raw=z['raw']))
    audited+=len(rr);return rr

hashes={'known':'ba4b3296d4151b624b975a4d4bfed5fe5517b2d12deb06a3cbe629d0e519c3a7','mixed':'8cef54c70dd054fcbf3175f47015ffbd509875755ecae49bc278be26e2cac2a2'}
results={};allrows={};init=[];selections={}
for arm,job in [('known','13312536'),('mixed','13312537')]:
    folder=OUT/arm;done=read(folder/'completed.json');train=read(folder/'training_completed.json');runtime=read(folder/'runtime.json')
    for n,h in done['files'].items():assert sha(folder/n)==h,(arm,n)
    assert done['job']==runtime['job']==job and done['arm']==runtime['arm']==arm
    assert train['steps']==runtime['step_cap']==200 and train['examples']==2400
    assert train['selected_hash']==done['selected_hash']==hashes[arm]
    assert train['start_adapter_sha256']==runtime['adapter_sha256']==manifest['original_adapter_sha256']
    assert runtime['package_sha256']==train['package_sha256']==sha(P/'manifest.json')
    assert runtime['driver_sha256']==train['driver_sha256']==sha(ROOT/'r5_metadata_20260918.py')
    initialization=read(folder/'initialization_audit.json')
    assert initialization['passed'] and initialization['exact_tensor_match'] and initialization['tensors']==392
    assert initialization['adapter_sha256']==manifest['original_adapter_sha256']
    err=(OUT/f'r5_{arm}_{job}.err').read_text(encoding='utf-8')
    assert 'Ran 2 tests' in err and 'Ran 4 tests' in err and err.count('\nOK')==2 and 'Traceback' not in err
    best=float('inf');selected=None;values=[]
    assert [s['step'] for s in train['scores']]==[0,50,100,150,200]
    for s in train['scores']:
        rr=audit_file(folder/f"validation_step{s['step']}.jsonl",'val',('known','unknown'))
        val={i:stats([r for r in rr if r['round']>1 and r['information']==i]) for i in ('known','unknown')}
        score=(val['known']['mse']+val['unknown']['mse'])/2;close(score,s['score'])
        for info in val:
            for k,v in s['by_information'][info].items():close(v,val[info][k])
        improved=score<best-1e-4;assert improved==s['improved']
        if improved:best=score;selected=s['step']
        values.append(dict(step=s['step'],score=score,selected_at_step=improved))
        if s['step']==0:init.append([(key(r),r['raw'],r['p']) for r in rr])
    assert train['selected_path'].endswith(f'/selected_step{selected}')
    selections[arm]=dict(selected_step=selected,validation=values,selected_hash=hashes[arm])
    results[arm]={};allrows[arm]={}
    logged=read(folder/'metrics.json')
    for split in ('test','fresh','long'):
        rr=audit_file(folder/f'eval_{split}.jsonl',split,('known',) if split=='long' else ('known','unknown'))
        allrows[arm][split]=rr;results[arm][split]=grouped(rr)
        for info,v in logged[split].items():
            for k,x in v.items():close(x,results[arm][split][info]['all'][k])
assert init[0]==init[1]
original=audit_file(OUT/'known/initial_A_fresh.jsonl','fresh',('known','unknown'))
results['original_A500']={'fresh':grouped(original)}
for info,v in read(OUT/'known/metrics.json')['original_A500_fresh'].items():
    for k,x in v.items():close(x,results['original_A500']['fresh'][info]['all'][k])

def paired(left,right,info='unknown',baseline=None,role=None,level=None):
    l={key(r):r for r in left if r['round']>1 and r['information']==info and (role is None or r['role']==role) and (level is None or r['level']==level)}
    r={key(z):z for z in right};groups={}
    def err(z):return (z['p']-z['q'])**2 if z['valid'] else 1
    for k,z in l.items():
        v=err(r[k]) if baseline is None else ((.5 if baseline=='constant' else z['frequency'])-z['q'])**2
        groups.setdefault((z['role'],z['level']),{}).setdefault(z['match_id'],[]).append(err(z)-v)
    cells=[np.array([np.mean(v) for v in g.values()]) for g in groups.values()]
    rng=np.random.default_rng(20260918);boot=np.zeros(5000);total=sum(len(c) for c in cells)
    for c in cells:boot+=rng.choice(c,size=(5000,len(c)),replace=True).sum(axis=1)/total
    return dict(delta_mse=float(sum(c.sum() for c in cells)/total),ci95=np.quantile(boot,[.025,.975]).tolist(),matches=total,replicates=5000,stratification='A role x ToM level',unit='whole match')

comparisons={}
for split in ('fresh','test','long'):
    for info in ('known',) if split=='long' else ('known','unknown'):
        comparisons[f'{split}_{info}_mixed_minus_known']=paired(allrows['mixed'][split],allrows['known'][split],info)
for arm in ('known','mixed'):
    comparisons[f'fresh_unknown_{arm}_minus_A500']=paired(allrows[arm]['fresh'],original)
    comparisons[f'fresh_unknown_{arm}_minus_constant']=paired(allrows[arm]['fresh'],[],baseline='constant')
    for role in (0,1):
        for level in range(3):
            comparisons[f'fresh_unknown_A{role}_ToM{level}_{arm}_minus_constant']=paired(allrows[arm]['fresh'],[],baseline='constant',role=role,level=level)
            if arm=='mixed':comparisons[f'fresh_unknown_A{role}_ToM{level}_mixed_minus_known']=paired(allrows['mixed']['fresh'],allrows['known']['fresh'],role=role,level=level)

report=dict(status='independently_audited',audited_predictions=audited,selected_predictions=5520,original_baseline_predictions=1200,validation_predictions=12000,
    simulator_targets_replayed=replayed,simulator_replay_tolerance=1e-8,max_target_delta=max_target_delta,initial_validation_identical=True,invalid_outputs=invalid,selections=selections,metrics=results,comparisons=comparisons,
    audit_source_sha256=sha(Path(__file__)),package_sha256=sha(P/'manifest.json'),
    limitations=['One training seed; bootstrap quantifies match uncertainty only.','Fixed replay with known opponent families, not live policy or new algorithms.','Remote adapter hashes independently queried; initialization equality attested by gated runtime tensor check, not reconstructed training.','Base HF revision/weight hash not recorded; base model identifier and local cache use recorded.','Cellwise intervals descriptive without multiplicity correction.'])
dest=OUT/'audit_20260918.json'
with dest.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=report['status'],rows=audited,replayed=replayed,invalid=invalid,primary=comparisons['fresh_unknown_mixed_minus_known'],fresh={a:results[a]['fresh']['unknown'] for a in results}),indent=2))
