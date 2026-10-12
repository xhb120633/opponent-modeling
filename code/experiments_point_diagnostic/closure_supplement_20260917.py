"""C03 preparation and same-observation baselines. No API calls in this module."""
import ast, copy, hashlib, json, re, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gpt_chain_elicitation_20260917 as base

ROOT=Path(__file__).resolve().parent
PACK=ROOT/'closure_package_20260917'
OUT=ROOT/'closure_baseline_20260917'
SOURCE=ROOT/'eval_job_12878223/corrected_prior_full_history'
sys.path.insert(0,str(ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm'))
from evaluate_d04_fixed_observation import load_source, observation

def belief(text):
    node=ast.parse(text,mode='eval').body
    for name in ('internal_states','own_states','p_op'):
        assert isinstance(node,ast.Dict)
        hits=[v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value==name]
        assert len(hits)==1
        node=hits[0]
    if isinstance(node,ast.Call):
        assert isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
        assert node.func.value.id=='np' and node.func.attr in ('float64','float32') and len(node.args)==1 and not node.keywords
        node=node.args[0]
    value=ast.literal_eval(node)
    assert type(value) in (int,float) and 0<=value<=1
    return float(value)

def features(r):
    h=r['history'];assert len(h)==r['round']-1 and len(h)<=9
    x=[int(r['role']=='hider'),len(h)]+[v for pair in h for v in pair]+[-1]*(18-2*len(h))
    for j in (0,1):
        x.append((1+sum(p[j] for p in h))/(2+len(h)))
        x.extend(float(np.mean([p[j] for p in h[-n:]])) if h else .5 for n in (1,3,5))
    return x

def prepare():
    assert not PACK.exists()
    old=base.read(ROOT/'g01_inputs.json'); frame=load_source(SOURCE)
    assert all(base.sha(SOURCE/n)==h for n,h in old['source_hashes'].items())
    header=old['records'][0]['prompt'].split('Your role:')[0]
    rows=[]
    for mid,g in frame.groupby('match_id',sort=True):
        rep=int(re.search(r'rep(\d+)$',mid)[1]); split='test' if rep<5 else 'val' if rep<10 else 'train'
        history=[]
        for r in g.itertuples():
            idx=int(r.llm_player_index);level=int(r.opponent_name[-1])
            hist_text='\n'.join(f'round={t}; self_action={a}; opponent_action={b}; self_payoff={pay}' for t,(a,b,pay) in enumerate(history,1)) or 'none (first round)'
            prompt=header+f'Your role: {r.llm_role}\nDecision round: {r.decision_round}\nComplete past observations:\n'+hist_text
            rows.append(dict(match_id=mid,round=int(r.decision_round),role=r.llm_role,level=level,rep=rep,split=split,
                history=[[a,b] for a,b,p in history],prompt=prompt,p_true=float(r.oracle_tom_p_self),
                belief_true=belief(getattr(r,f'history_agent{1-idx}')),y=int(r.opponent_action)))
            history.append(observation(r))
    lookup={(r['match_id'],r['round']):r for r in rows}
    for oldrow in old['records']:
        r=lookup[oldrow['match_id'],oldrow['decision_round']]
        assert r['prompt']==oldrow['prompt'] and abs(r['p_true']-oldrow['oracle'])<1e-12
    oldb=base.read(ROOT/'g02_inputs.json')['records']
    for r in oldb:assert abs(lookup[r['match_id'],r['decision_round']]['belief_true']-r['belief_target'])<1e-12
    assert len(rows)==1800 and len({r['match_id'] for r in rows})==180
    for r in rows:
        assert features(r)==features({**r,'level':99,'p_true':-99,'belief_true':-99,'y':9,'match_id':'hidden'})
    PACK.mkdir();base.dump(PACK/'rows.json',rows)
    base.dump(PACK/'replication_inputs.json',[{k:r[k] for k in ('match_id','round','prompt')} for r in rows if 1<=r['rep']<=4])
    paths=[Path(__file__),ROOT/'CLOSURE_SUPPLEMENT_20260917.md',ROOT/'g01_inputs.json',ROOT/'g02_inputs.json',PACK/'rows.json',PACK/'replication_inputs.json']+[SOURCE/n for n in old['source_hashes']]
    base.dump(PACK/'manifest.json',dict(hashes={str(p):base.sha(p) for p in paths},counts={s:len({r['match_id'] for r in rows if r['split']==s}) for s in ('train','val','test')},checks=['source and original-GPT targets/prompts','whole-match splits','features ignore targets/current action/identity']))
    print('Prepared180matches;120train/30val/30test;24new GPT matches',flush=True)

def metric(rs,p,target):
    y=np.array([r[target] for r in rs]);p=np.asarray(p)
    return dict(n=len(rs),mae=float(np.mean(abs(y-p))),mse=float(np.mean((y-p)**2)),r=base.corr(y,p))

def baseline():
    pre=base.read(PACK/'manifest.json');assert all(base.sha(p)==h for p,h in pre['hashes'].items())
    rows=[r for r in base.read(PACK/'rows.json') if r['round']>1]
    sets={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    assert not ({r['match_id'] for r in sets['train']} & {r['match_id'] for r in sets['test']})
    OUT.mkdir();predictions=[];summary={};rng=np.random.default_rng(20260917)
    for target in ('p_true','belief_true'):
        candidates={f'extratrees_leaf{leaf}':ExtraTreesRegressor(n_estimators=400,min_samples_leaf=leaf,max_features=1.,random_state=20260917,n_jobs=4) for leaf in (2,8)}
        candidates.update({f'ridge{a}':make_pipeline(StandardScaler(),Ridge(alpha=a)) for a in (1,10)})
        scores={}
        for name,model in candidates.items():
            model.fit([features(r) for r in sets['train']],[r[target] for r in sets['train']])
            scores[name]=metric(sets['val'],np.clip(model.predict([features(r) for r in sets['val']]),0,1),target)
        name=min(scores,key=lambda n:scores[n]['mse']);rs=sets['test'];ps=np.clip(candidates[name].predict([features(r) for r in rs]),0,1)
        frequency=[(1+sum(h[1 if target=='p_true' else 0] for h in r['history']))/(2+len(r['history'])) for r in rs]
        groups={}
        sels={'all':list(range(len(rs))),'old_six':[i for i,r in enumerate(rs) if r['rep']==0],'new_24':[i for i,r in enumerate(rs) if r['rep']>0]}
        sels.update({f'{role}_ToM{l}':[i for i,r in enumerate(rs) if r['role']==role and r['level']==l] for role in ('seeker','hider') for l in (0,1,2)})
        for g,ii in sels.items():
            rr=[rs[i] for i in ii];groups[g]={n:metric(rr,[pp[i] for i in ii],target) for n,pp in [('model',ps),('constant',[.5]*len(rs)),('frequency',frequency)]}
        matchdiff={mid:float(np.mean([(ps[i]-r[target])**2-(.5-r[target])**2 for i,r in enumerate(rs) if r['match_id']==mid])) for mid in {r['match_id'] for r in rs}}
        cells=[[mid for mid in matchdiff if next(r for r in rs if r['match_id']==mid)['role']==role and next(r for r in rs if r['match_id']==mid)['level']==l] for role in ('seeker','hider') for l in (0,1,2)]
        boot=[np.mean([matchdiff[mid] for cell in cells for mid in rng.choice(cell,len(cell))]) for _ in range(3000)]
        summary[target]=dict(selected=name,validation=scores,groups=groups,mse_minus_constant_ci=np.quantile(boot,[.025,.975]).tolist())
        predictions.extend(dict(match_id=r['match_id'],round=r['round'],target=target,p=float(p),truth=r[target]) for r,p in zip(rs,ps))
    base.dump(OUT/'predictions.json',predictions);base.dump(OUT/'results.json',summary)
    base.dump(OUT/'audit.json',dict(passed=True,package_sha256=base.sha(PACK/'manifest.json'),code_sha256=base.sha(__file__),output_hashes={n:base.sha(OUT/n) for n in ('predictions.json','results.json')},target_leakage_checks=True,sealed_final=False))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'baseline':baseline}[sys.argv[1]]()
