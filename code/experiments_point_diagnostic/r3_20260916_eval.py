"""Post-selection common development test. No retraining or API calls."""
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import r3_20260916_data as d
import r3_20260916_train as t

DEST=t.REMOTE/'common_eval'

def measure(rows):
    valid=[r for r in rows if r['valid']]
    n=len(rows)
    result={'n':n,'valid':len(valid),'missing':n-len(valid),
        'penalized_probability_mse':sum((r['p']-r['q'])**2 if r['valid'] else 1 for r in rows)/n if n else None,
        'near_exact_n':sum(abs(r['p']-r['q'])<=.001 for r in valid)}
    if valid:
        p=np.array([r['p'] for r in valid]);q=np.array([r['q'] for r in valid]);y=np.array([r['y'] for r in valid]);pc=np.clip(p,1e-6,1-1e-6)
        result['valid_only']={'mae':float(np.mean(abs(p-q))),'mse':float(np.mean((p-q)**2)),
          'brier':float(np.mean((p-y)**2)),'logloss':float(np.mean(-(y*np.log(pc)+(1-y)*np.log(1-pc))))}
    return result

def complete_rows(matches,output,arm):
    observed={(r['match_id'],r['round']):r for r in output['rows']}
    assert len(observed)==len(output['rows'])
    rows=[]
    for m in matches:
        for truth in m['rounds']:
            key=(m['id'],truth['round']);r=observed.get(key)
            item={'match_id':key[0],'round':key[1],'a_role':m['a_role'],'level':m['level'],'q':truth['p'],'y':truth['y'],
              'valid':bool(r and r['valid']),'p':r['p'] if r else None,'raw':r['raw'] if r else None,
              'missing_reason':None if r and r['valid'] else 'invalid_output' if r else 'prior_chain_stopped'}
            if r:
                assert r['q']==truth['p'] and r['y']==truth['y']
                parsed=d.parse(r['raw'],m,arm)
                assert (parsed is not None)==r['valid']
                if parsed:assert parsed['p_opponent_1']==r['p']
                if parsed and arm=='B':item['state_coordinate_mae']=float(np.mean(abs(np.asarray(parsed['state'])-truth['state'])))
            rows.append(item)
    assert len(rows)==600 and len({(r['match_id'],r['round']) for r in rows})==600
    assert sum(r['valid'] for r in rows)==output['valid']
    return rows

def main():
    assert not DEST.exists(),'Exclusive output: no implicit rerun'
    package=json.loads((d.OUT/'manifest.json').read_text())
    assert all(d.sha(d.OUT/f'{s}.json')==package['files'][s] for s in ('train','val','test'))
    selected={};sources={}
    for arm in ('A','B'):
        directory=t.REMOTE/arm
        runtime=json.loads((directory/'runtime.json').read_text())
        assert runtime['driver_sha256']==d.sha(Path(t.__file__))
        assert runtime['renderer_sha256']==d.sha(Path(d.__file__))
        assert runtime['manifest_sha256']==d.sha(d.OUT/'manifest.json')
        done=json.loads((directory/'completion.json').read_text())
        assert done['status']=='training_completed_not_tested'
        path=Path(done['selected_path']).resolve();assert path.is_relative_to(directory.resolve())
        assert d.sha(path/'adapter_model.safetensors')==done['selected_hash']
        selected[arm]=path;sources[arm]=done
    matches=json.loads((d.OUT/'test.json').read_text());assert len(matches)==60
    DEST.mkdir()
    d.dump(DEST/'runtime.json',{'started_unix':time.time(),'job':os.getenv('SLURM_JOB_ID'),'selected':sources,
      'hashes':{str(p):d.sha(p) for p in (Path(__file__),Path(t.__file__),Path(d.__file__),d.OUT/'test.json',d.OUT/'manifest.json')},'seed':2026})
    set_seed(2026)
    tok=AutoTokenizer.from_pretrained(t.MODEL,local_files_only=True);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(t.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    conditions={}
    for name,arm,trained in [('initial_A','A',False),('initial_B','B',False),('trained_A','A',True),('trained_B','B',True)]:
        if trained:model=PeftModel.from_pretrained(model,str(selected[arm]))
        started=time.monotonic();output,_=t.rollout(model,tok,matches,arm)
        rows=complete_rows(matches,output,arm);conditions[name]=rows
        d.dump(DEST/(name+'.json'),{'elapsed_seconds':time.monotonic()-started,'metrics':measure(rows),'rows':rows})
        if trained:model=model.unload()
    baselines={}
    for name in ('constant','frequency'):
        rows=[]
        for m in matches:
            for r in m['rounds']:
                p=.5 if name=='constant' else (1+sum(b for a,b in r['history']))/(2+len(r['history']))
                rows.append({'p':p,'q':r['p'],'y':r['y'],'valid':True})
        baselines[name]=measure(rows)
    grouped={};rng=np.random.default_rng(20260916)
    byid={name:{m['id']:measure([r for r in rows if r['match_id']==m['id']])['penalized_probability_mse'] for m in matches} for name,rows in conditions.items()}
    groups=[[m['id'] for m in matches if m['a_role']==role and m['level']==level] for role in (0,1) for level in (0,1,2)]
    assert all(len(g)==10 for g in groups)
    deltas={mid:byid['trained_B'][mid]-byid['trained_A'][mid] for mid in byid['trained_A']}
    boot=[float(np.mean([deltas[mid] for group in groups for mid in rng.choice(group,len(group))])) for _ in range(5000)]
    for name,rows in conditions.items():
        grouped[name]={f'A{role}_ToM{level}':measure([r for r in rows if r['a_role']==role and r['level']==level]) for role in (0,1) for level in (0,1,2)}
    common_keys={(r['match_id'],r['round']) for r in conditions['trained_A'] if r['valid']} & {(r['match_id'],r['round']) for r in conditions['trained_B'] if r['valid']}
    metrics={'conditions':{name:measure(rows) for name,rows in conditions.items()},'by_cell':grouped,'baselines':baselines,
      'B_minus_A_penalized_mse':{'mean':float(np.mean(list(deltas.values()))),'match_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()},
      'same_valid_keys':{name:measure([r for r in conditions[name] if (r['match_id'],r['round']) in common_keys]) for name in ('trained_A','trained_B')},
      'limitations':'One seed; complete development matches; state-supervised recurrence package versus direct prediction, not isolated causal mechanism.'}
    d.dump(DEST/'metrics.json',metrics)
    d.dump(DEST/'completed.json',{'completed_unix':time.time(),'hashes':{p.name:d.sha(p) for p in DEST.glob('*.json')}})
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
