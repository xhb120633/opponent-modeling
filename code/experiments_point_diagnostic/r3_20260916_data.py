"""Fresh controlled development data. No model/API calls; exclusive output files."""
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import tomsup.ktom_functions as kt
from tomsup.payoffmatrix import PayoffMatrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'r3_20260916_package'
PARAMS = {'volatility': -2, 'b_temp': -1, 'bias': 0}
MATRIX = PayoffMatrix(name='custom', predefined=np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]]))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def flatten(state):
    result = {}
    def visit(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in ('p_op', 'p_self'):
                    visit(v, path+[str(k)])
        elif isinstance(x, (list, np.ndarray)):
            for k, v in enumerate(x): visit(v, path+[str(k)])
        else: result['/'.join(path)] = float(x)
    visit(state, [])
    assert all(math.isfinite(v) for v in result.values())
    return result

def probability(state, level, role):
    logit, _ = kt.decision_function(copy.deepcopy(state), PARAMS, role, level, MATRIX)
    return float(kt.inv_logit(logit))

def build():
    assert not OUT.exists(), 'Never overwrite a prepared package'
    matches=[]
    for role in range(2):
        for level in range(3):
            for rep in range(60):
                split = 'train' if rep < 40 else 'val' if rep < 50 else 'test'
                seed=202609160 + role*10000 + level*1000 + rep
                rng=np.random.default_rng(seed)
                state=kt.init_k_tom(PARAMS,level)
                keys=list(flatten(state)); initial=[flatten(state)[k] for k in keys]
                m={'id':f'r3_A{role}_Btom{level}_{rep:03d}', 'split':split,'seed':seed,
                   'a_role':role,'b_role':1-role,'level':level,'keys':keys,'initial':initial,'rounds':[]}
                history=[]; prior=initial
                # A policy is chosen independently of B hidden state. Changes are fixed before play.
                phase=rng.uniform(0,2*np.pi); offset=rng.uniform(-0.6,0.6)
                m['a_policy']={'offset':float(offset),'phase':float(phase),'formula':'sigmoid(offset+1.4*sin(phase+0.9*t))'}
                for t in range(10):
                    current=flatten(state); vector=[float(f'{current[k]:.9g}') for k in keys]
                    p=probability(state,level,1-role)
                    pa=1/(1+np.exp(-(offset+1.4*np.sin(phase+0.9*t))))
                    a=int(rng.random()<pa); b=int(rng.random()<p)
                    m['rounds'].append({'round':t+1,'history':copy.deepcopy(history),'previous':prior,
                                        'state':vector,'p':p,'y':b})
                    history.append([a,b]); prior=vector
                    if t<9:
                        state=kt.learning_function(state,PARAMS,b,a,level,1-role,MATRIX)
                matches.append(m)
    assert len({m['id'] for m in matches})==360
    counts={s:sum(m['split']==s for m in matches) for s in ('train','val','test')}
    assert counts=={'train':240,'val':60,'test':60}
    OUT.mkdir()
    for split in counts: dump(OUT/f'{split}.json',[m for m in matches if m['split']==split])
    dump(OUT/'manifest.json',{'counts':counts,'rounds':10,'params':PARAMS,'seed':202609160,
        'simulator_sha256':sha(kt.__file__),'generator_sha256':sha(__file__),
        'files':{s:sha(OUT/f'{s}.json') for s in counts},'state_sizes':{str(k):len(flatten(kt.init_k_tom(PARAMS,k))) for k in range(3)},
        'deployment_solver':False,'known_level_params':True,'sealed_final_access':False})
    print(json.dumps({'output':str(OUT),'counts':counts}))

def prompt(match,row,arm,prior=None):
    # Only explicitly permitted columns are read; target/current/future mutations cannot leak.
    text=(f"Forecast B's next action1 probability. A index={match['a_role']}; B index={match['b_role']}. "
          f"Index0 is seeker (matching actions rewarded); index1 is hider (different actions rewarded). "
          f"B is the training task's ToM{match['level']} simulator, params={json.dumps(PARAMS)}. "
          "Only completed (A_action,B_action) pairs are observations. Current/future choices are unavailable.\n"
          f"Decision round={row['round']}; complete past={json.dumps(row['history'],separators=(',',':'))}\n")
    text+=('State coordinates are the complete recursive numeric state, excluding cached probabilities. '
           'Mean fields are logits; variance fields are log variances; p_k are mixture probabilities. '
           f"Vector key order={json.dumps(match['keys'],separators=(',',':'))}\n"
           f"Public state at task INITIALIZATION={json.dumps(match['initial'],separators=(',',':'))}. This is not the current state after observations.\n")
    if arm=='A': return text+'Return exactly JSON {"p_opponent_1": number}, no explanation.'
    if row['round']==1:
        text+=f"Public INITIAL state={json.dumps(match['initial'],separators=(',',':'))}. No observation to update.\n"
    else:
        text+=f"PREVIOUS estimated state={json.dumps(prior,separators=(',',':'))}. Update using the LAST completed pair above.\n"
    return text+'Return exactly JSON {"state": [updated numeric coordinates], "p_opponent_1": number}. No explanation.'

def answer(row,arm):
    obj={'p_opponent_1':float(f"{row['p']:.9g}")}
    if arm=='B': obj={'state':row['state'],**obj}
    return json.dumps(obj,separators=(',',':'))

def parse(text,match,arm,allow_fence=True):
    try:
        if allow_fence and isinstance(text,str):
            wrapped=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*',text,re.DOTALL)
            if wrapped:text=wrapped.group(1)
        obj=json.loads(text)
        assert set(obj)==({'state','p_opponent_1'} if arm=='B' else {'p_opponent_1'})
        p=obj['p_opponent_1']; assert type(p) in (float,int) and math.isfinite(p) and 0<=p<=1
        if arm=='B':
            vals=obj['state']; assert isinstance(vals,list) and len(vals)==len(match['keys'])
            assert all(type(v) in (float,int) and math.isfinite(v) for v in vals)
            groups={}
            for key,v in zip(match['keys'],vals):
                if '/p_k/' in key:
                    assert 0<=v<=1
                    groups.setdefault(key.rsplit('/',1)[0],[]).append(v)
            assert all(abs(sum(v)-1)<1e-4 for v in groups.values())
        return obj
    except (ValueError,TypeError,AssertionError): return None

if __name__=='__main__': build()
