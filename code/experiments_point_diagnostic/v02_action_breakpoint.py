"""Action-endpoint diagnostic. Targets are never loaded by the API runner."""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tomsup.ktom_functions as kt
from check_s1_program_reference import PARAMS, MATRIX, parse_history, replay, scalar_state, sha
from v01_validation import flat, reference_code, restore
from v01_readout_revision_check import readout

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'v02_package_20260915'
OUT = ROOT / 'v02_gpt54_20260915'
CONDITIONS = ('H', 'K', 'S')
FIELD = 'p_opponent_1'
SCHEMA = {'type':'object', 'properties':{FIELD:{'type':'number'}}, 'required':[FIELD], 'additionalProperties':False}
DEPLOYMENT = os.environ['TOM_AZURE_DEPLOYMENT']
LIMIT = 8192

def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def now():
    return datetime.now(timezone.utc).isoformat()

def readout_guide():
    return '''Readout guide (equivalent to the source): sigmoid(x)=1/(1+exp(-x)).
For level0, q=sigmoid(m/sqrt(1+0.36*exp(v))), with m=own_states/p_op_mean0 and v=own_states/p_op_var0.
For level k>0, for i=0,...,k-1 compute z_i=sum_j exp(own_states/param_var/i/j)*(own_states/gradient/i/j)^2 over j=0,1,2;
q=sum_i own_states/p_k/i * sigmoid(own_states/p_op_mean/i / sqrt(1+0.36*z_i)).
Here q is B's belief about A, NOT the requested final answer.
If B is seeker (agent index0), delta=4*q-2. If B is hider (agent index1), delta=2-4*q.
The requested answer is clip(sigmoid(delta/exp(b_temp)+bias),0.001,0.999).
The source decision_function returns (logit of B action probability, B belief q): sigmoid of its FIRST returned value is the answer.
Only outer own_states fields enter current readout; opponent_states are used during recursive learning, not this final readout.
'''

def render(condition, row, rules):
    assert condition in CONDITIONS
    if condition == 'H':
        return row['original_prompt']
    text = row['original_prompt'].replace(
        'Use only the complete past observations below. No current choices or hidden opponent state are available.',
        'Use the complete past observations and the additional model information supplied below. No current or future choices are available.')
    text += '\nADDITIONAL MODEL INFORMATION:\nA denotes you; B is the opponent whose NEXT ACTION you must predict.\n'
    text += f'B follows exactly the supplied algorithm at level={row["level"]}, B agent index={row["b_role"]}, params={json.dumps(PARAMS)}.\n'
    text += ('Agent index0 is seeker; index1 is hider. PayoffMatrix.payoff(choice_agent0,choice_agent1,agent) returns '
             'matrix[agent,choice_agent0,choice_agent1], with matrix=[[[1,-1],[-1,1]],[[-1,1],[1,-1]]]. '
             'Arrays are NumPy arrays. Flattened state uses slash-separated dictionary/array paths; numeric dictionary keys identify simulated levels. '
             'Mean belief fields are logits; variance fields are log variances. Parameter insertion order is volatility,b_temp,bias. '
             'Start at init_k_tom(params,level). For each completed past pair, call learning_function(state,params,B_action,A_action,level,B_agent_index,p_matrix). '
             'Do not sample a new action. You have no execution tools. Return only the final probability in p_opponent_1; do not return state or explanation.\n')
    text += readout_guide() + '\nREFERENCE SOURCE:\n' + rules + '\nEND SOURCE\n'
    if condition == 'K':
        text += 'Use the given initialization, history and mechanism to infer the current state and predict B action1 probability.\n'
    else:
        text += ('For this condition, the TRUE CURRENT state after ALL completed past rounds is supplied below. '
                 'Read the probability from this state: do NOT initialize, replay or update it again. '
                 'Cached p_self and p_op entries have been removed at all depths; they are not needed for readout.\nCURRENT STATE:\n'
                 + json.dumps(row['current'], separators=(',',':')) + '\nPredict B action1 probability only.\n')
    return text

def prepare():
    assert not PACK.exists()
    g01 = json.loads((ROOT/'g01_inputs.json').read_text())
    g02 = json.loads((ROOT/'g02_inputs.json').read_text())
    source = ROOT/'eval_job_12878223/corrected_prior_full_history/rounds.csv'
    assert sha(source) == g01['source_hashes']['rounds.csv'] == g02['source_hashes']['rounds.csv']
    with source.open(encoding='utf-8', newline='') as f:
        raw = list(csv.DictReader(f))
    keyed = {(r['match_id'],int(r['decision_round'])):r for r in raw}
    assert len(keyed) == len(raw)
    beliefs = {(r['match_id'],r['decision_round']):r for r in g02['records']}
    rules = reference_code()
    code = {}
    exec(compile(rules, '<trusted-installed-simulator>', 'exec'), code)
    prompts, targets, errors = [], [], []
    pilot = []
    for rec in g01['records']:
        mid, t = rec['match_id'], rec['decision_round']
        original = keyed[(mid,t)]
        history = parse_history(rec)
        assert history == parse_history(beliefs[(mid,t)])
        role = 0 if rec['role']=='seeker' else 1
        b_role, level = 1-role, int(rec['opponent'][-1])
        assert int(original['llm_player_index']) == role
        assert original['opponent_name'] == rec['opponent']
        for i,(a,b,payoff) in enumerate(history,1):
            prev = keyed[(mid,i)]
            assert a == int(prev[f'choice_agent{role}']) and b == int(prev[f'choice_agent{b_role}'])
            assert payoff == int(float(prev['llm_payoff']))
        target = scalar_state(original[f'history_agent{b_role}'], 'p_self')
        y = int(original[f'choice_agent{b_role}'])
        assert abs(target-rec['oracle'])<1e-12 and y==rec['y']
        state, belief, probability = replay(history,level,b_role)
        row = {'original_prompt':rec['prompt'], 'level':level, 'b_role':b_role, 'current':flat(state)}
        assert all(not any(x in ('p_op','p_self') for x in key.split('/')) for key in row['current'])
        restored = restore(row['current'],state)
        restored_p = float(kt.inv_logit(kt.decision_function(restored,PARAMS,b_role,level,MATRIX)[0]))
        q = readout(row)
        delta = (4*q-2) * (1 if b_role==0 else -1)
        guide_p = min(.999,max(.001,1/(1+math.exp(-(delta/math.exp(PARAMS['b_temp'])+PARAMS['bias'])))))
        other = code['init_k_tom'](PARAMS,level)
        for a,b,_ in history:
            other = code['learning_function'](other,PARAMS,b,a,level,b_role,MATRIX)
        source_p = float(kt.inv_logit(code['decision_function'](other,PARAMS,b_role,level,MATRIX)[0]))
        error = max(abs(p-target) for p in (probability, restored_p, guide_p, source_p))
        assert error<1e-10
        errors.append(error)
        for c in CONDITIONS:
            prompts.append({'condition':c,'match_id':mid,'round':t,'prompt':render(c,row,rules)})
        if t==10:
            pilot.extend([(c,mid,t) for c in CONDITIONS])
        targets.append({'match_id':mid,'round':t,'role':rec['role'],'level':level,'p_true':target,'y':y,
                        'belief_true':belief,'frequency':(1+sum(b for _,b,_ in history))/(2+len(history))})
    assert len(prompts)==180 and len(targets)==60 and len(pilot)==18
    assert len({(r['condition'],r['match_id'],r['round']) for r in prompts})==180
    PACK.mkdir()
    dump(PACK/'prompts.json',prompts)
    dump(PACK/'targets.json',targets)
    dump(PACK/'pilot_keys.json',pilot)
    (PACK/'rules.py.txt').write_text(rules,encoding='utf-8')
    files = [Path(__file__), ROOT/'g01_inputs.json',ROOT/'g02_inputs.json',source,Path(kt.__file__),
             ROOT/'v01_validation.py',ROOT/'check_s1_program_reference.py',ROOT/'v01_readout_revision_check.py',
             ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm/azure_chat_generator.py']
    hashes = {str(p):sha(p) for p in files}
    hashes.update({str(p):sha(p) for p in PACK.iterdir()})
    dump(PACK/'preflight.json',{'passed':True,'items':180,'matches':6,'pilot_items':18,'max_reference_error':max(errors),
         'max_prompt_utf8_bytes':{c:max(len(r['prompt'].encode()) for r in prompts if r['condition']==c) for c in CONDITIONS},
         'hashes':hashes,'target':'B action1 probability and sampled action','reasoning':'medium','max_completion_tokens':LIMIT})
    print(json.dumps(json.loads((PACK/'preflight.json').read_text()),indent=2))

def valid_probability(text):
    obj = json.loads(text)
    assert set(obj)=={FIELD}
    p = obj[FIELD]
    assert type(p) in (float,int) and math.isfinite(p) and 0<=p<=1
    return float(p)

def key(r):
    return r['condition'],r['match_id'],r['round']

def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def run(stage):
    pre = json.loads((PACK/'preflight.json').read_text())
    assert pre['passed'] and all(sha(Path(p))==h for p,h in pre['hashes'].items())
    prompts = json.loads((PACK/'prompts.json').read_text())
    pilot = {tuple(x) for x in json.loads((PACK/'pilot_keys.json').read_text())}
    if stage=='pilot':
        OUT.mkdir(exist_ok=False)
    else:
        marker = json.loads((OUT/'pilot_completed.json').read_text())
        assert marker['valid']==18 and marker['requests']==18 and not marker['halted']
    prior_requests = read_rows(OUT/'requests.jsonl')
    prior_results = read_rows(OUT/'responses.jsonl')
    assert len(prior_requests)==len(prior_results), 'Uncertain request: no automatic duplicate'
    done = {key(r) for r in prior_results}
    assert len(done)==len(prior_results)
    selected = [r for r in prompts if (key(r) in pilot)==(stage=='pilot') and key(r) not in done]
    assert len(selected)==(18 if stage=='pilot' else 162)
    sys.path.insert(0,str(ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm'))
    from azure_chat_generator import AzureChatGenerator
    gen = AzureChatGenerator(deployment=DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE"))
    client = gen.client.with_options(max_retries=0,timeout=180)
    import importlib.metadata
    dump(OUT/(stage+'_manifest.json'),{'pid':os.getpid(),'started_utc':now(),'stage':stage,'selected_keys':[key(r) for r in selected],
         'preflight_sha256':sha(PACK/'preflight.json'),'script_sha256':sha(Path(__file__)),'deployment':DEPLOYMENT,
         'api_version':os.environ.get('AZURE_OPENAI_API_VERSION'),'reasoning':'medium','max_completion_tokens':LIMIT,
         'timeout_seconds':180,'sdk_retries':0,'concurrency':3,'openai_version':importlib.metadata.version('openai')})
    lock = threading.Lock(); halt = threading.Event(); results=[]
    def append(name,value):
        with (OUT/name).open('a',encoding='utf-8') as f:
            f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def call(r):
        with lock:
            if halt.is_set(): return
            number = len(prior_requests)+1
            request = {**r,'request_number':number,'stage':stage,'started_utc':now(),'schema':SCHEMA,
                       'reasoning_effort':'medium','max_completion_tokens':LIMIT}
            append('requests.jsonl',request);prior_requests.append(request)
        output={k:r[k] for k in ('condition','match_id','round')}
        output.update(request_number=number,stage=stage)
        started=time.monotonic()
        try:
            response=client.chat.completions.create(model=DEPLOYMENT,messages=[{'role':'user','content':r['prompt']}],
                reasoning_effort='medium',max_completion_tokens=LIMIT,
                response_format={'type':'json_schema','json_schema':{'name':'action_prediction','strict':True,'schema':SCHEMA}})
            output.update(response_id=response.id,model=response.model,usage=response.usage.model_dump() if response.usage else None,
                          text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason)
            assert 'gpt-5.4' in response.model.lower()
            assert output['finish_reason']=='stop'
            output.update(p=valid_probability(output['text']),valid=True)
        except Exception as exc:
            status=getattr(exc,'status_code',None)
            output.update(valid=False,error_type=type(exc).__name__,status_code=status)
            # Retain structured API error codes only; do not serialize credentials/headers/URLs.
            body=getattr(exc,'body',None)
            if isinstance(body,dict):
                err=body.get('error',body)
                if isinstance(err,dict):
                    output['api_error_fields']={k:err[k] for k in ('type','code','param') if k in err}
            if status in (400,401,403,404,429): halt.set()
        output.update(elapsed_seconds=time.monotonic()-started,completed_utc=now())
        with lock:
            append('responses.jsonl',output);results.append(output)
            print(json.dumps({k:output[k] for k in ('request_number','condition','match_id','valid','elapsed_seconds')}),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call,selected))
    dump(OUT/(stage+'_completed.json'),{'requests':len(results),'valid':sum(r['valid'] for r in results),'halted':halt.is_set(),
         'completed_utc':now(),'responses_sha256':sha(OUT/'responses.jsonl'),'requests_sha256':sha(OUT/'requests.jsonl')})
    audit()

def metrics(rows, predictions):
    if not rows: return {'n':0}
    p=np.asarray(predictions);y=np.asarray([r['y'] for r in rows]);q=np.asarray([r['p_true'] for r in rows])
    pc=np.clip(p,1e-6,1-1e-6)
    return {'n':len(rows),'brier':float(np.mean((p-y)**2)),'log_loss':float(np.mean(-(y*np.log(pc)+(1-y)*np.log(1-pc)))),
            'probability_mae':float(np.mean(abs(p-q))),'probability_mse':float(np.mean((p-q)**2))}

def audit():
    prompts={key(r):r for r in json.loads((PACK/'prompts.json').read_text())}
    targets={(r['match_id'],r['round']):r for r in json.loads((PACK/'targets.json').read_text())}
    requests=read_rows(OUT/'requests.jsonl');responses=read_rows(OUT/'responses.jsonl')
    assert len({key(r) for r in requests})==len(requests) and len({key(r) for r in responses})==len(responses)
    numbered={r['request_number']:r for r in requests}
    for r in responses:
        request=numbered[r['request_number']]
        assert key(r)==key(request) and request['prompt']==prompts[key(r)]['prompt'] and request['schema']==SCHEMA
        if r['valid']: assert valid_probability(r['text'])==r['p'] and r['finish_reason']=='stop'
    report={'timestamp_utc':now(),'requests':len(requests),'responses':len(responses),'conditions':{},'paired':{},
            'hashes':{p.name:sha(p) for p in (OUT/'responses.jsonl',OUT/'requests.jsonl',PACK/'prompts.json',PACK/'targets.json')}}
    valid={c:{(r['match_id'],r['round']):r for r in responses if r['condition']==c and r['valid']} for c in CONDITIONS}
    common=set.intersection(*(set(valid[c]) for c in CONDITIONS))
    report['all_conditions_common_n']=len(common)
    for c in CONDITIONS:
        got=[r for r in responses if r['condition']==c]
        keys=list(valid[c]);ts=[targets[k] for k in keys]
        report['conditions'][c]={'attempted':len(got),'valid':len(keys),'metrics':metrics(ts,[valid[c][k]['p'] for k in keys]),
            'constant':metrics(ts,[.5]*len(ts)),'frequency':metrics(ts,[t['frequency'] for t in ts]),
            'oracle_reference':metrics(ts,[t['p_true'] for t in ts]),
            'common_metrics':metrics([targets[k] for k in common],[valid[c][k]['p'] for k in common]),
            'finish_reasons':{str(reason):sum(r.get('finish_reason')==reason for r in got) for reason in set(r.get('finish_reason') for r in got)},
            'by_cell':{f'{role}_ToM{level}':metrics([targets[k] for k in keys if targets[k]['role']==role and targets[k]['level']==level],
                [valid[c][k]['p'] for k in keys if targets[k]['role']==role and targets[k]['level']==level]) for role in ('seeker','hider') for level in range(3)}}
    for a,b in (('K','H'),('S','K'),('S','H')):
        keys=sorted(set(valid[a])&set(valid[b]))
        diffs={metric:[] for metric in ('brier','probability_mse')}
        mids=sorted({k[0] for k in keys})
        for mid in mids:
            ks=[k for k in keys if k[0]==mid];ts=[targets[k] for k in ks]
            ma=metrics(ts,[valid[a][k]['p'] for k in ks]);mb=metrics(ts,[valid[b][k]['p'] for k in ks])
            for m in diffs: diffs[m].append(ma[m]-mb[m])
        report['paired'][a+'-'+b]={'n':len(keys),'matches':len(mids),'mean_match_differences':{m:float(np.mean(v)) if v else None for m,v in diffs.items()}}
        if len(keys)==60:
            rng=np.random.default_rng(20260915)
            draws=rng.integers(0,len(mids),size=(10000,len(mids)))
            report['paired'][a+'-'+b]['exploratory_match_bootstrap95']={m:np.quantile(np.asarray(v)[draws].mean(axis=1),[.025,.975]).tolist() for m,v in diffs.items()}
    report['usage']={field:sum((r.get('usage') or {}).get(field,0) for r in responses) for field in ('prompt_tokens','completion_tokens','total_tokens')}
    path=OUT/('audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    dump(path,report)
    print(json.dumps({'audit':str(path),'requests':len(requests),'responses':len(responses),'common_n':len(common),'conditions':{c:{k:v for k,v in report['conditions'][c].items() if k in ('attempted','valid','metrics','finish_reasons')} for c in CONDITIONS}},indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','pilot','full','audit'))
    stage=parser.parse_args().stage
    if stage=='prepare': prepare()
    elif stage=='audit': audit()
    else: run(stage)
