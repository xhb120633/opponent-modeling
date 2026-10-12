"""Frozen, bounded GPT-5.4 capability diagnostics. Credentials never serialized."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tomsup.ktom_functions as kt
from check_s1_program_reference import PARAMS, MATRIX, parse_history, replay, sha

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'v01_package_20260915'
OUT = ROOT / 'v01_gpt54_20260915'
FIELD = 'opponent_belief_my_action_1_probability'
BIN = 'opponent_belief_my_action_bin'
CONDITIONS = ('B', 'C', 'D', 'E', 'F')

def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def flat(state):
    out = {}
    def visit(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in ('p_op', 'p_self'):
                    visit(v, path + [str(k)])
        elif isinstance(x, (np.ndarray, list)):
            for i, v in enumerate(x):
                visit(v, path + [str(i)])
        else:
            out['/'.join(path)] = float(x)
    visit(state, [])
    assert all(math.isfinite(v) for v in out.values())
    return out

def restore(values, template):
    assert set(values) == set(flat(template))
    def visit(x, path):
        if isinstance(x, dict):
            return {k: visit(v, path+[str(k)]) for k,v in x.items() if k not in ('p_op','p_self')}
        if isinstance(x, np.ndarray):
            return np.asarray([visit(v, path+[str(i)]) for i,v in enumerate(x)])
        v = values['/'.join(path)]
        assert type(v) in (int, float) and math.isfinite(v)
        return float(v)
    return visit(template, [])

class Compact(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        for a in node.args.args + node.args.kwonlyargs:
            a.annotation = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node

def reference_code():
    names = ['p_op_var0_update','p_op_mean0_update','p_opk_approx_fun','p_k_udpate',
             'param_var_update','param_mean_update','gradient_update','p_op0_fun','p_opk_fun',
             'expected_payoff_fun','softmax','learning_function','decision_function','init_k_tom']
    head = 'import numpy as np\nimport copy\nfrom scipy.special import expit as inv_logit, logit\n'
    text = head + '\n\n'.join(ast.unparse(Compact().visit(ast.parse(inspect.getsource(getattr(kt,n))))) for n in names)
    return text

def schema(condition, state):
    props = {FIELD: {'type':'number'}}
    if condition == 'F':
        props = {BIN: {'type':'integer','enum':list(range(11))}}
    if condition in ('C','E'):
        state_props = {k:{'type':'number'} for k in state}
        props['state'] = {'type':'object','properties':state_props,'required':list(state_props),'additionalProperties':False}
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}

def prompt(condition, row, rules, prior=None):
    if condition == 'F':
        return row['original_prompt'] + '\nReport this belief in the integer field opponent_belief_my_action_bin: 0 means 0%, 5 means 50%, 10 means 100%. Round your estimated probability to the nearest tenth, with half bins rounded up. This is a single belief-bin report, not an action-chain task.'
    intro = ('A is the evaluated player; B is the algorithmic opponent. Target B.p_op: B\'s subjective probability that A chooses1, NOT B\'s action probability. '
             'You have no tools; perform the requested inference/calculation and return the structured answer. '\
             'The following source is the exact algorithm definition, not approximate verbal ToM terminology. '
             'Agent index0 is seeker and1 hider. PayoffMatrix.payoff(choice_agent0,choice_agent1,agent) returns '
             'matrix[agent,choice_agent0,choice_agent1], with matrix=[[[1,-1],[-1,1]],[[-1,1],[1,-1]]]. '
             'Arrays in JSON are NumPy arrays for source arithmetic. State is flattened: slash-separated keys are dictionary/array paths; numerical dictionary keys identify recursively simulated levels. '
             'p_op_mean0 and p_op_mean are logits; variance fields are LOG variances. Omitted cached p_op/p_self fields are not used by learning_function/decision_function. '
             'decision_function returns (LOGIT of B action probability, B belief probability); report the SECOND value. '
             'Preserve parameters in insertion order volatility,b_temp,bias.\nREFERENCE SOURCE:\n' + rules + '\nEND SOURCE\n')
    intro += f'B level={row["level"]}; B agent index={row["b_role"]}; params={json.dumps(PARAMS)}; decision round={row["round"]}.\n'
    if condition == 'B':
        return intro + ('Start from init_k_tom(params,level). Replay each completed action pair in order using learning_function(state,params,B_action,A_action,level,B_agent_index,p_matrix), then compute decision_function. Do not sample actions. '
                        'Only completed past action pairs (A_action,B_action) follow:\n' + json.dumps(row['history']) + '\nReturn B current belief probability only.')
    if condition == 'D':
        return intro + 'No update is requested. Compute decision_function from this CURRENT state and return its SECOND value:\n' + json.dumps(row['current'], separators=(',',':'))
    previous = row['previous'] if prior is None else prior
    return intro + ('Use the supplied PREVIOUS state below. If new_observation is null, do not update (initial decision); otherwise call learning_function once with self_choice=B_action and op_choice=A_action. '
                    'Return the entire updated state in the identical flat layout AND the SECOND value of decision_function. Do not add cached p_op/p_self to the state. '
                    'Use sufficient numerical precision (at least6 decimal digits where needed).\nPREVIOUS STATE:\n' + json.dumps(previous,separators=(',',':')) +
                    '\nnew_observation (A_action,B_action):' + json.dumps(row['history'][-1] if row['history'] else None))

def prepare():
    assert not PACK.exists()
    frozen = json.loads((ROOT/'g02_inputs.json').read_text())
    check = json.loads((ROOT/'s1_program_reference_20260915.json').read_text())
    assert check['summary']['replay_passed']
    assert check['hashes'][str(ROOT/'g02_inputs.json')] == sha(ROOT/'g02_inputs.json')
    rules = reference_code()
    # Execute only AST-transformed trusted installed simulator code for equivalence tests.
    reference = {}
    exec(compile(rules,'<trusted-local-simulator-reference>','exec'),reference)
    rows=[]; max_prompt=0; max_answer=0; max_diff=0
    for r in frozen['records']:
        hist = parse_history(r); level=int(r['opponent'][-1]); b_role=1 if r['role']=='seeker' else 0
        current, p, _ = replay(hist,level,b_role)
        previous = replay(hist[:-1],level,b_role)[0] if hist else kt.init_k_tom(PARAMS,level)
        cur, prev = flat(current), flat(previous)
        assert set(cur)==set(prev)
        for k in cur:
            assert not any(x in ('p_op','p_self') for x in k.split('/'))
        restored=restore(cur,current)
        assert abs(kt.decision_function(restored,PARAMS,b_role,level,MATRIX)[1]-p)<1e-12
        state=reference['init_k_tom'](PARAMS,level)
        for a,b,_ in hist:
            state=reference['learning_function'](state,PARAMS,b,a,level,b_role,MATRIX)
        diff=max(abs(flat(state)[k]-cur[k]) for k in cur)
        max_diff=max(max_diff,diff)
        assert diff<1e-10 and abs(reference['decision_function'](state,PARAMS,b_role,level,MATRIX)[1]-r['belief_target'])<1e-10
        row={'match_id':r['match_id'],'round':r['decision_round'],'level':level,'b_role':b_role,
             'history':[[a,b] for a,b,_ in hist],'previous':prev,'current':cur,'original_prompt':r['prompt']}
        # Targets deliberately stored separately; prompt function never receives them.
        for condition in CONDITIONS:
            txt=prompt(condition,row,rules)
            max_prompt=max(max_prompt,len(txt.encode('utf-8')))
        max_answer=max(max_answer,len(json.dumps({'state':cur,FIELD:p}).encode('utf-8')))
        rows.append(row)
    assert len(rows)==60 and max_answer<4096 and max_prompt<60000
    PACK.mkdir()
    dump(PACK/'items.json',rows)
    dump(PACK/'targets.json',[{'match_id':r['match_id'],'round':r['decision_round'],'p':r['belief_target']} for r in frozen['records']])
    (PACK/'rules.py.txt').write_text(rules,encoding='utf-8')
    dump(PACK/'preflight.json',{'passed':True,'n':60,'max_reference_state_utf8_bytes':max_answer,
         'max_prompt_utf8_bytes':max_prompt,'length_check':'conservative byte bound, not exact tokens; reasoning budget tested by capability calls',
         'max_transformed_source_state_difference':max_diff,
         'source_sha256':sha(Path(kt.__file__)),'g02_sha256':sha(ROOT/'g02_inputs.json'),
         'script_sha256':sha(Path(__file__)),'items_sha256':sha(PACK/'items.json'),
         'targets_sha256':sha(PACK/'targets.json'),'rules_sha256':sha(PACK/'rules.py.txt'),
         'max_requests':300,'max_completion_tokens':4096,'concurrency':3,'capability_order':['D ToM2 round10','C ToM2 round10']})
    print((PACK/'preflight.json').read_text())

def validate_state(state, template, level, b_role):
    restored=restore(state,template)
    for key,value in state.items():
        if '/p_k/' in key:
            assert 0<=value<=1
    def check(s):
        own=s['own_states']
        if 'p_k' in own:
            assert abs(float(np.sum(own['p_k']))-1)<1e-4
        for child in s['opponent_states'].values():
            check(child)
    check(restored)
    with np.errstate(all='raise'):
        p=float(kt.decision_function(restored,PARAMS,b_role,level,MATRIX)[1])
    assert math.isfinite(p) and 0<=p<=1
    return p

def run():
    pre=json.loads((PACK/'preflight.json').read_text())
    assert pre['passed'] and pre['script_sha256']==sha(Path(__file__))
    for name in ('items','targets'):
        assert pre[name+'_sha256']==sha(PACK/(name+'.json'))
    assert pre['rules_sha256']==sha(PACK/'rules.py.txt')
    assert not OUT.exists(), 'No duplicate or implicit resume'
    sys.path.insert(0,str(ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm'))
    from azure_chat_generator import AzureChatGenerator
    generator=AzureChatGenerator(deployment=os.environ['TOM_AZURE_DEPLOYMENT'],env_file=os.environ.get("TOM_AZURE_ENV_FILE"))
    client=generator.client.with_options(max_retries=0,timeout=180)
    rows=json.loads((PACK/'items.json').read_text()); rules=(PACK/'rules.py.txt').read_text()
    OUT.mkdir()
    dump(OUT/'manifest.json',{'pid':os.getpid(),'preflight':pre,'deployment':generator.deployment,
         'reasoning':'medium','timeout_seconds':180,'retry':0,'conditions':list(CONDITIONS),'budget':300})
    lock=threading.Lock(); counter=[0]; stop=threading.Event(); completed={}
    def call(condition,row,prior=None):
        txt=prompt(condition,row,rules,prior)
        result={'condition':condition,'match_id':row['match_id'],'round':row['round']}
        with lock:
            assert counter[0]<300
            counter[0]+=1; result['request_number']=counter[0]
            with (OUT/'requests.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({**result,'prompt':txt,'schema':schema(condition,row['current'])})+'\n')
                f.flush()
        try:
            response=client.chat.completions.create(model=generator.deployment,
                messages=[{'role':'user','content':txt}],reasoning_effort='medium',max_completion_tokens=4096,
                response_format={'type':'json_schema','json_schema':{'name':'tom_validation','strict':True,'schema':schema(condition,row['current'])}})
            result.update(response_id=response.id,model=response.model,usage=response.usage.model_dump() if response.usage else None,
                          text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason)
            assert 'gpt-5.4' in response.model.lower()
            assert result['finish_reason']=='stop'
            obj=json.loads(result['text'])
            assert set(obj)==set(schema(condition,row['current'])['properties'])
            p=obj[BIN]/10 if condition=='F' else obj[FIELD]
            assert type(p) in (float,int) and math.isfinite(p) and 0<=p<=1
            if condition=='F':
                assert type(obj[BIN]) is int and 0<=obj[BIN]<=10
            result.update(p=p,valid=True)
            if condition in ('C','E'):
                template=kt.init_k_tom(PARAMS,row['level'])
                calculated=validate_state(obj['state'],template,row['level'],row['b_role'])
                result.update(state=obj['state'],state_readout=calculated,state_report_gap=abs(calculated-p))
        except Exception as exc:
            result.update(valid=False,error_type=type(exc).__name__,status_code=getattr(exc,'status_code',None))
            if result.get('status_code') in (400,401,403,404,429):
                stop.set()
        with lock:
            with (OUT/'responses.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(result,allow_nan=False)+'\n'); f.flush()
            completed[(condition,row['match_id'],row['round'])]=result
            print(f'completed={len(completed)} requested={counter[0]} condition={condition} round={row["round"]} valid={result["valid"]}',flush=True)
        return result
    # Hardest schema/readout case; capability calls are part of the final300, no retries.
    gate_row=next(r for r in rows if r['level']==2 and r['b_role']==1 and r['round']==10)
    for condition in ('D','C'):
        if not call(condition,gate_row)['valid']:
            dump(OUT/'stopped.json',{'reason':'capability_failure','condition':condition,'requests':counter[0]})
            return
    groups=[(c,mid) for c in CONDITIONS for mid in dict.fromkeys(r['match_id'] for r in rows)]
    def chain(group):
        condition,mid=group; prior=None
        for row in [r for r in rows if r['match_id']==mid]:
            if stop.is_set(): return
            key=(condition,mid,row['round'])
            if key in completed: continue
            result=call(condition,row,prior)
            if condition=='E':
                if not result['valid']: return
                prior=result['state']
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(chain,groups))
    dump(OUT/'completed.json',{'requests':counter[0],'responses':len(completed),'halted':stop.is_set(),
                             'responses_sha256':sha(OUT/'responses.jsonl'),'requests_sha256':sha(OUT/'requests.jsonl')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    prepare() if args.stage=='prepare' else run()
