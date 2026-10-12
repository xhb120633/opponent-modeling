"""Frozen K32 and explicit autonomous R32. Runtime never loads target values."""
import argparse,copy,importlib.metadata,json,math,os,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import v02_action_breakpoint as v
from v01_validation import flat

ROOT=Path(__file__).resolve().parent
PACK=ROOT/'gpt_history_recursive_package_20260916'
OUT=ROOT/'gpt_history_recursive_outputs_20260916'
MODEL='gpt-5.4-2026-03-05';TOKENS=32768

def state_schema(keys):
    return {'type':'object','properties':{'state':{'type':'object','properties':{k:{'type':'number'} for k in keys},'required':keys,'additionalProperties':False},
      'p_opponent_1':{'type':'number'}},'required':['state','p_opponent_1'],'additionalProperties':False}

def parse_state(obj,keys):
    assert isinstance(obj,dict) and set(obj)==set(keys)
    assert all(type(x) in (int,float) and math.isfinite(x) for x in obj.values())
    groups={}
    for k,x in obj.items():
        if '/p_k/' in k:
            assert 0<=x<=1
            groups.setdefault(k.rsplit('/',1)[0],[]).append(x)
    assert all(abs(sum(xs)-1)<=1e-4 for xs in groups.values())
    return obj

def recursive_prompt(m,row,prior,rules):
    text=v.render('K',{'original_prompt':'','level':m['level'],'b_role':m['b_role']},rules)
    old='Start at init_k_tom(params,level). For each completed past pair, call learning_function(state,params,B_action,A_action,level,B_agent_index,p_matrix). '
    text=text.replace(old,'Use the supplied prior state. At round1 it is public initialization and no update is needed. At later rounds apply learning_function exactly ONCE with the latest completed pair: self_choice=B_action, op_choice=A_action. Do not replay earlier history or reset the state. ')
    text=text.replace('Return only the final probability in p_opponent_1; do not return state or explanation.',
      'Return JSON containing the updated full flattened state in state and final action probability in p_opponent_1. No explanation. Omit cached p_op and p_self at every depth.')
    tail='Use the given initialization, history and mechanism to infer the current state and predict B action1 probability.\n'
    assert text.endswith(tail);text=text[:-len(tail)]
    text+=f"\nDecision round={row['round']}.\n"
    text+=('PUBLIC INITIAL STATE:\n' if row['round']==1 else 'YOUR PREVIOUSLY PREDICTED STATE (not oracle):\n')+json.dumps(prior,separators=(',',':'))
    text+='\nLatest completed (A_action,B_action):'+json.dumps(row['observation'])
    text+='\nReturn state with exactly these keys:'+json.dumps(m['keys'],separators=(',',':'))
    return text+'\nReturn updated state and p_opponent_1 only.\n'

def prepare():
    assert not PACK.exists()
    old=json.loads((v.PACK/'preflight.json').read_text())
    assert all(v.sha(Path(p))==h for p,h in old['hashes'].items())
    ks=[r for r in json.loads((v.PACK/'prompts.json').read_text()) if r['condition']=='K']
    targets={(r['match_id'],r['round']):r for r in json.loads((v.PACK/'targets.json').read_text())}
    records=json.loads((ROOT/'g01_inputs.json').read_text())['records'];matches={};truth=[]
    for r in records:
        mid=r['match_id'];t=r['decision_round'];level=int(r['opponent'][-1]);brole=1 if r['role']=='seeker' else 0
        init=flat(v.kt.init_k_tom(v.PARAMS,level));history=v.parse_history(r)
        if mid not in matches:matches[mid]={'match_id':mid,'level':level,'b_role':brole,'role':r['role'],'keys':list(init),'initial':init,'rounds':[]}
        observation=list(history[-1][:2]) if history else None
        matches[mid]['rounds'].append({'round':t,'observation':observation})
        state,belief,p=v.replay(history,level,brole)
        assert abs(p-targets[(mid,t)]['p_true'])<1e-10
        truth.append({**targets[(mid,t)],'state':flat(state)})
    for m in matches.values():
        m['rounds'].sort(key=lambda r:r['round']);assert [r['round'] for r in m['rounds']]==list(range(1,11))
        state=v.kt.init_k_tom(v.PARAMS,m['level'])
        for r in m['rounds']:
            if r['observation'] is not None:
                a,b=r['observation'];state=v.kt.learning_function(state,v.PARAMS,b,a,m['level'],m['b_role'],v.MATRIX)
            target=next(x for x in truth if x['match_id']==m['match_id'] and x['round']==r['round'])
            assert max(abs(flat(state)[k]-target['state'][k]) for k in target['state'])<1e-9
            parse_state(target['state'],m['keys'])
    assert len(ks)==60 and len(matches)==6 and len(truth)==60
    PACK.mkdir();v.dump(PACK/'inputs.json',{'K':ks,'R':list(matches.values())})
    v.dump(PACK/'targets.json',truth)
    paths=[Path(__file__),ROOT/'v02_action_breakpoint.py',ROOT/'v01_validation.py',Path(v.kt.__file__),ROOT/'g01_inputs.json',v.PACK/'preflight.json',v.PACK/'prompts.json',v.PACK/'targets.json',v.PACK/'rules.py.txt',PACK/'inputs.json',PACK/'targets.json',ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm/azure_chat_generator.py']
    v.dump(PACK/'preflight.json',{'passed':True,'logical_max':120,'physical_max':240,'completion_token_max':240*TOKENS,'hashes':{str(p):v.sha(p) for p in paths}})
    print('PREPARED 60K + 6x10R, state reference replay passed')

def run():
    pre=json.loads((PACK/'preflight.json').read_text());assert all(v.sha(Path(p))==h for p,h in pre['hashes'].items())
    data=json.loads((PACK/'inputs.json').read_text());rules=(v.PACK/'rules.py.txt').read_text()
    assert not OUT.exists(),'No implicit resume/duplicate calls'
    sys.path.insert(0,str(ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm'))
    from azure_chat_generator import AzureChatGenerator
    client=AzureChatGenerator(deployment=v.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
    OUT.mkdir();v.dump(OUT/'manifest.json',{'pid':os.getpid(),'started_utc':v.now(),'code_sha256':v.sha(Path(__file__)),'preflight_sha256':v.sha(PACK/'preflight.json'),
      'deployment':v.DEPLOYMENT,'expected_model':MODEL,'reasoning':'medium','max_completion_tokens':TOKENS,'timeout':900,'concurrency':3,'sdk_retries':0,'manual_transport_retries':1,
      'sdk_version':importlib.metadata.version('openai'),'api_version':os.getenv('AZURE_OPENAI_API_VERSION'),'logical_max':120,'physical_max':240})
    lock=threading.Lock();semaphore=threading.Semaphore(3);halt=threading.Event();requests=[];responses=[];chains=[]
    def append(name,obj):
        with (OUT/name).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def call(condition,mid,t,prompt,schema,prior_source=None,prior=None,keys=None):
        for attempt in (1,2):
            with semaphore:
                with lock:
                    if halt.is_set():return None
                    assert len(requests)<240
                    n=len(requests)+1
                    req={'condition':condition,'match_id':mid,'round':t,'attempt':attempt,'request_number':n,'prompt':prompt,'schema':schema,
                      'prior_source':prior_source,'prior':prior,'started_utc':v.now(),'reasoning_effort':'medium','max_completion_tokens':TOKENS}
                    append('requests.jsonl',req);requests.append(req)
                result={k:req[k] for k in ('condition','match_id','round','attempt','request_number')};start=time.monotonic()
                try:
                    res=client.chat.completions.create(model=v.DEPLOYMENT,messages=[{'role':'user','content':prompt}],reasoning_effort='medium',max_completion_tokens=TOKENS,
                      response_format={'type':'json_schema','json_schema':{'name':'tom_prediction','strict':True,'schema':schema}})
                    result.update(model=res.model,response_id=res.id,text=res.choices[0].message.content,finish_reason=res.choices[0].finish_reason,usage=res.usage.model_dump() if res.usage else None)
                    if res.model!=MODEL:halt.set();raise ValueError('Unexpected model identity')
                    assert result['finish_reason']=='stop'
                    if condition=='K':result.update(p=v.valid_probability(result['text']),action_valid=True,valid=True)
                    else:
                        obj=json.loads(result['text']);assert set(obj)=={'state','p_opponent_1'}
                        p=v.valid_probability(json.dumps({'p_opponent_1':obj['p_opponent_1']}))
                        result.update(p=p,action_valid=True)
                        result.update(state=parse_state(obj['state'],keys),state_valid=True,valid=True)
                except Exception as exc:
                    status=getattr(exc,'status_code',None)
                    result.update(valid=False,error_type=type(exc).__name__,status_code=status)
                    if status in (400,401,403,404,429):halt.set()
                result.update(completed_utc=v.now(),elapsed_seconds=time.monotonic()-start)
                with lock:
                    append('responses.jsonl',result);responses.append(result)
                    print(json.dumps({k:result.get(k) for k in ('condition','match_id','round','attempt','valid','error_type','finish_reason')}),flush=True)
            if result.get('error_type') not in ('APIConnectionError','APITimeoutError') or attempt==2:return result
            time.sleep(2)
    def do_k(r):return call('K',r['match_id'],r['round'],r['prompt'],v.SCHEMA)
    def do_r(m):
        prior=copy.deepcopy(m['initial']);source='public_initialization';last=0;reason=None
        for row in m['rounds']:
            result=call('R',m['match_id'],row['round'],recursive_prompt(m,row,prior,rules),state_schema(m['keys']),source,prior,m['keys'])
            last=row['round']
            if not result or not result.get('valid'):
                reason='global_halt' if result is None else result.get('error_type','invalid');break
            prior=result['state'];source={'previous_request_number':result['request_number'],'previous_response_id':result['response_id']}
        chain={'match_id':m['match_id'],'last_attempted_round':last,'completed_all':last==10 and reason is None,'stop_reason':reason}
        with lock:append('chains.jsonl',chain);chains.append(chain)
    first=do_k(data['K'][0]);accepted=bool(first and first.get('model')==MODEL and first.get('finish_reason') in ('stop','length'))
    v.dump(OUT/'smoke.json',{'accepted':accepted,'included_in_K60':True})
    if accepted and not halt.is_set():
        with ThreadPoolExecutor(max_workers=6) as pool:
            # Start recursive chains concurrently with independent history reconstruction.
            futures=[pool.submit(do_r,m) for m in data['R'][:3]]
            futures += [pool.submit(do_k,r) for r in data['K'][1:]]
            futures += [pool.submit(do_r,m) for m in data['R'][3:]]
            for f in futures:f.result()
    else:halt.set()
    v.dump(OUT/'completed.json',{'completed_utc':v.now(),'requests':len(requests),'responses':len(responses),'unique_items':len({v.key(r) for r in requests}),
      'valid_attempts':sum(r.get('valid',False) for r in responses),'halted':halt.is_set(),'chains':chains,'requests_sha256':v.sha(OUT/'requests.jsonl'),'responses_sha256':v.sha(OUT/'responses.jsonl')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args();prepare() if args.stage=='prepare' else run()
