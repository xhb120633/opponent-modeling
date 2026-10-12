"""Unknown opponent, full verbatim self-memory. Targets only in offline audit."""
import argparse,json,os,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import v02_action_breakpoint as v
ROOT=Path(__file__).resolve().parent
PACK=ROOT/'gpt_unknown_memory_package_20260916'
OUT=ROOT/'gpt_unknown_memory_outputs_20260916'
MODEL='gpt-5.4-2026-03-05'
INSTRUCTION='\nMaintain your own evolving model of this opponent using observed play and your earlier forecasts and notes in this conversation. Earlier notes are your hypotheses, not ground truth; revise them when observations warrant. Return p_opponent_1 and state_notes. Choose your own representation in state_notes: concise current opponent-model hypotheses, relevant estimates and uncertainty, not a detailed reasoning trace. No opponent algorithm or hidden state is supplied. Do not assume any specific algorithm. All your earlier outputs remain available.\n'
SCHEMA={'type':'object','properties':{'p_opponent_1':{'type':'number'},'state_notes':{'type':'string'}},'required':['p_opponent_1','state_notes'],'additionalProperties':False}

def parse(text):
    x=json.loads(text);assert set(x)=={'p_opponent_1','state_notes'}
    p=v.valid_probability(json.dumps({'p_opponent_1':x['p_opponent_1']}))
    assert isinstance(x['state_notes'],str) and x['state_notes'].strip()
    return p

def prepare():
    pre=json.loads((v.PACK/'preflight.json').read_text())
    assert all(v.sha(Path(p))==h for p,h in pre['hashes'].items())
    rows=[{k:r[k] for k in ('match_id','round','prompt')} for r in json.loads((v.PACK/'prompts.json').read_text()) if r['condition']=='H']
    assert len(rows)==60 and len({(r['match_id'],r['round']) for r in rows})==60
    assert all('ToM' not in r['prompt'] and 'REFERENCE SOURCE' not in r['prompt'] for r in rows)
    for mid in {r['match_id'] for r in rows}:assert sorted(r['round'] for r in rows if r['match_id']==mid)==list(range(1,11))
    PACK.mkdir();v.dump(PACK/'inputs.json',rows)
    paths=[Path(__file__),ROOT/'v02_action_breakpoint.py',v.PACK/'prompts.json',v.PACK/'targets.json',PACK/'inputs.json',ROOT/'GPT_UNKNOWN_MEMORY_20260916.md']
    v.dump(PACK/'preflight.json',{'passed':True,'hashes':{str(p):v.sha(p) for p in paths}})
    print('PASS60 exact H prompts, six ordered chains, no opponent identity in prompts')

def run():
    pre=json.loads((PACK/'preflight.json').read_text());assert all(v.sha(Path(p))==h for p,h in pre['hashes'].items())
    rows=json.loads((PACK/'inputs.json').read_text());assert not OUT.exists()
    sys.path.insert(0,str(ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm'))
    from azure_chat_generator import AzureChatGenerator
    client=AzureChatGenerator(deployment=v.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
    OUT.mkdir();v.dump(OUT/'manifest.json',{'pid':os.getpid(),'started':v.now(),'model':MODEL,'deployment':v.DEPLOYMENT,'effort':'medium','tokens':32768,'concurrency':3,'preflight_sha256':v.sha(PACK/'preflight.json')})
    lock=threading.Lock();halt=threading.Event();counter=[0]
    def journal(name,obj):
        with (OUT/name).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def chain(mid):
        messages=[];completed=0
        for row in sorted([r for r in rows if r['match_id']==mid],key=lambda x:x['round']):
            messages.append({'role':'user','content':row['prompt']+INSTRUCTION})
            result=None
            for attempt in (1,2):
                with lock:
                    if halt.is_set():break
                    counter[0]+=1;assert counter[0]<=120
                    q={'request_number':counter[0],'match_id':mid,'round':row['round'],'attempt':attempt,'messages':messages,'started':v.now()}
                    journal('requests.jsonl',q)
                result={k:q[k] for k in ('request_number','match_id','round','attempt')}
                try:
                    response=client.chat.completions.create(model=v.DEPLOYMENT,messages=messages,reasoning_effort='medium',max_completion_tokens=32768,response_format={'type':'json_schema','json_schema':{'name':'forecast_memory','strict':True,'schema':SCHEMA}})
                    result.update(model=response.model,response_id=response.id,text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason,usage=response.usage.model_dump() if response.usage else None)
                    if response.model!=MODEL:halt.set();raise ValueError('model mismatch')
                    assert result['finish_reason']=='stop'
                    result.update(p=parse(result['text']),valid=True)
                except Exception as e:
                    status=getattr(e,'status_code',None);result.update(valid=False,error_type=type(e).__name__,status_code=status)
                    if status in (400,401,403,404,429):halt.set()
                result['completed']=v.now()
                with lock:journal('responses.jsonl',result);print(json.dumps({k:result.get(k) for k in ('match_id','round','attempt','valid','finish_reason','error_type')}),flush=True)
                if result.get('error_type') not in ('APIConnectionError','APITimeoutError') or attempt==2:break
                time.sleep(2)
            if not result or not result.get('valid'):break
            messages.append({'role':'assistant','content':result['text']});completed+=1
        with lock:journal('chains.jsonl',{'match_id':mid,'completed_rounds':completed,'complete':completed==10})
    mids=list(dict.fromkeys(r['match_id'] for r in rows))
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(chain,mids))
    v.dump(OUT/'completed.json',{'at':v.now(),'halted':halt.is_set(),'requests':counter[0],'hashes':{n:v.sha(OUT/n) for n in ('requests.jsonl','responses.jsonl','chains.jsonl')}})

def audit():
    pre=json.loads((PACK/'preflight.json').read_text());assert all(v.sha(Path(p))==h for p,h in pre['hashes'].items())
    done=json.loads((OUT/'completed.json').read_text());assert all(v.sha(OUT/n)==h for n,h in done['hashes'].items())
    qs=v.read_rows(OUT/'requests.jsonl');rs=v.read_rows(OUT/'responses.jsonl');by={r['request_number']:r for r in rs}
    assert len(qs)==len(rs)==len(by)<=120
    inputs=json.loads((PACK/'inputs.json').read_text());lookup={(r['match_id'],r['round']):r for r in inputs};latest={};seen={}
    for q in sorted(qs,key=lambda x:x['request_number']):
        mid,t=q['match_id'],q['round'];key=mid,t;r=by[q['request_number']]
        assert all(q[k]==r[k] for k in ('match_id','round','attempt'))
        if key in seen:assert q['attempt']==2 and seen[key]['attempt']==1 and latest[key].get('error_type') in ('APIConnectionError','APITimeoutError')
        else:assert q['attempt']==1
        expected=[]
        for n in range(1,t+1):
            expected.append({'role':'user','content':lookup[(mid,n)]['prompt']+INSTRUCTION})
            if n<t:
                prev=latest[(mid,n)];assert prev.get('valid');expected.append({'role':'assistant','content':prev['text']})
        assert q['messages']==expected
        if r.get('model'):assert r['model']==MODEL
        if r.get('valid'):assert parse(r['text'])==r['p'] and r['finish_reason']=='stop'
        seen[key]=q;latest[key]=r
    ts=json.loads((v.PACK/'targets.json').read_text());groups={}
    for level in (0,1,2):
        for role in ('all','seeker','hider'):
            scheduled=[t for t in ts if t['round']>1 and t['level']==level and (role=='all' or t['role']==role)]
            valid=[t for t in scheduled if latest.get((t['match_id'],t['round']),{}).get('valid')];ps=[latest[(t['match_id'],t['round'])]['p'] for t in valid]
            groups[f'{level}_{role}']={'scheduled':len(scheduled),'valid':len(valid),'model':v.metrics(valid,ps),'constant':v.metrics(valid,[.5]*len(valid)),'frequency':v.metrics(valid,[t['frequency'] for t in valid]),'near_exact':sum(abs(p-t['p_true'])<=.001 for p,t in zip(ps,valid))}
    report={'audit_passed':True,'groups_updates':groups,'chains':v.read_rows(OUT/'chains.jsonl'),'hashes':done['hashes']}
    v.dump(OUT/'audit_results.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','audit']);globals()[p.parse_args().stage]()
