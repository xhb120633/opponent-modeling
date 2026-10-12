"""C03 matched intervention and independent-match replication, exclusive journals."""
import json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import gpt_chain_elicitation_20260917 as b
import gpt_depth_memory_20260917 as depth
from closure_supplement_20260917 import PACK,ROOT
TASKS=PACK/'gpt_tasks.json'
OUT=ROOT/'closure_gpt_20260917'
FIELDS={'assessment':{'type':'string'},'p_opponent_1':{'type':'number'},'own_action':{'type':'integer','enum':[0,1]}}
SCHEMA={'type':'object','properties':FIELDS,'required':list(FIELDS),'additionalProperties':False}

def prepare():
    manifest=b.read(PACK/'manifest.json');assert all(b.sha(p)==h for p,h in manifest['hashes'].items())
    rows=b.read(PACK/'rows.json');old=[r for r in rows if r['rep']==0 and r['round']>1]
    previous=ROOT/'gpt_depth_retry_merged_20260917/responses.jsonl'
    preds={(r['match_id'],r['round']):r['parsed'] for r in b.rows(previous) if r['condition']=='L2_N' and r['valid']}
    assert len(preds)==60
    tasks=[]
    for r in old:
        donor=next(x for x in old if x['role']==r['role'] and x['round']==r['round'] and x['level']==(r['level']+1)%3)
        for field,target in [('b','belief_true'),('q','p_true')]:
            for arm in ('self','true','shuffled'):
                value=preds[r['match_id'],r['round']][depth.B if field=='b' else depth.Q] if arm=='self' else (r if arm=='true' else donor)[target]
                meaning="B's subjective probability that A chooses1 now" if field=='b' else "the probability that B chooses1 now"
                text=r['prompt']+b.COMMON+f'\nA working estimate of {meaning} is supplied: {value:.17g}. Treat this as the supplied estimate for this decision. No information about its reliability is provided.\n'
                text+='Report assessment (at most three sentences), p_opponent_1 (your forecast of B action1), and own_action (your payoff-maximizing choice).'
                tasks.append(dict(condition=f'{field}_{arm}',match_id=r['match_id'],round=r['round'],messages=[{'role':'user','content':text}],schema=SCHEMA,supplied=value,donor_match_id=donor['match_id'] if arm=='shuffled' else None))
    for r in b.read(PACK/'replication_inputs.json'):
        for arm in ('L1_N','L2_N'):
            tasks.append(dict(condition='rep_'+arm,match_id=r['match_id'],round=r['round'],messages=depth.messages(r,arm,[]),schema=depth.schema(arm)))
    assert len(tasks)==804 and len({b.key(t) for t in tasks})==804
    random.Random(20260917).shuffle(tasks);b.dump(TASKS,tasks)
    paths=[Path(__file__),Path(depth.__file__),Path(b.__file__),TASKS,PACK/'manifest.json',previous,b.WRAPPER]
    b.dump(PACK/'gpt_preflight.json',dict(passed=True,planned=len(tasks),hashes={str(p):b.sha(p) for p in paths},no_new_training=True,sealed_final=False))
    print('804 requests prepared; no API calls',flush=True)

def parse(r,text):
    arm=r['condition']
    if arm.startswith('rep_'):return depth.parse(text,arm[4:])
    x=json.loads(text);assert set(x)==set(FIELDS)
    assert type(x['p_opponent_1']) in (float,int) and np.isfinite(x['p_opponent_1']) and 0<=x['p_opponent_1']<=1
    assert type(x['own_action']) is int and x['own_action'] in (0,1)
    assert isinstance(x['assessment'],str) and x['assessment'].strip()
    return x

def run():
    pre=b.read(PACK/'gpt_preflight.json');assert all(b.sha(p)==h for p,h in pre['hashes'].items())
    tasks=b.read(TASKS);OUT.mkdir()
    sys.path.insert(0,str(b.WRAPPER.parent));from azure_chat_generator import AzureChatGenerator
    client=AzureChatGenerator(deployment=b.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
    b.dump(OUT/'manifest.json',dict(pid=os.getpid(),started=b.now(),model=b.MODEL,effort='medium',max_completion_tokens=32768,planned=len(tasks),concurrency=6,preflight_sha256=b.sha(PACK/'gpt_preflight.json')))
    lock=threading.Lock();halt=threading.Event();count=[0]
    def journal(n,x):
        with (OUT/n).open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def call(t):
        with lock:
            if halt.is_set():return
            count[0]+=1;q=dict(t,number=count[0],started=b.now());journal('requests.jsonl',q)
        r={k:q[k] for k in ('number','condition','match_id','round')};start=time.monotonic()
        try:
            resp=client.chat.completions.create(model=b.DEPLOYMENT,messages=t['messages'],reasoning_effort='medium',max_completion_tokens=32768,response_format={'type':'json_schema','json_schema':{'name':'closure_forecast','strict':True,'schema':t['schema']}})
            r.update(model=resp.model,response_id=resp.id,text=resp.choices[0].message.content,finish_reason=resp.choices[0].finish_reason,usage=resp.usage.model_dump() if resp.usage else None)
            if r['model']!=b.MODEL:halt.set();raise ValueError('model mismatch')
            assert r['finish_reason']=='stop';r.update(parsed=parse(t,r['text']),valid=True)
        except Exception as e:
            status=getattr(e,'status_code',None);r.update(valid=False,error_type=type(e).__name__,status_code=status)
            if status in (400,401,403,404,429):halt.set()
        r.update(completed=b.now(),elapsed_seconds=time.monotonic()-start)
        with lock:journal('responses.jsonl',r);print(json.dumps({k:r.get(k) for k in ('number','condition','valid','error_type','finish_reason')}),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(call,tasks))
    b.dump(OUT/'completed.json',dict(completed=b.now(),halted=halt.is_set(),issued=count[0],hashes={n:b.sha(OUT/n) for n in ('requests.jsonl','responses.jsonl')}))
    audit()

def audit():
    pre=b.read(PACK/'gpt_preflight.json');assert all(b.sha(p)==h for p,h in pre['hashes'].items())
    done=b.read(OUT/'completed.json');assert all(b.sha(OUT/n)==h for n,h in done['hashes'].items())
    tasks={b.key(t):t for t in b.read(TASKS)};targets={(r['match_id'],r['round']):r for r in b.read(PACK/'rows.json')}
    qs=b.rows(OUT/'requests.jsonl');rs=b.rows(OUT/'responses.jsonl');assert len(qs)==len(rs)==done['issued']
    lookup={b.key(r):r for r in rs};assert len(lookup)==len(rs)
    for q in qs:
        t=tasks[b.key(q)];assert all(q[k]==v for k,v in t.items())
        r=lookup[b.key(q)];assert r['number']==q['number']
        if r['valid']:assert parse(t,r['text'])==r['parsed'] and r['model']==b.MODEL and r['finish_reason']=='stop'
    per=[]
    for key,t in tasks.items():
        y=targets[t['match_id'],t['round']];r=lookup.get(key,{});valid=r.get('valid',False);p=r['parsed']['p_opponent_1'] if valid else None
        q=y['p_true'];a=r['parsed']['own_action'] if valid else None
        payoff=(1 if y['role']=='seeker' else -1)*(2*(q if a else 1-q)-1) if valid else None
        per.append(dict(condition=t['condition'],match_id=y['match_id'],round=y['round'],role=y['role'],level=y['level'],valid=valid,p=p,q=q,
            mse=(p-q)**2 if valid else 1,mae=abs(p-q) if valid else None,regret=abs(2*q-1)-payoff if valid else 2,
            belief_error=abs(r['parsed'][depth.B]-y['belief_true']) if valid and depth.B in r['parsed'] else None))
    groups={}
    for arm in sorted({r['condition'] for r in per}):
        arows=[r for r in per if r['condition']==arm and r['round']>1];groups[arm]={}
        for name,rr in [('all',arows)]+[(f'{role}_ToM{l}',[r for r in arows if r['role']==role and r['level']==l]) for role in ('seeker','hider') for l in (0,1,2)]:
            vv=[r for r in rr if r['valid']];groups[arm][name]=dict(n=len(rr),valid=len(vv),penalized_mse=float(np.mean([r['mse'] for r in rr])),penalized_regret=float(np.mean([r['regret'] for r in rr])),valid_mae=float(np.mean([r['mae'] for r in vv])) if vv else None,constant_mse=float(np.mean([(.5-r['q'])**2 for r in rr])))
    contrasts={};rng=np.random.default_rng(20260917)
    pairs=[(f'{f}_true',f'{f}_{c}') for f in ('b','q') for c in ('self','shuffled')]+[('rep_L2_N','rep_L1_N')]
    for a,c in pairs:
        for metric in ('mse','regret'):
            means={}
            for arm in (a,c):
                arr=[r for r in per if r['condition']==arm and r['round']>1];means[arm]={mid:np.mean([r[metric] for r in arr if r['match_id']==mid]) for mid in {r['match_id'] for r in arr}}
            ids=sorted(means[a]);assert ids==sorted(means[c]);delta=np.array([means[a][m]-means[c][m] for m in ids]);boot=[np.mean(rng.choice(delta,len(delta))) for _ in range(3000)]
            contrasts[f'{a}-{c}_{metric}']=dict(delta=float(delta.mean()),ci=np.quantile(boot,[.025,.975]).tolist(),matches=len(ids))
    b.dump(OUT/'scored_rows.json',per);b.dump(OUT/'audit_results.json',dict(passed=True,planned=len(tasks),issued=len(rs),valid=sum(r['valid'] for r in rs),groups=groups,contrasts=contrasts,output_hashes={n:b.sha(OUT/n) for n in ('requests.jsonl','responses.jsonl','scored_rows.json')}))
    print('C03 audit completed',flush=True)

if __name__=='__main__':{'prepare':prepare,'run':run,'audit':audit}[sys.argv[1]]()
