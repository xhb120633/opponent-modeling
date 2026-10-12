"""Explicit13-failure retry; immutable first-attempt evidence."""
import json,sys,os,time,threading,runpy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import closure_gpt_20260917 as c
b=c.b;OLD=c.OUT;NEW=c.ROOT/'closure_retry_20260917';MERGED=c.ROOT/'closure_retry_merged_20260917'
def run():
    pre=b.read(c.PACK/'gpt_preflight.json');assert all(b.sha(p)==h for p,h in pre['hashes'].items())
    done=b.read(OLD/'completed.json');assert all(b.sha(OLD/n)==h for n,h in done['hashes'].items())
    qs=b.rows(OLD/'requests.jsonl');rs=b.rows(OLD/'responses.jsonl');failed={b.key(r):r for r in rs if not r['valid']}
    assert len(rs)==804 and len(failed)==13 and all(r.get('status_code')==500 for r in failed.values())
    tasks=[q for q in qs if b.key(q) in failed];NEW.mkdir()
    paths=[Path(__file__),c.ROOT/'C03_RETRY_20260917.md',*list(OLD.iterdir())]
    b.dump(NEW/'manifest.json',dict(pid=os.getpid(),started=b.now(),planned=13,model=b.MODEL,effort='medium',max_completion_tokens=32768,source_hashes={str(p):b.sha(p) for p in paths if p.is_file()}))
    sys.path.insert(0,str(b.WRAPPER.parent));from azure_chat_generator import AzureChatGenerator
    client=AzureChatGenerator(deployment=b.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
    lock=threading.Lock();halt=threading.Event();nr=[];nq=[]
    def journal(n,x):
        with (NEW/n).open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def call(q):
        with lock:
            if halt.is_set():return
            t=dict(q,started=b.now(),retry_of_number=q['number']);journal('requests.jsonl',t);nq.append(t)
        r={k:q[k] for k in ('number','condition','match_id','round')};r['retry_of_number']=q['number'];start=time.monotonic()
        try:
            response=client.chat.completions.create(model=b.DEPLOYMENT,messages=q['messages'],reasoning_effort='medium',max_completion_tokens=32768,response_format={'type':'json_schema','json_schema':{'name':'closure_forecast','strict':True,'schema':q['schema']}})
            r.update(model=response.model,response_id=response.id,text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason,usage=response.usage.model_dump() if response.usage else None)
            if r['model']!=b.MODEL:halt.set();raise ValueError('model mismatch')
            assert r['finish_reason']=='stop';r.update(valid=True,parsed=c.parse(q,r['text']))
        except Exception as e:
            status=getattr(e,'status_code',None);r.update(valid=False,error_type=type(e).__name__,status_code=status)
            if status in (400,401,403,404,429):halt.set()
        r.update(completed=b.now(),elapsed_seconds=time.monotonic()-start)
        with lock:journal('responses.jsonl',r);nr.append(r);print(json.dumps({k:r.get(k) for k in ('number','condition','valid','error_type')}),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(call,tasks))
    b.dump(NEW/'completed.json',dict(at=b.now(),halted=halt.is_set(),attempted=len(nr),valid=sum(r['valid'] for r in nr),hashes={n:b.sha(NEW/n) for n in ('requests.jsonl','responses.jsonl')}))
    assert len(nr)==13 and not halt.is_set()
    manifest=b.read(NEW/'manifest.json');assert all(b.sha(p)==h for p,h in manifest['source_hashes'].items())
    replacements={b.key(r):r for r in nr};replacementq={b.key(q):q for q in nq}
    merged=[replacements.get(b.key(r),r) for r in rs];mergedq=[replacementq.get(b.key(q),q) for q in qs]
    assert all(x==y for x,y in zip(rs,merged) if x['valid'])
    for q in nq:
        original=next(x for x in qs if b.key(x)==b.key(q));assert original['messages']==q['messages'] and original['schema']==q['schema']
    MERGED.mkdir()
    for name,rows in [('requests.jsonl',mergedq),('responses.jsonl',merged)]:
        with (MERGED/name).open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    b.dump(MERGED/'provenance.json',dict(view='retry-assisted, not first-attempt success',original_valid_preserved=791,retry_keys=list(failed),source_hashes=manifest['source_hashes'],retry_completed_sha256=b.sha(NEW/'completed.json')))
    b.dump(MERGED/'completed.json',dict(at=b.now(),halted=False,issued=804,hashes={n:b.sha(MERGED/n) for n in ('requests.jsonl','responses.jsonl')}))
    c.OUT=MERGED;c.audit();runpy.run_path(str(c.ROOT/'summarize_c03_20260917.py'),run_name='__main__')
if __name__=='__main__':run()
