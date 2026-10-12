"""Explicit C02-R supplement; never overwrite the original single-attempt result."""
import json,sys,os,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gpt_depth_memory_20260917 as s
b=s.base
OLD=s.ROOT/'gpt_depth_memory_merged_20260917'
NEW=s.ROOT/'gpt_depth_retry_20260917'
MERGED=s.ROOT/'gpt_depth_retry_merged_20260917'
PROTO=s.ROOT/'GPT_DEPTH_RETRY_20260917.md'
def run():
 pre=b.read(s.PACK/'preflight.json');assert all(b.sha(p)==h for p,h in pre['hashes'].items())
 done=b.read(OLD/'completed.json');assert all(b.sha(OLD/n)==h for n,h in done['hashes'].items())
 assert b.read(OLD/'audit_results.json')['audit_passed']
 oq=b.rows(OLD/'requests.jsonl');rr=b.rows(OLD/'responses.jsonl');old={b.key(r):r for r in rr};qkey={b.key(q):q for q in oq}
 failures={k:r for k,r in old.items() if not r['valid']};assert len(rr)==346 and len(failures)==7 and all(r['finish_reason']=='length' for r in failures.values())
 inputs={(r['match_id'],r['round']):r for r in b.read(s.PACK/'inputs.json')};tasks=b.read(s.ROOT/'gpt_depth_memory_outputs_20260917/manifest.json')['tasks']
 pending=[(a,m,t) for a,m in tasks for t in range(1,11) if (a,m,t) not in old or not old[a,m,t]['valid']];assert len(pending)==21
 NEW.mkdir();sources=[Path(__file__),PROTO,Path(s.__file__),Path(b.__file__),*list(OLD.iterdir())]
 b.dump(NEW/'manifest.json',{'started':b.now(),'pid':os.getpid(),'scope':'one unchanged-config attempt per7 failure keys, then14 own-memory descendants if possible','pending':pending,'config':{'model':b.MODEL,'effort':'medium','max_completion_tokens':32768,'concurrency':6},'source_hashes':{str(p):b.sha(p) for p in sources if p.is_file()}})
 sys.path.insert(0,str(b.WRAPPER.parent));from azure_chat_generator import AzureChatGenerator
 client=AzureChatGenerator(deployment=b.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
 lock=threading.Lock();halt=threading.Event();counter=[max(q['number'] for q in oq)];newrs=[];newqs=[]
 def journal(n,x):
  with (NEW/n).open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def chain(task):
  arm,mid=task;past=[]
  for t in range(1,11):
   k=arm,mid,t;row=inputs[mid,t];r=old.get(k)
   if r and r['valid']:
    if arm.endswith('M'):past=s.messages(row,arm,past)+[{'role':'assistant','content':r['text']}]
    continue
   with lock:
    if halt.is_set():break
    counter[0]+=1;q={'number':counter[0],'condition':arm,'match_id':mid,'round':t,'started':b.now(),'retry_of_number':r['number'] if r else None,'messages':s.messages(row,arm,past),'response_format':{'type':'json_schema','json_schema':{'name':'depth_forecast','strict':True,'schema':s.schema(arm)}},'api_parameters':{'model':b.DEPLOYMENT,'reasoning_effort':'medium','max_completion_tokens':32768}}
    if r:assert q['messages']==qkey[k]['messages'] and q['response_format']==qkey[k]['response_format']
    journal('requests.jsonl',q);newqs.append(q)
   resrow={x:q[x] for x in ('number','condition','match_id','round','retry_of_number')};start=time.monotonic()
   try:
    response=client.chat.completions.create(**q['api_parameters'],messages=q['messages'],response_format=q['response_format'])
    resrow.update(model=response.model,response_id=response.id,text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason,usage=response.usage.model_dump() if response.usage else None)
    if response.model!=b.MODEL:halt.set();raise ValueError('model mismatch')
    assert resrow['finish_reason']=='stop';resrow.update(valid=True,parsed=s.parse(resrow['text'],arm))
   except Exception as e:
    status=getattr(e,'status_code',None);resrow.update(valid=False,error_type=type(e).__name__,status_code=status)
    if status in (400,401,403,404,429):halt.set()
   resrow.update(completed=b.now(),elapsed_seconds=time.monotonic()-start)
   with lock:journal('responses.jsonl',resrow);newrs.append(resrow);print(json.dumps({x:resrow.get(x) for x in ('number','condition','round','retry_of_number','valid','finish_reason','error_type')}),flush=True)
   if arm.endswith('M'):
    if not resrow['valid']:break
    past=q['messages']+[{'role':'assistant','content':resrow['text']}]
 with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(chain,tasks))
 b.dump(NEW/'completed.json',{'at':b.now(),'halted':halt.is_set(),'attempted':len(newrs),'valid':sum(r['valid'] for r in newrs),'hashes':{n:b.sha(NEW/n) for n in ('requests.jsonl','responses.jsonl')}})
 assert not halt.is_set(),'halted; do not publish complete scoring view'
 manifest=b.read(NEW/'manifest.json');assert all(b.sha(p)==h for p,h in manifest['source_hashes'].items())
 replacements={b.key(r):r for r in newrs};allr=[r for r in rr if b.key(r) not in replacements]+newrs;allq=[q for q in oq if b.key(q) not in replacements]+newqs;lookup={b.key(r):r for r in allr}
 assert len(lookup)==len(allr);assert all(lookup[k]==r for k,r in old.items() if r['valid'])
 chains=[];missing=[]
 for arm,mid in tasks:
  records=[r for r in allr if r['condition']==arm and r['match_id']==mid];failed=False
  for t in range(1,11):
   r=lookup.get((arm,mid,t))
   if not r:assert arm.endswith('M') and failed;missing.append((arm,mid,t))
   elif not r['valid']:failed=True
  chains.append({'condition':arm,'match_id':mid,'issued':len(records),'valid':sum(r['valid'] for r in records),'complete':sum(r['valid'] for r in records)==10})
 MERGED.mkdir()
 for n,rs in [('requests.jsonl',allq),('responses.jsonl',allr),('chains.jsonl',chains)]:
  with (MERGED/n).open('x',encoding='utf-8') as f:
   for r in rs:f.write(json.dumps(r,allow_nan=False)+'\n')
 b.dump(MERGED/'provenance.json',{'source_manifest':b.sha(NEW/'manifest.json'),'retry_completion':b.sha(NEW/'completed.json'),'valid_original_preserved':339,'original_failures_retained_at':str(OLD),'replacement_links':[{x:r[x] for x in ('number','condition','match_id','round','retry_of_number')} for r in newrs],'still_blocked':missing,'interpretation':'Retry-assisted selected-failure supplement, not original single-attempt completion. Original scores/failures stay authoritative for single-attempt comparisons.'})
 b.dump(MERGED/'completed.json',{'at':b.now(),'halted':False,'requests':len(allr),'hashes':{n:b.sha(MERGED/n) for n in ('requests.jsonl','responses.jsonl','chains.jsonl')}})
 s.OUT=MERGED;s.audit()
if __name__=='__main__':run()
