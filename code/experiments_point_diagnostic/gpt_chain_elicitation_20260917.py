"""C01 matched output-structure pilot. API runner never loads target values."""
import argparse, hashlib, json, math, os, random, re, sys, threading, time
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT=Path(__file__).resolve().parent
PACK=ROOT/'gpt_chain_package_20260917'
OUT=ROOT/'gpt_chain_outputs_20260917'
SOURCE=ROOT/'v02_package_20260915'
MODEL='gpt-5.4-2026-03-05'
DEPLOYMENT=os.environ['TOM_AZURE_DEPLOYMENT']
WRAPPER=ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm/azure_chat_generator.py'
PROTO=ROOT/'GPT_CHAIN_ELICITATION_20260917.md'
FIELDS={
 'D':{'p_opponent_1':{'type':'number'},'own_action':{'type':'integer','enum':[0,1]}},
 'F':{'assessment':{'type':'string'},'p_opponent_1':{'type':'number'},'own_action':{'type':'integer','enum':[0,1]}},
 'C':{'opponent_belief_about_my_action_1':{'type':'number'},'belief_to_action_assessment':{'type':'string'},'p_opponent_1':{'type':'number'},'own_action':{'type':'integer','enum':[0,1]}}
}
COMMON='\nYou are A; the opponent is B. No identity, algorithm or private state of B is known. Do not assume a specific algorithm. Also report own_action, the action (0 or 1) you would choose to maximize your expected payoff this round. Your answer will not change the recorded history. Return the requested JSON fields only.\n'
INSTRUCTIONS={
 'D':'Report p_opponent_1 (your forecast of B choosing 1), followed by own_action. No explanation is requested.',
 'F':'First report assessment: a brief, freely organized assessment of the observed play and your forecast (at most three sentences, not a detailed reasoning trace). Then report p_opponent_1, followed by own_action.',
 'C':"Make the opponent-modeling chain explicit. First report opponent_belief_about_my_action_1: your estimate of B's subjective probability that A will choose 1 now (not your own intended action probability). Next report belief_to_action_assessment: a brief account of how that estimated belief and B's incentives or observed tendencies inform your forecast (at most three sentences, not a detailed reasoning trace). Then report p_opponent_1: your forecast of B choosing 1. Finally report own_action. Keep A's and B's perspectives distinct; your estimate of B's belief is uncertain, not a supplied fact."
}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()] if Path(p).exists() else []
def key(r):return (r['condition'],r['match_id'],r['round'])
def render(r,c):return r['prompt']+COMMON+INSTRUCTIONS[c]
def schema(c):return {'type':'object','properties':FIELDS[c],'required':list(FIELDS[c]),'additionalProperties':False}
def parse(text,c):
 x=json.loads(text);assert set(x)==set(FIELDS[c])
 for k,s in FIELDS[c].items():
  if s['type']=='number':assert type(x[k]) in (int,float) and math.isfinite(x[k]) and 0<=x[k]<=1
  elif s['type']=='integer':assert type(x[k]) is int and x[k] in (0,1)
  else:assert isinstance(x[k],str) and x[k].strip()
 return x
def prepare():
 pre=read(SOURCE/'preflight.json')
 assert all(sha(p)==h for p,h in pre['hashes'].items()),'frozen V02 provenance changed'
 src=[{k:r[k] for k in ('match_id','round','prompt')} for r in read(SOURCE/'prompts.json') if r['condition']=='H' and r['round']>1]
 ts={(r['match_id'],r['round']):r for r in read(SOURCE/'targets.json')}
 g1={(r['match_id'],r['decision_round']):r for r in read(ROOT/'g01_inputs.json')['records']}
 g2={(r['match_id'],r['decision_round']):r for r in read(ROOT/'g02_inputs.json')['records']}
 assert len(src)==54 and len({(r['match_id'],r['round']) for r in src})==54
 local=[]
 for r in src:
  k=(r['match_id'],r['round']);t=dict(ts[k]);assert r['prompt']==g1[k]['prompt']
  assert abs(t['p_true']-g1[k]['oracle'])<1e-10
  assert abs(t['belief_true']-g2[k]['belief_target'])<1e-10
  past=re.findall(r'round=(\d+); self_action=([01]); opponent_action=([01]); self_payoff=(-?1)',r['prompt'])
  assert [int(p[0]) for p in past]==list(range(1,r['round']))
  t['belief_frequency']=(1+sum(int(p[1]) for p in past))/(2+len(past));local.append(t)
  assert all(s not in r['prompt'] for s in ('ToM','REFERENCE SOURCE','CURRENT STATE',r['match_id']))
  changed={**r,'belief_true':.12345,'p_true':.98765,'level':99}
  for c in FIELDS:assert render(changed,c)==render(r,c)
 for c in FIELDS:
  mock={k:(.5 if v['type']=='number' else 0 if v['type']=='integer' else 'brief assessment') for k,v in FIELDS[c].items()}
  assert parse(json.dumps(mock),c)==mock
  for val in (-.1,1.1,True):
   bad={**mock,'p_opponent_1':val}
   try:parse(json.dumps(bad),c)
   except AssertionError:pass
   else:raise AssertionError('invalid probability accepted')
 PACK.mkdir();dump(PACK/'inputs.json',src);dump(PACK/'targets.json',local)
 paths=[Path(__file__),PROTO,WRAPPER,SOURCE/'preflight.json',SOURCE/'prompts.json',SOURCE/'targets.json',ROOT/'g01_inputs.json',ROOT/'g02_inputs.json',PACK/'inputs.json',PACK/'targets.json']
 dump(PACK/'preflight.json',{'passed':True,'points':54,'conditions':3,'hashes':{str(p):sha(p) for p in paths},'checks':['source hashes','q and b perspective/target match','history lengths','no identity in prompt','renderer ignores targets','parser positive/negative']})
 print('Prepared 162 paired items; no API calls.',flush=True)
def run():
 pre=read(PACK/'preflight.json');assert all(sha(p)==h for p,h in pre['hashes'].items())
 src=read(PACK/'inputs.json');items=[{**r,'condition':c} for r in src for c in FIELDS];random.Random(20260917).shuffle(items)
 sys.path.insert(0,str(WRAPPER.parent));from azure_chat_generator import AzureChatGenerator
 client=AzureChatGenerator(deployment=DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
 OUT.mkdir();dump(OUT/'manifest.json',{'pid':os.getpid(),'started':now(),'model':MODEL,'effort':'medium','max_completion_tokens':32768,'concurrency':6,'planned':len(items),'preflight_hash':sha(PACK/'preflight.json')})
 lock=threading.Lock();halt=threading.Event();counter=[0]
 def journal(name,x):
  with (OUT/name).open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def call(r):
  with lock:
   if halt.is_set():return
   counter[0]+=1;n=counter[0]
   q={k:r[k] for k in ('condition','match_id','round')};q.update(number=n,started=now(),messages=[{'role':'user','content':render(r,r['condition'])}],response_format={'type':'json_schema','json_schema':{'name':'opponent_forecast','strict':True,'schema':schema(r['condition'])}});journal('requests.jsonl',q)
  result={k:q[k] for k in ('condition','match_id','round','number')};start=time.monotonic()
  try:
   response=client.chat.completions.create(model=DEPLOYMENT,messages=q['messages'],reasoning_effort='medium',max_completion_tokens=32768,response_format=q['response_format'])
   result.update(model=response.model,response_id=response.id,text=response.choices[0].message.content,finish_reason=response.choices[0].finish_reason,usage=response.usage.model_dump() if response.usage else None)
   if response.model!=MODEL:halt.set();raise ValueError('model identity mismatch')
   assert result['finish_reason']=='stop'
   result.update(parsed=parse(result['text'],r['condition']),valid=True)
  except Exception as e:
   status=getattr(e,'status_code',None);result.update(valid=False,error_type=type(e).__name__,status_code=status)
   if status in (400,401,403,404,429):halt.set()
  result.update(completed=now(),elapsed_seconds=time.monotonic()-start)
  with lock:journal('responses.jsonl',result);print(json.dumps({k:result.get(k) for k in ('number','condition','round','valid','finish_reason','error_type')}),flush=True)
 with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(call,items))
 dump(OUT/'completed.json',{'at':now(),'halted':halt.is_set(),'requests':counter[0],'hashes':{n:sha(OUT/n) for n in ('requests.jsonl','responses.jsonl')}})
 audit()
def corr(a,b):
 if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
 return float(np.corrcoef(a,b)[0,1])
def metric(ts,ps,field='p_true'):
 if not ts:return {'n':0}
 y=np.array([t[field] for t in ts]);p=np.array(ps);ac=[];bc=[];ad=[];bd=[]
 for mid in sorted({t['match_id'] for t in ts}):
  ix=sorted([i for i,t in enumerate(ts) if t['match_id']==mid],key=lambda i:ts[i]['round']);a=p[ix];b=y[ix];ac.extend(a-a.mean());bc.extend(b-b.mean())
  for j in range(1,len(ix)):
   if ts[ix[j]]['round']==ts[ix[j-1]]['round']+1:ad.append(a[j]-a[j-1]);bd.append(b[j]-b[j-1])
 return {'n':len(ts),'mae':float(np.abs(p-y).mean()),'mse':float(((p-y)**2).mean()),'pearson':corr(p,y),'centered_r':corr(ac,bc),'change_r':corr(ad,bd)}
def audit():
 pre=read(PACK/'preflight.json');assert all(sha(p)==h for p,h in pre['hashes'].items())
 done=read(OUT/'completed.json');assert all(sha(OUT/n)==h for n,h in done['hashes'].items())
 qs=rows(OUT/'requests.jsonl');rs=rows(OUT/'responses.jsonl');assert len(qs)==len(rs)==done['requests']
 src={(r['match_id'],r['round']):r for r in read(PACK/'inputs.json')};responses={key(r):r for r in rs};assert len(responses)==len(rs)
 bynum={r['number']:r for r in rs};assert len(bynum)==len(rs)
 for q in qs:
  assert q['messages']==[{'role':'user','content':render(src[q['match_id'],q['round']],q['condition'])}]
  assert q['response_format']['json_schema']['schema']==schema(q['condition'])
  r=bynum[q['number']];assert key(q)==key(r)
  if r.get('model'):assert r['model']==MODEL
  if r['valid']:assert parse(r['text'],r['condition'])==r['parsed'] and r['finish_reason']=='stop'
 ts=read(PACK/'targets.json');groups={};matched={}
 def response(c,t):return responses.get((c,t['match_id'],t['round']),{})
 selectors={'all':ts,**{f'ToM{l}':[t for t in ts if t['level']==l] for l in (0,1,2)},**{f'{role}_ToM{l}':[t for t in ts if t['level']==l and t['role']==role] for role in ('seeker','hider') for l in (0,1,2)}}
 for name,subset in selectors.items():
  g={'scheduled_per_condition':len(subset),'constant':metric(subset,[.5]*len(subset)),'frequency':metric(subset,[t['frequency'] for t in subset])}
  for c in FIELDS:
   valid=[t for t in subset if response(c,t).get('valid')];ps=[response(c,t)['parsed']['p_opponent_1'] for t in valid];m=metric(valid,ps)
   m['penalized_mse']=float(np.mean([(response(c,t)['parsed']['p_opponent_1']-t['p_true'])**2 if response(c,t).get('valid') else 1 for t in subset]))
   m['valid_constant']=metric(valid,[.5]*len(valid));m['valid_frequency']=metric(valid,[t['frequency'] for t in valid])
   if valid:
    acts=[response(c,t)['parsed']['own_action'] for t in valid]
    m['best_response_consistency']=float(np.mean([p==.5 or a==int((p>.5)==(t['role']=='seeker')) for t,p,a in zip(valid,ps,acts)]))
    m['oracle_expected_one_step_payoff']=float(np.mean([(1 if t['role']=='seeker' else -1)*(2*(t['p_true'] if a else 1-t['p_true'])-1) for t,a in zip(valid,acts)]))
    m['sampled_action_brier']=float(np.mean([(p-t['y'])**2 for t,p in zip(valid,ps)]))
   if c=='C':
    m['belief']=metric(valid,[response(c,t)['parsed']['opponent_belief_about_my_action_1'] for t in valid],'belief_true');m['belief_constant']=metric(valid,[.5]*len(valid),'belief_true');m['belief_frequency']=metric(valid,[t['belief_frequency'] for t in valid],'belief_true')
   g[c]=m
  groups[name]=g
 for comparator in ('D','F'):
  common=[t for t in ts if all(response(c,t).get('valid') for c in ('C',comparator))];ds={}
  for t in common:ds.setdefault(t['match_id'],[]).append(abs(response('C',t)['parsed']['p_opponent_1']-t['p_true'])-abs(response(comparator,t)['parsed']['p_opponent_1']-t['p_true']))
  vals=[float(np.mean(x)) for x in ds.values()];rng=np.random.default_rng(20260917)
  matched['C_minus_'+comparator]={'paired_points':len(common),'matches':len(vals),'mean_point_mae_delta':float(np.mean([d for a in ds.values() for d in a])) if vals else None,'equal_match_delta':float(np.mean(vals)) if vals else None,'exploratory_match_bootstrap_95':np.quantile(np.mean(rng.choice(vals,(5000,len(vals))),axis=1),[.025,.975]).tolist() if vals else None}
 usage={c:{'returned':sum(r['condition']==c for r in rs),'valid':sum(r['condition']==c and r['valid'] for r in rs),'length':sum(r['condition']==c and r.get('finish_reason')=='length' for r in rs),'completion_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in rs if r['condition']==c),'prompt_tokens':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in rs if r['condition']==c)} for c in FIELDS}
 dump(OUT/'audit_results.json',{'audit_passed':True,'execution_complete':len(rs)==162,'groups':groups,'contrasts':matched,'usage':usage,'hashes':done['hashes']})
 print(json.dumps({'audit_passed':True,'usage':usage,'contrasts':matched}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','audit']);globals()[p.parse_args().stage]()
