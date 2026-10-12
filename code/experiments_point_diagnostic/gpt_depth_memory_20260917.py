"""C02 six-arm depth/memory pilot; no target values in the inference path."""
import argparse, json, random, re, sys, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import gpt_chain_elicitation_20260917 as base
ROOT=Path(__file__).resolve().parent
PACK=ROOT/'gpt_depth_memory_package_20260917'
OUT=ROOT/'gpt_depth_memory_outputs_20260917'
PROTO=ROOT/'GPT_DEPTH_MEMORY_20260917.md'
ARMS=tuple(f'L{d}_{m}' for d in (1,2,3) for m in ('N','M'))
Q='p_opponent_1';B='opponent_belief_about_my_action_1';C='opponent_estimate_of_my_prediction_of_opponent_action_1'
def fields(arm):
 d=int(arm[1]);p={}
 if d==3:p[C]={'type':'number'}
 if d>=2:p[B]={'type':'number'}
 p.update(assessment={'type':'string'},p_opponent_1={'type':'number'},own_action={'type':'integer','enum':[0,1]});return p
def schema(arm):
 p=fields(arm);return {'type':'object','properties':p,'required':list(p),'additionalProperties':False}
def parse(text,arm):
 x=json.loads(text);assert set(x)==set(fields(arm))
 for k,s in fields(arm).items():
  if s['type']=='number':assert type(x[k]) in (int,float) and np.isfinite(x[k]) and 0<=x[k]<=1
  elif s['type']=='integer':assert type(x[k]) is int and x[k] in (0,1)
  else:assert isinstance(x[k],str) and x[k].strip()
 return x
def render(row,arm):
 d=int(arm[1]);s=row['prompt']+base.COMMON
 s+='Any previously proposed own_action is only a recommendation, not an observed action. Use the recorded past observations as the actual outcomes. Do not treat your forecasts as facts.\n'
 if arm.endswith('M'):s+='All your earlier outputs are retained below in the conversation. Reassess your prior forecasts and belief estimates in light of the observed outcomes; you may revise them.\n'
 else:s+='No earlier model forecasts or reports are supplied in this independent request.\n'
 if d==3:s+=f"First report {C}: your estimate of the probability B believes A assigns to B choosing 1 now. In words: what does B think A predicts about B's next action? This is a nested belief, not your own direct forecast of B and not B's belief about A's action.\n"
 if d>=2:s+=f"{'Next' if d==3 else 'First'} report {B}: your estimate of B's subjective probability that A chooses 1 now. This is B's prediction of A, not A's intended action probability.\n"
 s+='Report assessment: a brief assessment connecting '+('these estimated beliefs, ' if d>=2 else '')+'observed play, incentives and your forecast (at most three sentences, not a detailed reasoning trace). Then report p_opponent_1: your own forecast of B choosing 1 now. Finally report own_action. Keep perspectives distinct; all estimates are uncertain.\n'
 return s
def messages(row,arm,past):return (list(past) if arm.endswith('M') else [])+[{'role':'user','content':render(row,arm)}]
def prepare():
 old=base.read(base.SOURCE/'preflight.json');assert all(base.sha(p)==h for p,h in old['hashes'].items())
 inputs=[{k:r[k] for k in ('match_id','round','prompt')} for r in base.read(base.SOURCE/'prompts.json') if r['condition']=='H']
 assert len(inputs)==60 and len({(r['match_id'],r['round']) for r in inputs})==60
 g1={(r['match_id'],r['decision_round']):r for r in base.read(ROOT/'g01_inputs.json')['records']};g2={(r['match_id'],r['decision_round']):r for r in base.read(ROOT/'g02_inputs.json')['records']}
 targets=base.read(base.SOURCE/'targets.json');by={(r['match_id'],r['round']):r for r in targets}
 for r in inputs:
  k=r['match_id'],r['round'];t=by[k];assert r['prompt']==g1[k]['prompt'] and abs(t['p_true']-g1[k]['oracle'])<1e-10 and abs(t['belief_true']-g2[k]['belief_target'])<1e-10
  hist=re.findall(r'round=(\d+); self_action=([01]); opponent_action=([01]); self_payoff=(-?1)',r['prompt']);assert [int(x[0]) for x in hist]==list(range(1,r['round']))
  t['belief_frequency']=(1+sum(int(x[1]) for x in hist))/(2+len(hist))
  assert all(x not in r['prompt'] for x in ('ToM','REFERENCE SOURCE','CURRENT STATE',r['match_id']))
  for arm in ARMS:assert render(r,arm)==render({**r,'p_true':.99,'belief_true':.01,'level':99},arm)
 for arm in ARMS:
  mock={k:(.5 if v['type']=='number' else 0 if v['type']=='integer' else 'brief') for k,v in fields(arm).items()};raw=json.dumps(mock);assert parse(raw,arm)==mock
  for value in (-.1,1.1,True):
   try:parse(json.dumps({**mock,Q:value}),arm)
   except AssertionError:pass
   else:raise AssertionError('bad parser')
  prior=messages(inputs[0],arm,[])+[{'role':'assistant','content':raw}]
  nxt=messages(inputs[1],arm,prior);assert len(nxt)==(3 if arm.endswith('M') else 1)
  if arm.endswith('M'):assert nxt[1]['content']==raw
 PACK.mkdir();base.dump(PACK/'inputs.json',inputs);base.dump(PACK/'targets.json',targets)
 paths=[Path(__file__),Path(base.__file__),PROTO,base.WRAPPER,base.SOURCE/'preflight.json',base.SOURCE/'prompts.json',base.SOURCE/'targets.json',ROOT/'g01_inputs.json',ROOT/'g02_inputs.json',PACK/'inputs.json',PACK/'targets.json']
 base.dump(PACK/'preflight.json',{'passed':True,'planned':360,'hashes':{str(p):base.sha(p) for p in paths},'checks':['source hash','q/b target perspective','complete past prefixes','no identity','target mutation','parser','memory mock']})
 print('PASS six arms x six matches x ten rounds; no API calls',flush=True)
def run():
 pre=base.read(PACK/'preflight.json');assert all(base.sha(p)==h for p,h in pre['hashes'].items())
 inputs=base.read(PACK/'inputs.json');mids=list(dict.fromkeys(r['match_id'] for r in inputs));rng=random.Random(20260917);rng.shuffle(mids)
 tasks=[]
 for mid in mids:
  arms=list(ARMS);rng.shuffle(arms);tasks.extend((a,mid) for a in arms)
 sys.path.insert(0,str(base.WRAPPER.parent));from azure_chat_generator import AzureChatGenerator
 client=AzureChatGenerator(deployment=base.DEPLOYMENT,env_file=os.environ.get("TOM_AZURE_ENV_FILE")).client.with_options(max_retries=0,timeout=900)
 OUT.mkdir();base.dump(OUT/'manifest.json',{'started':base.now(),'pid':os.getpid(),'model':base.MODEL,'effort':'medium','max_completion_tokens':32768,'concurrency':6,'planned':360,'preflight_sha256':base.sha(PACK/'preflight.json'),'tasks':tasks})
 lock=threading.Lock();halt=threading.Event();counter=[0]
 def journal(n,x):
  with (OUT/n).open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def chain(task):
  arm,mid=task;past=[];valid_count=0;issued=0
  for row in sorted([r for r in inputs if r['match_id']==mid],key=lambda r:r['round']):
   with lock:
    if halt.is_set():break
    counter[0]+=1;issued+=1;q={'number':counter[0],'condition':arm,'match_id':mid,'round':row['round'],'messages':messages(row,arm,past),'response_format':{'type':'json_schema','json_schema':{'name':'depth_forecast','strict':True,'schema':schema(arm)}},'started':base.now()};journal('requests.jsonl',q)
   result={k:q[k] for k in ('number','condition','match_id','round')};start=time.monotonic()
   try:
    res=client.chat.completions.create(model=base.DEPLOYMENT,messages=q['messages'],reasoning_effort='medium',max_completion_tokens=32768,response_format=q['response_format'])
    result.update(model=res.model,response_id=res.id,text=res.choices[0].message.content,finish_reason=res.choices[0].finish_reason,usage=res.usage.model_dump() if res.usage else None)
    if res.model!=base.MODEL:halt.set();raise ValueError('model mismatch')
    assert result['finish_reason']=='stop';result.update(valid=True,parsed=parse(result['text'],arm))
   except Exception as e:
    status=getattr(e,'status_code',None);result.update(valid=False,error_type=type(e).__name__,status_code=status)
    if status in (400,401,403,404,429):halt.set()
   result.update(completed=base.now(),elapsed_seconds=time.monotonic()-start)
   with lock:journal('responses.jsonl',result);print(json.dumps({k:result.get(k) for k in ('number','condition','round','valid','error_type','finish_reason')}),flush=True)
   if result['valid']:
    valid_count+=1;past=q['messages']+[{'role':'assistant','content':result['text']}]
   elif arm.endswith('M'):break
  with lock:journal('chains.jsonl',{'condition':arm,'match_id':mid,'issued':issued,'valid':valid_count,'complete':valid_count==10})
 with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(chain,tasks))
 base.dump(OUT/'completed.json',{'at':base.now(),'halted':halt.is_set(),'requests':counter[0],'hashes':{n:base.sha(OUT/n) for n in ('requests.jsonl','responses.jsonl','chains.jsonl')}})
 audit()
def audit():
 pre=base.read(PACK/'preflight.json');assert all(base.sha(p)==h for p,h in pre['hashes'].items())
 done=base.read(OUT/'completed.json');assert all(base.sha(OUT/n)==h for n,h in done['hashes'].items())
 inputs={(r['match_id'],r['round']):r for r in base.read(PACK/'inputs.json')};qs=base.rows(OUT/'requests.jsonl');rs=base.rows(OUT/'responses.jsonl');chains=base.rows(OUT/'chains.jsonl')
 assert len(qs)==len(rs)==done['requests'];by={r['number']:r for r in rs};lookup={base.key(r):r for r in rs};assert len(by)==len(lookup)==len(rs)
 for q in qs:
  arm,mid,t=base.key(q);assert arm in ARMS and (mid,t) in inputs;r=by[q['number']];assert base.key(q)==base.key(r)
  expected=[]
  if arm.endswith('M'):
   for n in range(1,t):
    prev=lookup[arm,mid,n];assert prev['valid'];expected.extend([{'role':'user','content':render(inputs[mid,n],arm)},{'role':'assistant','content':prev['text']}])
  expected.append({'role':'user','content':render(inputs[mid,t],arm)});assert q['messages']==expected and q['response_format']['json_schema']['schema']==schema(arm)
  if r.get('model'):assert r['model']==base.MODEL
  if r['valid']:assert parse(r['text'],arm)==r['parsed'] and r['finish_reason']=='stop'
 assert len(chains)==36 and len({(c['condition'],c['match_id']) for c in chains})==36
 for c in chains:
  rr=[r for r in rs if r['condition']==c['condition'] and r['match_id']==c['match_id']];assert c['issued']==len(rr) and c['valid']==sum(r['valid'] for r in rr) and c['complete']==(c['valid']==10)
 targets=[t for t in base.read(PACK/'targets.json') if t['round']>1]
 def response(arm,t):return lookup.get((arm,t['match_id'],t['round']),{})
 groups={};sels={'all':targets,**{f'ToM{l}':[t for t in targets if t['level']==l] for l in (0,1,2)},**{f'{r}_ToM{l}':[t for t in targets if t['role']==r and t['level']==l] for r in ('seeker','hider') for l in (0,1,2)},**{f'round{n}':[t for t in targets if t['round']==n] for n in range(2,11)}}
 for name,ts in sels.items():
  g={'scheduled':len(ts),'constant':base.metric(ts,[.5]*len(ts)),'frequency':base.metric(ts,[t['frequency'] for t in ts])}
  for arm in ARMS:
   valid=[t for t in ts if response(arm,t).get('valid')];ps=[response(arm,t)['parsed'][Q] for t in valid];m=base.metric(valid,ps)
   m['penalized_mse']=float(np.mean([(response(arm,t)['parsed'][Q]-t['p_true'])**2 if response(arm,t).get('valid') else 1 for t in ts]))
   m['valid_constant']=base.metric(valid,[.5]*len(valid));m['valid_frequency']=base.metric(valid,[t['frequency'] for t in valid])
   if valid:
    acts=[response(arm,t)['parsed']['own_action'] for t in valid]
    m['brier']=float(np.mean([(p-t['y'])**2 for p,t in zip(ps,valid)]));m['own_action_consistency']=float(np.mean([p==.5 or a==int((p>.5)==(t['role']=='seeker')) for p,a,t in zip(ps,acts,valid)]));m['expected_one_step_payoff']=float(np.mean([(1 if t['role']=='seeker' else -1)*(2*(t['p_true'] if a else 1-t['p_true'])-1) for t,a in zip(valid,acts)]))
   if int(arm[1])>=2:m.update(belief=base.metric(valid,[response(arm,t)['parsed'][B] for t in valid],'belief_true'),belief_constant=base.metric(valid,[.5]*len(valid),'belief_true'),belief_frequency=base.metric(valid,[t['belief_frequency'] for t in valid],'belief_true'))
   g[arm]=m
  groups[name]=g
 contrasts={}
 def contrast(weights):
  common=[t for t in targets if all(response(arm,t).get('valid') for arm in weights)];ds={}
  for t in common:ds.setdefault(t['match_id'],[]).append(sum(w*abs(response(arm,t)['parsed'][Q]-t['p_true']) for arm,w in weights.items()))
  vals=[float(np.mean(x)) for x in ds.values()];rng=np.random.default_rng(20260917)
  return {'paired_points':len(common),'matches':len(vals),'mean_mae_delta':float(np.mean([x for v in ds.values() for x in v])) if vals else None,'equal_match_delta':float(np.mean(vals)) if vals else None,'exploratory_match_bootstrap_95':np.quantile(np.mean(rng.choice(vals,(5000,len(vals))),axis=1),[.025,.975]).tolist() if vals else None}
 for d in (1,2,3):contrasts[f'L{d}_M_minus_N']=contrast({f'L{d}_M':1,f'L{d}_N':-1})
 for m in ('N','M'):
  for x,y in ((2,1),(3,2),(3,1)):contrasts[f'L{x}_minus_L{y}_{m}']=contrast({f'L{x}_{m}':1,f'L{y}_{m}':-1})
 contrasts['memory_x_depth_L3_L1']=contrast({'L3_M':1,'L1_M':-1,'L3_N':-1,'L1_N':1})
 usage={}
 for arm in ARMS:
  r=[x for x in rs if x['condition']==arm];usage[arm]={'returned':len(r),'valid':sum(x['valid'] for x in r),'length':sum(x.get('finish_reason')=='length' for x in r),'complete_chains':sum(c['condition']==arm and c['complete'] for c in chains),**{k:sum((x.get('usage') or {}).get(k,0) for x in r) for k in ('prompt_tokens','completion_tokens')},'reasoning_tokens':sum(((x.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0 for x in r)}
 report={'audit_passed':True,'all_360_attempted':len(rs)==360,'groups':groups,'contrasts':contrasts,'usage':usage,'chains':chains,'hashes':done['hashes'],'L3_nested_truth':'not validated; no c accuracy claim'};base.dump(OUT/'audit_results.json',report)
 print(json.dumps({'audit_passed':True,'usage':usage,'contrasts':contrasts}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','audit']);globals()[p.parse_args().stage]()
