"""R5 frozen matched continuation; training/test boundaries are explicit."""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import r3_20260916_data as d
from r4_transfer_20260917 import unknown

ROOT=Path(__file__).resolve().parent
PACK=ROOT/'r5_package_20260918'
OLD=Path(os.environ['TOM_R3_RUN_DIR'])
AHASH='fa6d83dba94e7142455d232b32a262723a51d73308d7dc6d8e1f432be53e78c7'
SOURCES=['r5_metadata_20260918.py','r5_metadata_20260918.sh','test_r5_metadata_20260918.py','r3_20260916_data.py','r3_20260916_train.py','r4_transfer_20260917.py','test_r3_gradient_accum.py']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def condition(arm,index,round_index):
    return 'unknown' if arm=='mixed' and (index+round_index)%2 else 'known'
def render(m,r,info):return unknown(m,r) if info=='unknown' else d.prompt(m,r,'A')
def encode_prompt(tok,text):return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)

def prepare():
    assert not PACK.exists(),'Exclusive package; never overwrite'
    oldman=load(d.OUT/'manifest.json')
    old={s:load(d.OUT/f'{s}.json') for s in ('train','val','test')}
    assert all(d.sha(d.OUT/f'{s}.json')==oldman['files'][s] for s in old)
    longpath=ROOT/'r4_package_20260917/long.json'
    long=load(longpath)
    seeds={m['seed'] for mm in [*old.values(),long] for m in mm}
    fresh=[]
    for role in (0,1):
        for level in (0,1,2):
            for rep in range(10):
                seed=2026180000+role*10000+level*1000+rep
                assert seed not in seeds
                rng=np.random.default_rng(seed);state=d.kt.init_k_tom(d.PARAMS,level)
                flat=d.flatten(state);keys=list(flat);history=[]
                m=dict(id=f'r5fresh_A{role}_ToM{level}_{rep:02d}',seed=seed,a_role=role,b_role=1-role,level=level,keys=keys,initial=[flat[k] for k in keys],rounds=[])
                phase=float(rng.uniform(0,2*np.pi));offset=float(rng.uniform(-.6,.6));m['a_policy']=dict(phase=phase,offset=offset,formula='sigmoid(offset+1.4*sin(phase+0.9*t))')
                for t in range(10):
                    p=d.probability(state,level,1-role);pa=1/(1+np.exp(-(offset+1.4*np.sin(phase+.9*t))))
                    a=int(rng.random()<pa);b=int(rng.random()<p)
                    m['rounds'].append(dict(round=t+1,history=copy.deepcopy(history),p=p,y=b))
                    history.append([a,b]);state=d.kt.learning_function(state,d.PARAMS,b,a,level,1-role,d.MATRIX)
                fresh.append(m)
    ids=[{m['id'] for m in mm} for mm in [*old.values(),fresh,long]]
    assert all(not (a&b) for i,a in enumerate(ids) for b in ids[i+1:])
    examples={a:[] for a in ('known','mixed')}
    for arm in examples:
        for i,m in enumerate(old['train']):
            for j,r in enumerate(m['rounds']):
                info=condition(arm,i,j);text=render(m,r,info)
                mutate={**r,'p':-99,'y':-99,'state':[999],'previous':[999]}
                assert text==render(m,mutate,info)
                if info=='unknown':assert text==render({**m,'level':999,'keys':[],'initial':[999],'seed':-1,'a_policy':{}},mutate,info)
                examples[arm].append(dict(match_id=m['id'],round=r['round'],role=m['a_role'],level=m['level'],information=info,prompt=text,target=d.answer(r,'A')))
    assert len(examples['known'])==len(examples['mixed'])==2400
    assert sum(x['information']=='unknown' for x in examples['mixed'])==1200
    assert all((a['match_id'],a['round'],a['target'])==(b['match_id'],b['round'],b['target']) for a,b in zip(examples['known'],examples['mixed']))
    for role in (0,1):
        for level in (0,1,2):
            for rd in range(1,11):
                assert sum(x['information']=='unknown' for x in examples['mixed'] if x['role']==role and x['level']==level and x['round']==rd)==20
    PACK.mkdir()
    for s,mm in {**old,'fresh':fresh,'long':long}.items():d.dump(PACK/f'{s}.json',mm)
    for arm,rows in examples.items():d.dump(PACK/f'train_{arm}.json',rows)
    manifest=dict(files={p.name:d.sha(p) for p in sorted(PACK.glob('*.json'))},sources={n:d.sha(ROOT/n) for n in SOURCES},simulator_sha256=d.sha(d.kt.__file__),old_manifest_sha256=d.sha(d.OUT/'manifest.json'),original_adapter_sha256=AHASH,counts=dict(train=240,val=60,replay=60,fresh=60,long=12),frozen_seed=2026180000,checks=['complete-match split and RNG seeds disjoint','same target/order across arms','50% exposure in every role/level/round','unknown ignores hidden targets/metadata'],sealed_final_access=False)
    d.dump(PACK/'manifest.json',manifest);print('PREPARED',d.sha(PACK/'manifest.json'),flush=True)

def verify():
    pre=load(PACK/'manifest.json')
    assert all(d.sha(PACK/n)==h for n,h in pre['files'].items())
    assert all(d.sha(ROOT/n)==h for n,h in pre['sources'].items())
    assert d.sha(d.kt.__file__)==pre['simulator_sha256']
    return pre

def preflight():
    verify()
    from transformers import AutoTokenizer
    from r3_20260916_train import tokenize
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True);tok.pad_token=tok.eos_token
    counts={};val=load(PACK/'val.json')
    for arm in ('known','mixed'):
        rows=load(PACK/f'train_{arm}.json');data=[tokenize(tok,r['prompt'],r['target']) for r in rows]
        assert all(abs(sum(r['weights'])-1.1)<1e-6 and r['state_tokens']==0 for r in data)
        counts[arm]=dict(n=len(data),max_tokens=max(len(r['input_ids']) for r in data),action_tokens=sum(r['action_tokens'] for r in data))
    assert counts['known']['action_tokens']==counts['mixed']['action_tokens']
    maxval=0
    for info in ('known','unknown'):
        for m in val:
            for r in m['rounds']:maxval=max(maxval,len(tok.encode(encode_prompt(tok,render(m,r,info)),add_special_tokens=False)))
    assert maxval+48<4096
    done=load(OLD/'A/completion.json');adapter=Path(done['selected_path'])
    assert adapter.resolve().is_relative_to((OLD/'A').resolve())
    assert d.sha(adapter/'adapter_model.safetensors')==done['selected_hash']==AHASH
    result=dict(package_sha256=d.sha(PACK/'manifest.json'),driver_sha256=d.sha(__file__),counts=counts,max_validation_prompt_tokens=maxval,adapter=str(adapter),adapter_sha256=AHASH)
    print('PREFLIGHT_OK',json.dumps(result),flush=True);return result

def forecast(model,tok,matches,infos,dest,label):
    import torch
    records=[];was=model.training;model.eval();padding=tok.padding_side;tok.padding_side='left'
    with torch.inference_mode(),dest.open('x',encoding='utf-8') as f:
        for info in infos:
            work=[(m,r) for m in matches for r in m['rounds']]
            for start in range(0,len(work),8):
                batch=work[start:start+8];texts=[render(m,r,info) for m,r in batch]
                inp=tok([encode_prompt(tok,t) for t in texts],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
                assert inp['input_ids'].shape[1]+48<4096
                gen=model.generate(**inp,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                raws=tok.batch_decode(gen[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                for (m,r),text,raw in zip(batch,texts,raws):
                    obj=d.parse(raw,m,'A')
                    z=dict(label=label,information=info,match_id=m['id'],round=r['round'],role=m['a_role'],level=m['level'],prompt=text,raw=raw,valid=obj is not None,p=obj['p_opponent_1'] if obj else None,q=r['p'],frequency=(1+sum(b for a,b in r['history']))/(2+len(r['history'])))
                    f.write(json.dumps(z)+'\n');records.append(z)
                f.flush()
    tok.padding_side=padding;model.train(was)
    return records

def score(rows):
    rr=[r for r in rows if r['round']>1];valid=[r for r in rr if r['valid']]
    return dict(n=len(rr),valid=len(valid),mse=float(np.mean([(r['p']-r['q'])**2 if r['valid'] else 1 for r in rr])),mae=float(np.mean([abs(r['p']-r['q']) for r in valid])) if valid else None,constant=float(np.mean([(.5-r['q'])**2 for r in rr])),frequency=float(np.mean([(r['frequency']-r['q'])**2 for r in rr])))

def run(arm):
    pre=preflight()
    import torch, transformers, peft
    from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,TrainerCallback,set_seed
    from peft import PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from r3_20260916_train import tokenize,Collator,WeightedTrainer
    dest=ROOT/arm;dest.mkdir()
    d.dump(dest/'runtime.json',dict(**pre,arm=arm,job=os.getenv('SLURM_JOB_ID'),host=os.uname().nodename,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,started=time.time(),step_cap=200,seed=2026))
    train=load(PACK/f'train_{arm}.json');val=load(PACK/'val.json')
    set_seed(2026);tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True);tok.pad_token=tok.eos_token
    data=[tokenize(tok,r['prompt'],r['target']) for r in train]
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    model=PeftModel.from_pretrained(base,pre['adapter'],is_trainable=True)
    saved=load_file(str(Path(pre['adapter'])/'adapter_model.safetensors'));actual=get_peft_model_state_dict(model)
    assert saved.keys()==actual.keys() and all(torch.equal(saved[k],actual[k].cpu()) for k in saved),'Initialization weights differ'
    d.dump(dest/'initialization_audit.json',dict(passed=True,adapter_sha256=AHASH,tensors=len(saved),exact_tensor_match=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.config.use_cache=False;model.enable_input_require_grads()
    class Check(TrainerCallback):
        def __init__(self):self.best=float('inf');self.selected=None;self.scores=[]
        def evaluate(self,step):
            rr=forecast(model,tok,val,('known','unknown'),dest/f'validation_step{step}.jsonl',f'val_step{step}')
            per={info:score([r for r in rr if r['information']==info]) for info in ('known','unknown')}
            value=(per['known']['mse']+per['unknown']['mse'])/2
            assert all(z['n']==540 for z in per.values())
            if step==0:assert all(z['valid']==540 for z in per.values()),'Initial A coverage changed; stop before training'
            improved=value<self.best-1e-4
            if improved:
                self.best=value;self.selected=dest/f'selected_step{step}';model.save_pretrained(self.selected);tok.save_pretrained(self.selected)
            self.scores.append(dict(step=step,score=value,improved=improved,by_information=per));print('VALIDATION',arm,json.dumps(self.scores[-1]),flush=True)
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step%50==0:self.evaluate(state.global_step)
            return control
    cb=Check();cb.evaluate(0)
    # Restore identical RNG before training; different prompt lengths may still change dropout draws.
    set_seed(2026)
    args=TrainingArguments(output_dir=str(dest/'trainer'),max_steps=200,per_device_train_batch_size=1,gradient_accumulation_steps=16,learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,bf16=True,gradient_checkpointing=True,logging_steps=10,save_strategy='no',eval_strategy='no',remove_unused_columns=False,report_to=[],seed=2026,data_seed=2026,optim='adamw_torch',max_grad_norm=1.)
    trainer=WeightedTrainer(model=model,args=args,train_dataset=data,data_collator=Collator(tok),callbacks=[cb]);trainer.model_accepts_loss_kwargs=False
    start=time.monotonic();trainer.train();assert trainer.state.global_step==200
    model.save_pretrained(dest/'last');tok.save_pretrained(dest/'last')
    checkpoint=cb.selected;h=d.sha(checkpoint/'adapter_model.safetensors')
    d.dump(dest/'training_completed.json',dict(status='completed_pending_external_audit',steps=200,selected_path=str(checkpoint),selected_hash=h,scores=cb.scores,train_wall_seconds=time.monotonic()-start,package_sha256=d.sha(PACK/'manifest.json'),driver_sha256=d.sha(__file__),start_adapter_sha256=AHASH,examples=len(data),input_tokens=sum(len(x['input_ids']) for x in data)))
    # Dependent eval gate: completed exact step count, frozen inputs/sources and selected adapter integrity.
    verify();assert d.sha(checkpoint/'adapter_model.safetensors')==h
    set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
    summary={}
    for name,infos in [('test',('known','unknown')),('fresh',('known','unknown')),('long',('known',))]:
        rr=forecast(model,tok,load(PACK/f'{name}.json'),infos,dest/f'eval_{name}.jsonl',f'{arm}_selected')
        summary[name]={i:score([r for r in rr if r['information']==i]) for i in infos}
        print('EVAL',arm,name,json.dumps(summary[name]),flush=True)
    if arm=='known':
        set_peft_model_state_dict(model,saved)
        rr=forecast(model,tok,load(PACK/'fresh.json'),('known','unknown'),dest/'initial_A_fresh.jsonl','original_A500')
        summary['original_A500_fresh']={i:score([r for r in rr if r['information']==i]) for i in ('known','unknown')}
    d.dump(dest/'metrics.json',summary)
    d.dump(dest/'completed.json',dict(status='executed_pending_independent_audit',job=os.getenv('SLURM_JOB_ID'),arm=arm,files={p.name:d.sha(p) for p in dest.glob('*.json*')},selected_hash=h))
    print('COMPLETE',arm,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','run']);ap.add_argument('--arm',choices=['known','mixed']);a=ap.parse_args()
    if a.mode=='run':assert a.arm;run(a.arm)
    elif a.mode=='prepare':prepare()
    else:preflight()
