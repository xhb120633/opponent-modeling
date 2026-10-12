"""Bounded two-arm LLM-native experiment. No simulator calls at model inference."""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from r3_20260916_data import OUT, sha, dump, prompt, answer, parse

MODEL='Qwen/Qwen2.5-7B-Instruct'
REMOTE=Path(os.environ['TOM_R3_RUN_DIR'])
MAX_NEW=768

def controls(model,tok,matches,dest):
    selected=[];seen=set()
    for m in matches:
        cell=(m['a_role'],m['level'])
        if cell not in seen:selected.append(m);seen.add(cell)
    records=[];started=time.monotonic();model.eval()
    with torch.inference_mode():
        for condition in ('current_readout','single_update'):
            for m in selected:
                print('CONTROL_START',condition,m['id'],flush=True)
                row=m['rounds'][1]
                if condition=='current_readout':
                    text=prompt(m,row,'A')+f"\nTRUE CURRENT state={json.dumps(row['state'])}. Read action probability from this state; do NOT update again."
                    arm='A'
                else:text=prompt(m,row,'B',row['previous']);arm='B'
                chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
                inputs=tok(chat,return_tensors='pt',add_special_tokens=False).to(model.device)
                output=model.generate(**inputs,max_new_tokens=MAX_NEW if arm=='B' else 48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                raw=tok.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True);obj=parse(raw,m,arm)
                records.append({'condition':condition,'match_id':m['id'],'round':2,'valid':obj is not None,'raw':raw,'q':row['p'],
                    'strict_valid':parse(raw,m,arm,allow_fence=False) is not None,
                    'p':obj['p_opponent_1'] if obj else None,'generated_tokens':int(output.shape[1]-inputs['input_ids'].shape[1])})
                with (dest/'controls_progress.jsonl').open('a') as f:
                    f.write(json.dumps(records[-1])+'\n');f.flush();os.fsync(f.fileno())
                print('CONTROL_END',condition,m['id'],obj is not None,flush=True)
    counts={c:sum(r['valid'] for r in records if r['condition']==c) for c in ('current_readout','single_update')}
    dump(dest/'raw_controls.json',{'elapsed_seconds':time.monotonic()-started,'counts':counts,'per_condition':6,'rows':records,
        'smoke_runtime_sha256':sha(dest/'smoke_runtime.json'),
        'interpretation':'known identity/params, no equation source; accuracy not training gate; all-invalid interface requires review'})
    assert all(v>0 for v in counts.values()), 'All-invalid raw interface: stop before training and review measurement'
    model.train()

def tokenize(tok,text,target):
    prefix=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    pre=tok.encode(prefix,add_special_tokens=False)
    enc=tok(target,add_special_tokens=False,return_offsets_mapping=True)
    prob=re.search(r'"p_opponent_1":([-+0-9.eE]+)',target).span(1)
    state=re.search(r'"state":\[([^]]*)\]',target)
    spans=[]
    if state:
        for m in re.finditer(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',state.group(1)):
            spans.append((state.start(1)+m.start(),state.start(1)+m.end()))
    action_idx=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<prob[1] and b>prob[0]]
    state_idx=[i for i,(a,b) in enumerate(enc['offset_mapping']) if any(a<d and b>c for c,d in spans)]
    assert action_idx and (not state or state_idx)
    weights=[0.0]*len(enc['input_ids'])
    for i in action_idx:weights[i]=1/len(action_idx)
    for i in state_idx:weights[i]=.5/len(state_idx)
    other=[i for i in range(len(weights)) if i not in action_idx and i not in state_idx]
    for i in other:weights[i]=.1/(len(other)+1)
    ids=pre+enc['input_ids']+[tok.eos_token_id]
    assert len(ids)<4096 and len(enc['input_ids'])<MAX_NEW
    return {'input_ids':ids,'labels':[-100]*len(pre)+enc['input_ids']+[tok.eos_token_id],
            'weights':[0.]*len(pre)+weights+[.1/(len(other)+1)], 'action_tokens':len(action_idx),'state_tokens':len(state_idx)}

class Collator:
    def __init__(self,tok):self.tok=tok
    def __call__(self,rows):
        width=max(len(r['input_ids']) for r in rows)
        result={k:[] for k in ('input_ids','attention_mask','labels','weights')}
        for r in rows:
            n=len(r['input_ids']);pad=width-n
            result['input_ids'].append(r['input_ids']+[self.tok.pad_token_id]*pad)
            result['attention_mask'].append([1]*n+[0]*pad)
            result['labels'].append(r['labels']+[-100]*pad)
            result['weights'].append(r['weights']+[0.]*pad)
        return {k:torch.tensor(v,dtype=torch.float32 if k=='weights' else torch.long) for k,v in result.items()}

class WeightedTrainer(Trainer):
    def training_step(self,model,inputs,num_items_in_batch=None):
        # Single-device only: Trainer scales by its actual accumulation-group length.
        # Transformers5.5 + Accelerate1.8 also configures Accelerate to divide again.
        # Disable only that second backward scaling during this call; preserve scheduling.
        assert self.accelerator.num_processes==1
        previous=self.accelerator.gradient_accumulation_steps
        self.accelerator.gradient_accumulation_steps=1
        try:return super().training_step(model,inputs,num_items_in_batch)
        finally:self.accelerator.gradient_accumulation_steps=previous

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        weights=inputs.pop('weights');labels=inputs.pop('labels')
        output=model(**inputs,use_cache=False)
        loss=F.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),labels[:,1:].reshape(-1),reduction='none',ignore_index=-100)
        loss=(loss.reshape(labels.shape[0],-1)*weights[:,1:]).sum(-1).mean()
        return (loss,output) if return_outputs else loss

def rollout(model,tok,matches,arm,collection=False):
    was_training=model.training;model.eval();old_padding=tok.padding_side;tok.padding_side='left'
    priors={m['id']:m['initial'] for m in matches};stopped=set();rows=[];examples=[]
    with torch.inference_mode():
        for rindex in range(10):
            for start in range(0,len(matches),8):
                batch=[m for m in matches[start:start+8] if collection or m['id'] not in stopped]
                if not batch:continue
                texts=[prompt(m,m['rounds'][rindex],arm,priors[m['id']]) for m in batch]
                chats=[tok.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True) for t in texts]
                inputs=tok(chats,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
                outputs=model.generate(**inputs,max_new_tokens=MAX_NEW if arm=='B' else 48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                generated=tok.batch_decode(outputs[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                for bi,(m,t,raw) in enumerate(zip(batch,texts,generated)):
                    r=m['rounds'][rindex];obj=parse(raw,m,arm)
                    sequence=outputs[bi,inputs['input_ids'].shape[1]:].tolist()
                    eos=sequence.index(tok.eos_token_id) if tok.eos_token_id in sequence else None
                    generated_count=eos+1 if eos is not None else len(sequence)
                    rows.append({'match_id':m['id'],'round':rindex+1,'valid':obj is not None,'strict_valid':parse(raw,m,arm,allow_fence=False) is not None,'p':obj['p_opponent_1'] if obj else None,'q':r['p'],'y':r['y'],'raw':raw,
                                 'generated_tokens':generated_count,'finish_reason':'eos' if eos is not None else 'length'})
                    if collection:examples.append({'prompt':t,'target':answer(r,arm),'match_id':m['id'],'round':rindex+1,'prior_source':'own_frozen_warmup' if rindex else 'public_init'})
                    if arm=='B':
                        if obj:priors[m['id']]=obj['state']
                        else:
                            invalid_ids=tok.encode(raw,add_special_tokens=False)
                            priors[m['id']]={'invalid_state_output':tok.decode(invalid_ids[:256],skip_special_tokens=True),
                                             'text_truncated':len(invalid_ids)>256}
                            if not collection:stopped.add(m['id'])
    tok.padding_side=old_padding
    if was_training:model.train()
    expected=len(matches)*10
    # Missing/invalid entries receive fixed worst squared-error penalty1, never valid-only selection.
    score=(sum((r['p']-r['q'])**2 for r in rows if r['valid'])+expected-sum(r['valid'] for r in rows))/expected
    return {'penalized_probability_mse':score,'expected':expected,'valid':sum(r['valid'] for r in rows),'rows':rows},examples

class Validation(TrainerCallback):
    def __init__(self,tok,matches,arm,dest):self.tok=tok;self.matches=matches;self.arm=arm;self.dest=dest;self.best=math.inf;self.bad=0;self.scores=[]
    def evaluate(self,model,step):
        metrics,_=rollout(model,self.tok,self.matches,self.arm)
        dump(self.dest/f'validation_step{step}.json',metrics)
        value=metrics['penalized_probability_mse'];improved=value<self.best-1e-4
        if improved:
            self.best=value;self.bad=0
            checkpoint=self.dest/f'selected_step{step}';model.save_pretrained(checkpoint);self.tok.save_pretrained(checkpoint)
            self.selected=checkpoint
        else:self.bad+=1
        self.scores.append({'step':step,'score':value,'valid':metrics['valid'],'improved':improved})
        print('VALIDATION',self.arm,json.dumps(self.scores[-1]),flush=True)
    def on_step_end(self,args,state,control,model=None,**kwargs):
        if state.global_step%50==0:
            self.evaluate(model,state.global_step)
            if self.bad>=3:control.should_training_stop=True
        return control

def dataset(tok,matches,arm):
    return [tokenize(tok,prompt(m,r,arm,r['previous']),answer(r,arm)) for m in matches for r in m['rounds']]

def stage(model,tok,data,val,arm,dest,steps):
    dest.mkdir();callback=Validation(tok,val,arm,dest);callback.evaluate(model,0)
    args=TrainingArguments(output_dir=str(dest/'trainer'),max_steps=steps,per_device_train_batch_size=1,
        gradient_accumulation_steps=16,learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,
        bf16=True,gradient_checkpointing=True,logging_steps=10,save_strategy='no',eval_strategy='no',
        remove_unused_columns=False,report_to=[],seed=2026,data_seed=2026,optim='adamw_torch',max_grad_norm=1.)
    trainer=WeightedTrainer(model=model,args=args,train_dataset=data,data_collator=Collator(tok),callbacks=[callback])
    # This loss is per-microbatch mean; HF must perform accumulation scaling, not assume kwargs-aware loss.
    trainer.model_accepts_loss_kwargs=False
    started=time.monotonic();trainer.train()
    model.save_pretrained(dest/'last');tok.save_pretrained(dest/'last')
    dump(dest/'completion.json',{'steps':trainer.state.global_step,'selected_path':str(callback.selected),'scores':callback.scores,
        'selected_hash':sha(callback.selected/'adapter_model.safetensors'),'action_loss_weight':1,'state_loss_weight':.5,'format_loss_weight':.1,'train_wall_seconds':time.monotonic()-started,
        'train_examples':len(data),'supervised_action_tokens':sum(r['action_tokens'] for r in data),'supervised_state_tokens':sum(r['state_tokens'] for r in data)})
    return callback.selected

def main():
    print('PROCESS_START',os.getpid(),os.uname().nodename,time.time(),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A','B'],required=True);parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--allow-train',action='store_true');cli=parser.parse_args()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert all(sha(OUT/f'{s}.json')==manifest['files'][s] for s in ('train','val','test'))
    train=json.loads((OUT/'train.json').read_text());val=json.loads((OUT/'val.json').read_text())
    # The training process deliberately never loads diagnostic-test contents.
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token=tok.eos_token
    data=dataset(tok,train,cli.arm);validation_tokens=dataset(tok,val,cli.arm)
    if cli.arm=='B':
        for m in train+val:
            worst={'invalid_state_output':tok.decode(tok.encode('badstate '*256,add_special_tokens=False)[:256]),'text_truncated':True}
            for r in m['rounds'][-1:]:tokenize(tok,prompt(m,r,'B',worst),answer(r,'B'))
    dest=REMOTE/cli.arm;dest.mkdir(parents=True,exist_ok=True)
    config={'arm':cli.arm,'model':MODEL,'seed':2026,'manifest_sha256':sha(OUT/'manifest.json'),'driver_sha256':sha(__file__),
        'max_tokens':max(len(r['input_ids']) for r in data+validation_tokens),'max_new_tokens_B':MAX_NEW,'validation_cadence':50,'patience':3,'min_delta':1e-4,
        'loss':'mean_action_numeric_token_CE + 0.5*mean_state_numeric_token_CE + 0.1*mean_format_EOS_CE','invalid_selection_penalty':1,'job':os.getenv('SLURM_JOB_ID'),
        'renderer_sha256':sha(Path(__file__).with_name('r3_20260916_data.py'))}
    dump(dest/('preflight_v2.json' if cli.preflight else 'smoke_runtime.json' if cli.smoke else 'runtime.json'),config)
    if cli.preflight:print('PREFLIGHT_OK',json.dumps(config));return
    if not cli.smoke:
        assert cli.allow_train, 'Training explicitly gated until remote novelty, tokenizer preflight and smoke audits'
        prior=json.loads((dest/'preflight_v2.json').read_text());assert prior['driver_sha256']==config['driver_sha256'] and prior['renderer_sha256']==config['renderer_sha256']
        smoke=json.loads((dest/'raw_controls.json').read_text());assert all(v>0 for v in smoke['counts'].values())
        assert smoke['smoke_runtime_sha256']==sha(dest/'smoke_runtime.json')
        sr=json.loads((dest/'smoke_runtime.json').read_text())
        assert all(sr[k]==config[k] for k in ('driver_sha256','renderer_sha256','manifest_sha256','model','arm'))
    set_seed(2026)
    print('MODEL_LOAD_START',flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');model.config.use_cache=False
    print('MODEL_ON_GPU',torch.cuda.get_device_name(),flush=True)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    print('ADAPTER_READY',flush=True)
    model.enable_input_require_grads()
    if cli.smoke:controls(model,tok,val,dest);return
    first=stage(model,tok,data,val,cli.arm,dest/'warmup',500 if cli.arm=='A' else 250)
    if cli.arm=='B':
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model,load_file(str(first/'adapter_model.safetensors')))
        metrics,collection=rollout(model,tok,train,'B',collection=True)
        dump(dest/'own_state_collection.json',{'source_checkpoint':str(first),'source_sha256':sha(first/'adapter_model.safetensors'),'metrics':metrics,'examples':collection})
        own=[tokenize(tok,r['prompt'],r['target']) for r in collection]
        assert len(own)==len(data)==2400
        selected=stage(model,tok,data+own,val,'B',dest/'repair',250)
    else:selected=first
    dump(dest/'completion.json',{'selected_path':str(selected),'selected_hash':sha(selected/'adapter_model.safetensors'),'status':'training_completed_not_tested'})

if __name__=='__main__':main()
