"""Matched SFT arms with history-only action validation; no oracle input."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import TrainerCallback, EarlyStoppingCallback, set_seed

ROOT=Path(os.environ['TOM_PACE_ROOT'])/'LLM_towards_ToM_bridge_v1_20260808/LLM_vs_Agents_vllm'
RUN=ROOT/'experiments/t04_convergence_20260915'
PARENT=Path(os.environ['TOM_PACE_ROOT'])/'LLM_towards_ToM/Hide_and_Seek/LLM_vs_Agents_vllm/checkpoints/sft_locked_v1_qwen25_7b_structured_tom_progress10_ep5/checkpoint-300'
sys.path.insert(0,str(ROOT))
import train_sft_locked_v1_trl as old
from evaluate_predictive_action_logits_vllm import build_probe_prompt

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,default=str,allow_nan=False)

def setup(arm):
    data=ROOT/'data_predictive_state_bridge_v1/predictive_state'
    manifest=json.loads((data/'manifest.json').read_text())
    assert not set(manifest['matches']['train'])&set(manifest['matches']['val'])
    assert sha(data/'train.jsonl')==manifest['output_sha256']['train.jsonl']=='958d0999ce92905becb6f2f081c47d1e89b3849e45d81a7f7295e9c350d92025'
    assert sha(data/'val.jsonl')==manifest['output_sha256']['val.jsonl']=='0f67749fd24341de08c6c80b14537bff618294529e863c1405c18a0cedc8f2d3'
    source=ROOT/'analysis/predictive_state_identifiability_v1/predictive_state_features.csv'
    assert sha(source)==manifest['source_sha256']
    assert sha(ROOT/'train_sft_locked_v1_trl.py')=='ddf8091d9344221d22f537a63eb3a8364afcd89b680e6f5f5c3eee08ad7970bc'
    assert sha(PARENT/'adapter_model.safetensors')=='e58c754e9068b50337ffbc8a3b390d95cb208d91fbc333bb1648815dd8688ed4'
    with source.open() as f: source_rows=list(csv.DictReader(f))
    keyed={(r['match_id'],int(r['decision_round'])):r for r in source_rows}
    assert len(keyed)==len(source_rows)
    tok=old.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    ids=[tok.encode(str(i),add_special_tokens=False) for i in (0,1)]
    assert all(len(x)==1 for x in ids)
    records=[]
    with (data/'val.jsonl').open() as f:
        for line in f:
            r=json.loads(line);m=r['metadata'];s=keyed[(m['match_id'],m['decision_round'])]
            assert s['role']==m['role'] and s['opponent_family']==m['opponent_family'] and s['split']=='id_val'
            q=float(s['p_self']);assert 0<=q<=1
            assert math.floor(10*q+.5)==r['target']['opponent_action_bin']
            prompt=build_probe_prompt(r['prompt'])
            assert 'opponent_belief_my_action_bin' not in prompt and 'Required fields:' not in prompt
            txt=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            token_ids=tok.encode(txt,add_special_tokens=False);assert len(token_ids)<8192
            records.append({'match_id':m['match_id'],'round':m['decision_round'],'q':q,'input_ids':token_ids})
    assert len(records)==1350
    train=old.load_jsonl_dataset(data/'train.jsonl');val=old.load_jsonl_dataset(data/'val.jsonl')
    collator=old.ValueOnlyDataCollator(tok,8192)
    counts=[];lengths=[]
    for ds in (train,val):
        for r in ds:
            batch=collator([r]);counts.append(int((batch['labels']!=-100).sum()));lengths.append(int(batch['attention_mask'].sum()))
    assert min(counts)>=5 and max(lengths)<8192
    args=torch.load(ROOT/'checkpoints/predictive_state_bridge_steps200_v1/training_args.bin',weights_only=False,map_location='cpu')
    args.output_dir=str(RUN/arm/'checkpoints');args.max_steps=-1;args.num_train_epochs=10
    args.eval_steps=100;args.save_steps=100;args.save_total_limit=3
    args.metric_for_best_model='action_expected_brier';args.greater_is_better=False;args.load_best_model_at_end=True
    config={'arm':arm,'init_adapter_path':str(PARENT) if arm=='parent' else None,'settings':args.to_dict(),
            'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','trl','peft')},
            'hashes':{str(p):sha(p) for p in (Path(__file__),data/'train.jsonl',data/'val.jsonl',source,PARENT/'adapter_model.safetensors')},
            'train_rows':len(train),'validation_rows':len(val),'min_value_tokens':min(counts),'max_training_tokens':max(lengths),
            'max_action_prompt_tokens':max(len(r['input_ids']) for r in records),'digit_token_ids':[i[0] for i in ids],
            'action_records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'patience':5,'threshold':1e-4,'constant_expected_brier':.25,'validation_role_source_checked':True}
    return tok,train,val,collator,args,records,config

class ActionValidation(TrainerCallback):
    def __init__(self,records,tokenizer,ids,dest):
        self.records=records;self.tok=tokenizer;self.ids=ids;self.dest=dest;self.scores=[]
    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        was_training=model.training;model.eval();predictions=[];masses=[]
        with torch.inference_mode():
            for start in range(0,len(self.records),8):
                records=self.records[start:start+8];width=max(len(r['input_ids']) for r in records)
                inputs=torch.full((len(records),width),self.tok.pad_token_id,dtype=torch.long,device=model.device)
                mask=torch.zeros_like(inputs)
                for i,r in enumerate(records):
                    n=len(r['input_ids']);inputs[i,-n:]=torch.tensor(r['input_ids'],device=model.device);mask[i,-n:]=1
                logits=model(input_ids=inputs,attention_mask=mask,use_cache=False).logits[:,-1,:].float()
                binary=logits[:,self.ids]
                predictions.extend(torch.softmax(binary,dim=-1)[:,1].cpu().tolist())
                masses.extend(torch.exp(torch.logsumexp(binary,dim=-1)-torch.logsumexp(logits,dim=-1)).cpu().tolist())
        if was_training:model.train()
        q=np.array([r['q'] for r in self.records]);p=np.array(predictions)
        mse=float(np.mean((p-q)**2));score=float(np.mean((p-q)**2+q*(1-q)))
        assert math.isfinite(score) and len(p)==1350
        metrics.update(eval_action_expected_brier=score,eval_action_probability_mse=mse,eval_digit_mass=float(np.mean(masses)))
        row={'step':state.global_step,'epoch':state.epoch,'expected_brier':score,'probability_mse':mse,'mean_digit_mass':float(np.mean(masses)),
             'predictions':[{'match_id':r['match_id'],'round':r['round'],'q':r['q'],'p':pv,'digit_mass':mass} for r,pv,mass in zip(self.records,predictions,masses)]}
        dump(self.dest/f'action_validation_step{state.global_step}.json',row)
        self.scores.append({k:v for k,v in row.items() if k!='predictions'})
        print('ACTION_VALIDATION',json.dumps(self.scores[-1]),flush=True)
        return control

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('base','parent'),required=True);parser.add_argument('--preflight',action='store_true')
    cli=parser.parse_args();dest=RUN/cli.arm
    tok,train,val,collator,args,records,config=setup(cli.arm)
    config['job_id']=os.getenv('SLURM_JOB_ID')
    dump(dest/('preflight.json' if cli.preflight else 'runtime.json'),config)
    if cli.preflight:
        print('PREFLIGHT_OK',cli.arm,len(records),flush=True);return
    assert not Path(args.output_dir).exists()
    set_seed(2026)
    model=old.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',torch_dtype=torch.bfloat16,local_files_only=True)
    model.config.use_cache=False
    lora=None
    if cli.arm=='parent':model=old.PeftModel.from_pretrained(model,str(PARENT),is_trainable=True)
    else:lora=old.LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    callback=ActionValidation(records,tok,config['digit_token_ids'],dest)
    trainer=old.SFTTrainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=collator,processing_class=tok,peft_config=lora,
                           callbacks=[callback,EarlyStoppingCallback(early_stopping_patience=5,early_stopping_threshold=1e-4)])
    trainer.save_model(str(dest/'initial'))
    trainer.evaluate()
    trainer.train()
    trainer.save_model(str(dest/'best_trained'));tok.save_pretrained(str(dest/'best_trained'))
    best=min(callback.scores,key=lambda r:(r['expected_brier'],r['step']))
    selected=dest/'initial' if best['step']==0 else Path(trainer.state.best_model_checkpoint)
    assert selected.exists()
    dump(dest/'completion.json',{'arm':cli.arm,'global_step':trainer.state.global_step,'max_steps':trainer.state.max_steps,
        'stop_reason':'early_stopping' if trainer.state.global_step<trainer.state.max_steps else 'epoch_cap_not_proof_of_convergence',
        'selected_path':str(selected),'selected_step':best['step'],'selected_score':best['expected_brier'],
        'selected_adapter_sha256':sha(selected/'adapter_model.safetensors'),'training_best_checkpoint':trainer.state.best_model_checkpoint,
        'scores':callback.scores,'runtime_sha256':sha(dest/'runtime.json')})
    print('T04_TRAINING_COMPLETED',cli.arm,flush=True)

if __name__=='__main__':main()
