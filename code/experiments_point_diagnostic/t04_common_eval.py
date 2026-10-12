"""T04 selection audit and frozen common action/belief evaluation."""
import argparse, hashlib, importlib.metadata, inspect, json, math, os, sys
from pathlib import Path
ROOT=Path(os.environ['TOM_PACE_ROOT'])/'LLM_towards_ToM_bridge_v1_20260808/LLM_vs_Agents_vllm'
RUN=ROOT/'experiments/t04_convergence_20260915'
OLD=ROOT/'experiments/g03_matched_pop_20260915'
sys.path.insert(0,str(OLD))
from g03_matched_eval import MODEL,PARENT,KEY,SCHEMA,sha

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False,default=str)

def audit():
    arms={}
    for arm,job in [('base','13238347'),('parent','13238348')]:
        d=RUN/arm;c=read(d/'completion.json');r=read(d/'runtime.json');p=read(f't04_{arm}_preflight_20260915.json')
        assert str(r['job_id'])==job and c['runtime_sha256']==sha(d/'runtime.json')
        p.pop('job_id',None);rc=r.copy();rc.pop('job_id',None)
        for x in (p,rc):x['hashes']={str(Path(k).resolve()):v for k,v in x['hashes'].items()}
        assert p==rc
        assert all(sha(k)==v for k,v in rc['hashes'].items())
        scores=c['scores'];assert [s['step'] for s in scores]==list(range(0,c['global_step']+1,100))
        truth=None
        for s in scores:
            v=read(d/f"action_validation_step{s['step']}.json");rows=v.pop('predictions');assert v==s and len(rows)==1350
            keys=[(x['match_id'],x['round'],x['q']) for x in rows]
            assert len(set((x[0],x[1]) for x in keys))==1350
            if truth is None:truth=keys
            assert keys==truth and all(0<=x['p']<=1 and 0<=x['q']<=1 and 0<=x['digit_mass']<=1.00001 for x in rows)
            mse=sum((x['p']-x['q'])**2 for x in rows)/1350
            brier=sum((x['p']-x['q'])**2+x['q']*(1-x['q']) for x in rows)/1350
            assert abs(mse-s['probability_mse'])<1e-12 and abs(brier-s['expected_brier'])<1e-12
        best=min(scores,key=lambda s:(s['expected_brier'],s['step']))
        assert c['selected_step']==best['step'] and c['selected_score']==best['expected_brier']
        cp=Path(c['selected_path']);assert cp==d/f"checkpoints/checkpoint-{best['step']}"
        assert sha(cp/'adapter_model.safetensors')==c['selected_adapter_sha256']
        state=read(cp/'trainer_state.json');assert state['global_step']==best['step']
        assert c['stop_reason']=='early_stopping' and c['global_step']<c['max_steps']
        assert c['global_step']-best['step']==500
        assert all(s['expected_brier']>=best['expected_brier']-1e-4 for s in scores if s['step']>best['step'])
        arms[arm]={'completion':c,'runtime_sha256':sha(d/'runtime.json'),'validation_keys_sha256':hashlib.sha256(json.dumps(truth).encode()).hexdigest()}
    assert arms['base']['validation_keys_sha256']==arms['parent']['validation_keys_sha256']
    return arms

def prepare():
    from transformers import AutoTokenizer
    arms=audit()
    assert sha('g01_inputs.json')=='df86b93ceb54d80cb525a6455e235bf7858a4b9ecb8b49a6b7b871f3ba2a3f59'
    data=read('g01_inputs.json')['records'];assert len(data)==60
    old=read(OLD/'outputs/provenance.json');belief=read(OLD/'g03_prompts.json')['records']
    assert sha(OLD/'g03_prompts.json')==old['input_sha256']
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    def render(text):return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    old_sentence='Report your predictive probability, not a sampled action.'
    action=[]
    for r in data:
        assert r['prompt'].count(old_sentence)==1
        s=r['prompt'].replace(old_sentence,'Predict the opponent next action. Respond with exactly one digit: 0 or 1. Do not explain.')
        action.append({'match_id':r['match_id'],'decision_round':r['decision_round'],'prompt':s,'rendered':render(s)})
    assert len({(r['match_id'],r['decision_round']) for r in action})==60
    assert {(r['match_id'],r['decision_round']) for r in action}=={(r['match_id'],r['decision_round']) for r in belief}
    for r in belief:r['rendered']=render(r['prompt'])
    assert [hashlib.sha256(r['rendered'].encode()).hexdigest() for r in belief]==old['rendered_prompt_sha256']
    assert all(len(tok.encode(r['rendered'],add_special_tokens=False))<8192 for r in action+belief)
    ids=[tok.encode(str(i),add_special_tokens=False) for i in (0,1)];assert all(len(x)==1 for x in ids)
    models={'original_qwen':None,'original_parent':str(PARENT),**{f'{a}_selected':arms[a]['completion']['selected_path'] for a in arms}}
    prov={'job_id':os.getenv('SLURM_JOB_ID'),'code_sha256':sha(__file__),'training_audit':arms,'model_snapshot':str(MODEL),
          'models':{k:{'path':p,'adapter_sha256':sha(Path(p)/'adapter_model.safetensors') if p else None} for k,p in models.items()},
          'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','vllm')},
          'g01_sha256':sha('g01_inputs.json'),'g03_sha256':sha(OLD/'g03_prompts.json'),
          'action_rendered_hashes':[hashlib.sha256(r['rendered'].encode()).hexdigest() for r in action],
          'belief_rendered_hashes':[hashlib.sha256(r['rendered'].encode()).hexdigest() for r in belief],
          'action_settings':{'dtype':'bfloat16','batch_size':1,'literal_digit_ids':[x[0] for x in ids],'no_generation':True,'attention':'eager'},
          'belief_settings':{'temperature':0,'top_p':1,'seed':20260915,'max_tokens':4096,'structured_schema':SCHEMA}}
    assert prov['versions']==old['versions']
    return tok,action,belief,models,prov

def run(stage):
    tok,action,belief,models,prov=prepare()
    if stage=='preflight':dump('preflight.json',prov);print('TRAINING_AUDIT_AND_COMMON_PREFLIGHT_PASSED');return
    pre=read('preflight.json');check=prov.copy();check['job_id']=pre['job_id'];assert check==pre
    out=Path(stage);out.mkdir(exist_ok=False);dump(out/'provenance.json',prov)
    if stage=='action':
        import gc,torch
        from transformers import AutoModelForCausalLM,set_seed
        from peft import PeftModel
        set_seed(20260915);ids=prov['action_settings']['literal_digit_ids']
        with (out/'responses.jsonl').open('x') as f:
            for name,path in models.items():
                model=AutoModelForCausalLM.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').to('cuda')
                if path:model=PeftModel.from_pretrained(model,path)
                model.eval()
                with torch.inference_mode():
                    for r in action:
                        inputs=tok(r['rendered'],return_tensors='pt',add_special_tokens=False).to('cuda')
                        logits=model(**inputs,use_cache=False).logits[0,-1].float();binary=logits[ids]
                        p=torch.softmax(binary,dim=-1)[1].item();mass=torch.exp(torch.logsumexp(binary,0)-torch.logsumexp(logits,0)).item()
                        assert math.isfinite(p) and 0<=p<=1
                        f.write(json.dumps({'model':name,'match_id':r['match_id'],'round':r['decision_round'],'p':p,'digit_mass':mass,'valid':True})+'\n');f.flush()
                del model;gc.collect();torch.cuda.empty_cache();print('ACTION_MODEL_COMPLETED',name,flush=True)
    else:
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        import vllm.sampling_params as sp
        sampling=SamplingParams(temperature=0,top_p=1,max_tokens=4096,seed=20260915,structured_outputs=sp.StructuredOutputsParams(json=SCHEMA))
        assert str(sampling)==read(OLD/'outputs/provenance.json')['sampling']
        llm=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.85,max_model_len=8192,max_num_seqs=8,enforce_eager=True,enable_lora=True,max_loras=1,max_cpu_loras=3,max_lora_rank=16)
        with (out/'responses.jsonl').open('x') as f:
            for i,(name,path) in enumerate(models.items(),1):
                results=llm.generate([r['rendered'] for r in belief],sampling,lora_request=LoRARequest(name,i,path) if path else None)
                assert len(results)==60
                for r,result in zip(belief,results):
                    assert result.prompt==r['rendered'];c=result.outputs[0];p=None;error=None
                    try:
                        obj=json.loads(c.text);assert set(obj)=={KEY};p=obj[KEY];assert type(p) in (float,int) and math.isfinite(p) and 0<=p<=1 and c.finish_reason=='stop'
                    except Exception as exc:p=None;error=type(exc).__name__
                    f.write(json.dumps({'model':name,'match_id':r['match_id'],'round':r['decision_round'],'p':p,'text':c.text,'valid':p is not None,'parse_error':error,'finish_reason':c.finish_reason,'output_tokens':len(c.token_ids)})+'\n');f.flush()
                print('BELIEF_MODEL_COMPLETED',name,flush=True)
    dump(out/'completion.json',{'rows':240,'responses_sha256':sha(out/'responses.jsonl'),'provenance_sha256':sha(out/'provenance.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','action','belief']);run(p.parse_args().stage)
