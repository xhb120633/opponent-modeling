"""R4 fixed-observation information-removal and horizon transfer. No training."""
import os
import copy,json,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=Path(os.environ['TOM_R3_RUN_DIR'])
if OLD.exists():sys.path.insert(0,str(OLD))
import r3_20260916_data as d
PACK=ROOT/'r4_package_20260917'
DEST=ROOT/'r4_outputs_20260917'
HASH='fa6d83dba94e7142455d232b32a262723a51d73308d7dc6d8e1f432be53e78c7'

def unknown(m,r):
    return (f"Forecast B's next action1 probability. A index={m['a_role']}; B index={m['b_role']}. "
        'Index0 is seeker (matching actions rewarded); index1 is hider (different actions rewarded). '
        'B identity, parameters and private state are unknown. Only completed (A_action,B_action) pairs are observations. Current/future choices are unavailable.\n'
        f"Decision round={r['round']}; complete past={json.dumps(r['history'],separators=(',',':'))}\n"
        'Return exactly JSON {"p_opponent_1": number}, no explanation.')

def prepare():
    assert not PACK.exists()
    matches=[]
    for role in (0,1):
        for level in (0,1,2):
            for rep in (0,1):
                seed=2026091700+role*100+level*10+rep;rng=np.random.default_rng(seed)
                state=d.kt.init_k_tom(d.PARAMS,level);keys=list(d.flatten(state));initial=[d.flatten(state)[k] for k in keys]
                m=dict(id=f'r4_A{role}_ToM{level}_{rep}',seed=seed,a_role=role,b_role=1-role,level=level,keys=keys,initial=initial,rounds=[])
                phase=rng.uniform(0,2*np.pi);offset=rng.uniform(-.6,.6);history=[]
                m['a_policy']=dict(phase=float(phase),offset=float(offset),formula='sigmoid(offset+1.4*sin(phase+0.9*t))')
                for t in range(30):
                    p=d.probability(state,level,1-role);pa=1/(1+np.exp(-(offset+1.4*np.sin(phase+.9*t))))
                    a=int(rng.random()<pa);b=int(rng.random()<p)
                    m['rounds'].append(dict(round=t+1,history=copy.deepcopy(history),p=p,y=b))
                    history.append([a,b]);state=d.kt.learning_function(state,d.PARAMS,b,a,level,1-role,d.MATRIX)
                matches.append(m)
    test=json.loads((d.OUT/'test.json').read_text());assert len(test)==60
    previous_seeds={m['seed'] for s in ('train','val','test') for m in json.loads((d.OUT/f'{s}.json').read_text())}
    assert not previous_seeds & {m['seed'] for m in matches}
    for m in test:
        for r in m['rounds']:
            assert len(r['history'])==r['round']-1
            assert unknown(m,r)==unknown({**m,'level':999,'keys':[],'initial':[]},{**r,'p':999,'state':[],'y':99})
    PACK.mkdir();d.dump(PACK/'long.json',matches);d.dump(PACK/'test.json',test)
    d.dump(PACK/'manifest.json',dict(files={n:d.sha(PACK/n) for n in ('long.json','test.json')},script_sha256=d.sha(__file__),renderer_sha256=d.sha(d.__file__),simulator_sha256=d.sha(d.kt.__file__),old_test_sha256=d.sha(d.OUT/'test.json'),counts=dict(unknown=600,long=360),checks=['disjoint generator seeds','unknown renderer ignores hidden targets and identity','same R3 test bytes after JSON serialization semantic check']))
    print('R4 prepared:960 predictions/model, two models; no inference',flush=True)

def run():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    from peft import PeftModel
    pre=json.loads((PACK/'manifest.json').read_text());assert all(d.sha(PACK/n)==h for n,h in pre['files'].items())
    assert d.sha(__file__)==pre['script_sha256'] and d.sha(d.__file__)==pre['renderer_sha256'] and d.sha(d.kt.__file__)==pre['simulator_sha256']
    done=json.loads((OLD/'A/completion.json').read_text());adapter=Path(done['selected_path']);assert adapter.resolve().is_relative_to((OLD/'A').resolve())
    assert done['selected_hash']==d.sha(adapter/'adapter_model.safetensors')==HASH
    DEST.mkdir();d.dump(DEST/'runtime.json',dict(job=os.getenv('SLURM_JOB_ID'),started=time.time(),adapter=str(adapter),adapter_sha256=HASH,package_sha256=d.sha(PACK/'manifest.json'),source_sha256=d.sha(__file__)))
    set_seed(2026);name='Qwen/Qwen2.5-7B-Instruct';tok=AutoTokenizer.from_pretrained(name,local_files_only=True);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');model.eval()
    groups={};allrows=[]
    for modelname in ('initial','trained_A'):
        if modelname=='trained_A':model=PeftModel.from_pretrained(model,str(adapter));model.eval()
        with torch.inference_mode(),(DEST/f'{modelname}.jsonl').open('x') as f:
            for condition,filename in [('unknown_same_history','test.json'),('known_long_history','long.json')]:
                for m in json.loads((PACK/filename).read_text()):
                    for r in m['rounds']:
                        prompt=unknown(m,r) if condition.startswith('unknown') else d.prompt(m,r,'A')
                        chat=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
                        inputs=tok(chat,return_tensors='pt',add_special_tokens=False).to(model.device)
                        output=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                        raw=tok.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True);parsed=d.parse(raw,m,'A')
                        row=dict(model=modelname,condition=condition,match_id=m['id'],round=r['round'],role=m['a_role'],level=m['level'],prompt=prompt,raw=raw,valid=parsed is not None,p=parsed['p_opponent_1'] if parsed else None,q=r['p'],y=r['y'],frequency=(1+sum(b for a,b in r['history']))/(2+len(r['history'])))
                        f.write(json.dumps(row)+'\n');f.flush();allrows.append(row)
                    print(modelname,condition,m['id'],'completed',flush=True)
        if modelname=='trained_A':model=model.unload()
    assert len(allrows)==1920
    def metrics(rr):
        vv=[r for r in rr if r['valid']]
        return dict(n=len(rr),valid=len(vv),mse=float(np.mean([(r['p']-r['q'])**2 if r['valid'] else 1 for r in rr])),valid_mae=float(np.mean([abs(r['p']-r['q']) for r in vv])) if vv else None,constant_mse=float(np.mean([(.5-r['q'])**2 for r in rr])),frequency_mse=float(np.mean([(r['frequency']-r['q'])**2 for r in rr])))
    for modelname in ('initial','trained_A'):
        for condition in ('unknown_same_history','known_long_history'):
            rr=[r for r in allrows if r['model']==modelname and r['condition']==condition and r['round']>1]
            groups[modelname+'_'+condition]={'all':metrics(rr),**{f'A{role}_ToM{l}':metrics([r for r in rr if r['role']==role and r['level']==l]) for role in (0,1) for l in (0,1,2)}}
            if condition=='known_long_history':groups[modelname+'_'+condition].update(early=metrics([r for r in rr if r['round']<=10]),late=metrics([r for r in rr if r['round']>10]))
    d.dump(DEST/'metrics.json',groups);d.dump(DEST/'completed.json',dict(status='executed_pending_independent_audit',rows=len(allrows),files={n:d.sha(DEST/n) for n in ('runtime.json','initial.jsonl','trained_A.jsonl','metrics.json')}))

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
