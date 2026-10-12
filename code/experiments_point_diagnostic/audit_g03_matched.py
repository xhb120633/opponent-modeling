"""Audit frozen matched G03 responses and score B.p_op, no inference."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from transformers.utils.chat_template_utils import _compile_jinja_template

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'g03_job13228623'
OUT=RUN/'outputs'
MODELS=('base','parent','belief_sft_s100')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]
def correlation(a,b):
    return None if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def metric(rows):
    p=np.array([r['p'] for r in rows]); y=np.array([r['target'] for r in rows])
    if not len(rows):return {'n':0}
    groups=defaultdict(list)
    for r in rows:groups[r['match_id']].append(r)
    cp=[];cy=[];dp=[];dy=[];cells={}
    for mid,group in groups.items():
        group=sorted(group,key=lambda x:x['round'])
        a=np.array([r['p'] for r in group]);b=np.array([r['target'] for r in group])
        cp.extend(a-a.mean());cy.extend(b-b.mean())
        cells[mid]={'n':len(group),'mae':float(np.mean(abs(a-b))),'mse':float(np.mean((a-b)**2)),
                    'pearson':correlation(a,b),'spearman':None if correlation(a,b) is None else float(spearmanr(a,b).statistic),
                    'prediction_sd':float(np.std(a)),'target_sd':float(np.std(b))}
        for x,z in zip(group,group[1:]):
            if z['round']==x['round']+1:
                dp.append(z['p']-x['p']);dy.append(z['target']-x['target'])
    return {'n':len(rows),'matches':len(groups),'mae':float(np.mean(abs(p-y))),'mse':float(np.mean((p-y)**2)),
            'bias':float(np.mean(p-y)),'prediction_sd':float(np.std(p)),
            'pearson':correlation(p,y),'spearman':None if correlation(p,y) is None else float(spearmanr(p,y).statistic),
            'centered_pearson':correlation(cp,cy),'delta_pearson':correlation(dp,dy),'delta_pairs':len(dp),
            'unique_predictions':len(set(p.tolist())),'at_half_count':int(np.sum(p==.5)),'cells':cells,
            'by_round':{str(t):{'n':sum(r['round']==t for r in rows),
                'mae':float(np.mean([abs(r['p']-r['target']) for r in rows if r['round']==t]))} for t in sorted({r['round'] for r in rows})}}

def bootstrap(a,b):
    # Delta = b minus a, exact shared keys, resample complete matches.
    amap={(r['match_id'],r['round']):r for r in a};bmap={(r['match_id'],r['round']):r for r in b}
    keys=sorted(set(amap)&set(bmap));groups=defaultdict(list)
    for k in keys:
        ar,br=amap[k],bmap[k];assert ar['target']==br['target']
        ae=abs(ar['p']-ar['target']);be=abs(br['p']-br['target'])
        groups[k[0]].append((be-ae,be**2-ae**2))
    matrices=[np.array(v) for v in groups.values()]
    sums=np.array([v.sum(axis=0) for v in matrices]);counts=np.array([len(v) for v in matrices])
    rng=np.random.default_rng(20260915);idx=rng.integers(0,len(matrices),size=(10000,len(matrices)))
    samples=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None]
    point=sums.sum(axis=0)/counts.sum()
    return {'n':len(keys),'n_matches':len(matrices),'delta_mae':float(point[0]),'delta_mse':float(point[1]),
            'paired_match_bootstrap95_mae':np.quantile(samples[:,0],[.025,.975]).tolist(),
            'paired_match_bootstrap95_mse':np.quantile(samples[:,1],[.025,.975]).tolist(),
            'resamples':10000,'seed':20260915,'limit':'Exploratory six-match CI, no training-seed variance.'}

def main():
    completion=load(OUT/'completion.json');prov=load(OUT/'provenance.json');pre=load(ROOT/'g03_preflight_20260915.json')
    assert completion['rows']==180
    assert completion['responses_sha256']==sha(OUT/'responses.jsonl')
    assert completion['provenance_sha256']==sha(OUT/'provenance.json')
    assert prov['job_id']=='13228623'
    assert prov['code_sha256']==sha(ROOT/'g03_matched_eval.py')
    assert prov['input_sha256']==sha(ROOT/'g03_prompts.json')
    for k in pre:
        if k!='job_id':assert prov[k]==pre[k],k
    frozen=load(ROOT/'g02_inputs.json');prompts=load(ROOT/'g03_prompts.json')
    assert prov['source_g02_sha256']==sha(ROOT/'g02_inputs.json')
    expected=[{k:r[k] for k in ('match_id','decision_round','prompt')} for r in frozen['records']]
    assert prompts['records']==expected
    tc=load(RUN/'tokenizer_config.json')
    assert sha(RUN/'tokenizer_config.json')==prov['models']['base']['hashes']['tokenizer_config.json']
    template=_compile_jinja_template(tc['chat_template'])
    for i,r in enumerate(expected):
        rendered=template.render(messages=[{'role':'user','content':r['prompt']}],add_generation_prompt=True,tools=None)
        assert hashlib.sha256(rendered.encode()).hexdigest()==prov['rendered_prompt_sha256'][i]
    target={(r['match_id'],r['decision_round']):r for r in frozen['records']}
    responses=lines(OUT/'responses.jsonl'); keyed={(r['model'],r['match_id'],r['round']):r for r in responses}
    assert len(keyed)==len(responses)==180
    assert set(keyed)=={(m,*k) for m in MODELS for k in target}
    groups={m:[] for m in (*MODELS,'gpt54','constant','frequency')};validity={}
    for m in MODELS:
        raw=[r for r in responses if r['model']==m];good=[]
        for r in raw:
            if r['p'] is None:continue
            p=r['p'];assert type(p) in (int,float) and math.isfinite(p) and 0<=p<=1
            assert r['finish_reason']=='stop' and r['parse_error'] is None
            obj=json.loads(r['text']);assert set(obj)=={'opponent_belief_my_action_1_probability'} and obj['opponent_belief_my_action_1_probability']==p
            good.append({**r,'target':target[(r['match_id'],r['round'])]['belief_target']})
        groups[m]=good;validity[m]={'requests':len(raw),'valid':len(good),'max_output_tokens':max(r['output_tokens'] for r in raw)}
    gpt=lines(ROOT/'g02_local_20260909/responses.jsonl')
    assert len(gpt)==60
    for r in gpt:
        k=(r['match_id'],r['decision_round']);assert r['prompt']==target[k]['prompt']
        if r['valid']:groups['gpt54'].append({'match_id':k[0],'round':k[1],'p':r['p'],'target':target[k]['belief_target']})
    for k,r in target.items():
        for label,p in [('constant',.5),('frequency',r['frequency_baseline'])]:
            groups[label].append({'match_id':k[0],'round':k[1],'p':p,'target':r['belief_target']})
    common=set.intersection(*[{(r['match_id'],r['round']) for r in groups[m]} for m in (*MODELS,'gpt54')])
    same={m:[r for r in rows if (r['match_id'],r['round']) in common] for m,rows in groups.items()}
    comparisons={}
    for a,b in [('parent','belief_sft_s100'),('base','belief_sft_s100'),('constant','belief_sft_s100'),('gpt54','belief_sft_s100')]:
        comparisons[a+'_to_'+b]=bootstrap(groups[a],groups[b])
    result={'audit_passed':True,'job_id':13228623,'execution':'completed exit0,113seconds','validity':validity,
            'common_keys_n':len(common),'all_valid_metrics':{m:metric(r) for m,r in groups.items()},
            'common_metrics':{m:metric(r) for m,r in same.items()},
            'noninitial_common_metrics':{m:metric([x for x in r if x['round']>1]) for m,r in same.items()},
            'comparisons':comparisons,'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'responses.jsonl',OUT/'provenance.json',ROOT/'g02_inputs.json',ROOT/'g02_local_20260909/responses.jsonl',Path(__file__)]},
            'rows':groups,'limitations':['Same-library tokenizer rendering verified independently from saved hashes.',
            'Untouched base vs structured parent vs bridge s100 are different training stages; only direct-parent delta is incremental bridge effect.',
            'Common continuous reporting interface differs from old binned multi-field SFT; no native-interface retention conclusion.',
            'Six reused fixed development trajectories; not live play, independent confirmation, or mechanism identification.',
            'Base config and index hashes plus snapshot verified, not full base tensor hashes.',
            'NCCL/nanobind teardown warnings after all three model completion markers; no missing output or nonzero Slurm exit.']}
    dest=RUN/'audit_metrics.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    compact={m:{k:v for k,v in metric(r).items() if k not in ('cells','by_round')} for m,r in same.items()}
    print(json.dumps({'audit_passed':True,'validity':validity,'common59':compact,'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
