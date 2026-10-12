"""Offline T04 provenance, matched outcome metrics and exploratory uncertainty."""
import csv,json,hashlib,re
from pathlib import Path
import numpy as np
from audit_g03_matched import metric,bootstrap,load,sha,lines
from transformers.utils.chat_template_utils import _compile_jinja_template
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'t04_job13242123'
MODELS=['original_qwen','original_parent','base_selected','parent_selected']

def main():
    pre=load(ROOT/'t04_common_preflight_20260915.json')
    assert pre['code_sha256']==sha(ROOT/'t04_common_eval.py')
    g1=load(ROOT/'g01_inputs.json');g2=load(ROOT/'g02_inputs.json')
    assert sha(ROOT/'g01_inputs.json')==pre['g01_sha256']
    assert sha(ROOT/'g03_prompts.json')==pre['g03_sha256']
    source=ROOT/'eval_job_12878223/corrected_prior_full_history/rounds.csv'
    assert sha(source)==g1['source_hashes']['rounds.csv']==g2['source_hashes']['rounds.csv']
    with source.open() as f:original={(r['match_id'],int(r['decision_round'])):r for r in csv.DictReader(f)}
    targets={(r['match_id'],r['round']):r for r in load(ROOT/'v02_package_20260915/targets.json')}
    assert sha(ROOT/'v02_package_20260915/targets.json')=='0359dc91fccd8254517042f85728e1bbf89bd80fb1ef9ed7d8ee855a662ab165'
    from check_s1_program_reference import scalar_state
    for r in g1['records']:
        k=(r['match_id'],r['decision_round']);s=original[k];t=targets[k];b=1-int(s['llm_player_index'])
        assert abs(scalar_state(s[f'history_agent{b}'],'p_self')-t['p_true'])<1e-12
        assert int(s[f'choice_agent{b}'])==t['y']==r['y'] and abs(r['oracle']-t['p_true'])<1e-12
        past=re.findall(r'opponent_action=([01])',r['prompt'])
        assert len(past)==k[1]-1 and abs((1+sum(map(int,past)))/(2+len(past))-t['frequency'])<1e-12
    belief={(r['match_id'],r['decision_round']):r for r in g2['records']}
    for k,r in belief.items():
        s=original[k];b=1-int(s['llm_player_index'])
        assert abs(r['belief_target']-scalar_state(s[f'history_agent{b}'],'p_op'))<1e-12
        assert abs(r['belief_target']-targets[k]['belief_true'])<1e-10
    tc=load(ROOT/'g03_job13228623/tokenizer_config.json');template=_compile_jinja_template(tc['chat_template'])
    for stage,data in [('action',g1['records']),('belief',g2['records'])]:
        for i,r in enumerate(data):
            text=r['prompt']
            if stage=='action':text=text.replace('Report your predictive probability, not a sampled action.','Predict the opponent next action. Respond with exactly one digit: 0 or 1. Do not explain.')
            rendered=template.render(messages=[{'role':'user','content':text}],add_generation_prompt=True,tools=None)
            assert hashlib.sha256(rendered.encode()).hexdigest()==pre[stage+'_rendered_hashes'][i]
    report={'audit_passed':True,'job':13242123,'stages':{},'source_sha256':sha(source),'audit_script_sha256':sha(__file__)}
    for stage in ('action','belief'):
        d=RUN/stage;c=load(d/'completion.json');p=load(d/'provenance.json')
        assert c['rows']==240 and c['responses_sha256']==sha(d/'responses.jsonl') and c['provenance_sha256']==sha(d/'provenance.json')
        assert p['job_id']=='13242123';p['job_id']=pre['job_id'];assert p==pre
        raw=lines(d/'responses.jsonl');assert len(raw)==240
        assert len({(r['model'],r['match_id'],r['round']) for r in raw})==240
        assert {(r['model'],r['match_id'],r['round']) for r in raw}=={(m,*k) for m in MODELS for k in targets}
        groups={m:[] for m in MODELS};validity={}
        for r in raw:
            if not r['valid']:continue
            assert isinstance(r['p'],(float,int)) and 0<=r['p']<=1
            if stage=='belief':
                assert r['finish_reason']=='stop' and r['parse_error'] is None
                assert json.loads(r['text'])=={'opponent_belief_my_action_1_probability':r['p']}
            else:assert 0<=r['digit_mass']<=1.00001
            t=targets[(r['match_id'],r['round'])]
            groups[r['model']].append({**r,'target':t['p_true'] if stage=='action' else belief[(r['match_id'],r['round'])]['belief_target'],'y':t['y']})
        for m in MODELS:
            rs=groups[m];validity[m]={'attempted':60,'valid':len(rs)}
            if stage=='action':validity[m].update(mean_digit_mass=float(np.mean([r['digit_mass'] for r in rs])),min_digit_mass=min(r['digit_mass'] for r in rs))
        for label in ('constant','frequency'):
            groups[label]=[{'match_id':k[0],'round':k[1],'target':t['p_true'] if stage=='action' else belief[k]['belief_target'],'y':t['y'],
                'p':.5 if label=='constant' else t['frequency'] if stage=='action' else belief[k]['frequency_baseline']} for k,t in targets.items()]
        def score(rs):
            out=metric(rs)
            if stage=='action':
                p=np.clip([r['p'] for r in rs],1e-6,1-1e-6);y=np.array([r['y'] for r in rs])
                out.update(brier=float(np.mean([(r['p']-r['y'])**2 for r in rs])),log_loss=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))))
            return out
        pairs={}
        for a,b in [('original_qwen','base_selected'),('original_parent','parent_selected'),('base_selected','parent_selected'),('constant','base_selected'),('constant','parent_selected'),('frequency','parent_selected')]:
            v=bootstrap(groups[a],groups[b])
            if stage=='action':
                yrows=lambda rs:[{**r,'target':r['y']} for r in rs]
                bv=bootstrap(yrows(groups[a]),yrows(groups[b]));v['delta_brier']=bv['delta_mse'];v['brier_ci95']=bv['paired_match_bootstrap95_mse']
            pairs[a+'_to_'+b]=v
        report['stages'][stage]={'validity':validity,'metrics':{m:score(rs) for m,rs in groups.items()},'comparisons':pairs,'rows':groups,'responses_sha256':sha(d/'responses.jsonl')}
    with (RUN/'audit_metrics.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({s:{'metrics':{m:{k:v for k,v in x.items() if k in ('mae','mse','brier','log_loss','centered_pearson','delta_pearson','at_half_count')} for m,x in r['metrics'].items()},'validity':r['validity'],'comparisons':r['comparisons']} for s,r in report['stages'].items()},indent=2))

if __name__=='__main__':main()
