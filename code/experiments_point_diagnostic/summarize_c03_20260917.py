"""Supplement terminal C03 audit with same-valid-key comparisons; CPU only."""
import json,collections,re
import numpy as np
import gpt_chain_elicitation_20260917 as b
import gpt_depth_memory_20260917 as depth
import closure_gpt_20260917 as c
root=c.ROOT;out=c.OUT
pre=b.read(c.PACK/'gpt_preflight.json');assert all(b.sha(p)==h for p,h in pre['hashes'].items())
done=b.read(out/'completed.json');assert all(b.sha(out/n)==h for n,h in done['hashes'].items())
responses=b.rows(out/'responses.jsonl');tasks=b.read(c.TASKS);assert len(responses)==len(tasks)==804
by={b.key(r):r for r in responses};assert len(by)==804
truth={(r['match_id'],r['round']):r for r in b.read(c.PACK/'rows.json')}
strong={(r['match_id'],r['round'],r['target']):r['p'] for r in b.read(root/'closure_baseline_20260917/predictions.json')}
old={(r['match_id'],r['round']):r for r in b.rows(root/'gpt_depth_retry_merged_20260917/responses.jsonl') if r['condition']=='L2_N'}
for t in tasks:
    y=truth[t['match_id'],t['round']];arm=t['condition'];r=by[b.key(t)]
    if arm.startswith('rep_'):
        assert y['rep'] in (1,2,3,4) and y['split']=='test'
        assert t['messages']==depth.messages(y,arm[4:],[]) and t['schema']==depth.schema(arm[4:])
    else:
        f,a=arm.split('_');target='belief_true' if f=='b' else 'p_true'
        if a=='self':expected=old[t['match_id'],t['round']]['parsed'][depth.B if f=='b' else depth.Q]
        elif a=='true':expected=y[target]
        else:
            donor=truth[t['donor_match_id'],t['round']];assert donor['role']==y['role'] and donor['match_id']!=y['match_id']
            expected=donor[target]
        assert t['supplied']==expected
        assert 'No information about its reliability is provided.' in t['messages'][0]['content']
    if r['valid']:assert c.parse(t,r['text'])==r['parsed'] and r['model']==b.MODEL
def metric(rs,ps,target='p_true'):
    yy=[r[target] for r in rs];return {'n':len(rs),'mae':float(np.mean(np.abs(np.array(ps)-yy))),'mse':float(np.mean((np.array(ps)-yy)**2)),'r':b.corr(yy,ps)}
def metrics(keys,arm):
    yy=[truth[k] for k in keys];rr=[by[(arm,*k)] for k in keys];p=[r['parsed'][depth.Q] for r in rr]
    result={'q':metric(yy,p),'constant':metric(yy,[.5]*len(yy)),'frequency':metric(yy,[(1+sum(h[1] for h in y['history']))/(2+len(y['history'])) for y in yy]),'supervised':metric(yy,[strong[(*k,'p_true')] for k in keys])}
    regret=[]
    for y,r in zip(yy,rr):
        q=y['p_true'];a=r['parsed']['own_action'];pay=(1 if y['role']=='seeker' else -1)*(2*(q if a else 1-q)-1);regret.append(max(0,abs(2*q-1)-pay))
    result['regret']=float(np.mean(regret));result['optimal_action_n']=sum(v<1e-10 for v in regret)
    if arm=='rep_L2_N':
        result['belief']=metric(yy,[r['parsed'][depth.B] for r in rr],'belief_true')
        result['belief_constant']=metric(yy,[.5]*len(yy),'belief_true')
        result['belief_frequency']=metric(yy,[(1+sum(h[0] for h in y['history']))/(2+len(y['history'])) for y in yy],'belief_true')
        result['belief_supervised']=metric(yy,[strong[(*k,'belief_true')] for k in keys],'belief_true')
    return result
groups={};validkeys={}
for arm in sorted({r['condition'] for r in responses}):
    keys=sorted((r['match_id'],r['round']) for r in responses if r['condition']==arm and r['round']>1 and r['valid']);validkeys[arm]=set(keys)
    groups[arm]={'all':metrics(keys,arm)}
    for role in ('seeker','hider'):
        for l in (0,1,2):
            kk=[k for k in keys if truth[k]['role']==role and truth[k]['level']==l];groups[arm][f'{role}_ToM{l}']=metrics(kk,arm)
rng=np.random.default_rng(20260917);contrasts={}
for a,z in [('b_true','b_self'),('b_true','b_shuffled'),('q_true','q_self'),('q_true','q_shuffled'),('rep_L2_N','rep_L1_N')]:
    keys=sorted(validkeys[a]&validkeys[z]);aa=metrics(keys,a);zz=metrics(keys,z);deltas={}
    for mid in sorted({k[0] for k in keys}):
        kk=[k for k in keys if k[0]==mid];am=metrics(kk,a);zm=metrics(kk,z)
        deltas[mid]=[am['q']['mse']-zm['q']['mse'],am['regret']-zm['regret']]
    values=np.array(list(deltas.values()));draws=np.array([values[rng.integers(0,len(values),len(values))].mean(axis=0) for _ in range(5000)])
    contrasts[a+'-'+z]=dict(common_points=len(keys),matches=len(deltas),a=aa,z=zz,match_mean_mse_regret_delta=values.mean(axis=0).tolist(),ci=np.quantile(draws,[.025,.975],axis=0).T.tolist())
result=dict(passed=True,requests=804,valid=sum(r['valid'] for r in responses),failures=dict(collections.Counter(str(r.get('error_type'))+':'+str(r.get('status_code')) for r in responses if not r['valid'])),groups=groups,common_valid_contrasts=contrasts,limits=['conditional valid subsets may be selected by server failures','six intervention matches only','supervised comparator has different learning prior'],hashes={str(p):b.sha(p) for p in (c.TASKS,out/'responses.jsonl',out/'audit_results.json',root/'closure_baseline_20260917/predictions.json')})
b.dump(out/'supplementary_audit.json',result)
print(json.dumps({'arms':{a:v['all'] for a,v in groups.items()},'contrasts':{a:{k:v[k] for k in ('common_points','match_mean_mse_regret_delta','ci')} for a,v in contrasts.items()}},indent=2))
