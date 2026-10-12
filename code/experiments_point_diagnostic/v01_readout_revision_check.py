"""CPU-only proposed D prompt simplification; never calls an API."""
import json
import math
from pathlib import Path
from v01_validation import PACK, OUT, sha

TEXT = """Compute only B's current belief probability from its supplied state; no learning/update is requested.
Define sigmoid(x)=1/(1+exp(-x)). State keys use slash-separated dictionary/array paths.
For level0: m=state['own_states/p_op_mean0']; log_v=state['own_states/p_op_var0'];
answer=sigmoid(m/sqrt(1+0.36*exp(log_v))).
For level k>0, for i=0,...,k-1:
q_i=sum over j=0,1,2 of exp(state['own_states/param_var/i/j'])*state['own_states/gradient/i/j']^2;
m_i=state['own_states/p_op_mean/i']; w_i=state['own_states/p_k/i'];
answer=sum over i of w_i*sigmoid(m_i/sqrt(1+0.36*q_i)).
Replace i,j in paths by the numeric indices. p_op_mean entries are logits; param_var entries are LOG variances.
Use only outer own_states entries; opponent_states entries are not needed for this readout.
Return answer as opponent_belief_my_action_1_probability; it is B's belief that A chooses1, NOT B's own action probability.
"""

def readout(row):
    s=row['current']; level=row['level']
    sig=lambda x:1/(1+math.exp(-x))
    if level==0:
        return sig(s['own_states/p_op_mean0']/math.sqrt(1+.36*math.exp(s['own_states/p_op_var0'])))
    return sum(s[f'own_states/p_k/{i}']*sig(s[f'own_states/p_op_mean/{i}']/math.sqrt(1+.36*sum(
        math.exp(s[f'own_states/param_var/{i}/{j}'])*s[f'own_states/gradient/{i}/{j}']**2 for j in range(3)))) for i in range(level))

if __name__=='__main__':
    rows=json.loads((PACK/'items.json').read_text())
    targets={(r['match_id'],r['round']):r['p'] for r in json.loads((PACK/'targets.json').read_text())}
    errors=[abs(readout(r)-targets[(r['match_id'],r['round'])]) for r in rows]
    assert max(errors)<1e-10
    result={'proposed_not_run':True,'n':len(rows),'max_error':max(errors),'readout_instructions':TEXT,
            'source_sha256':sha(Path(__file__)),'new_API_requests':0,
            'scope':'D-only rule simplification; same full current state, target and4096 cap. Requires user decision before rerun.'}
    with (OUT/'proposed_D_revision_cpu_check.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='readout_instructions'},indent=2))
