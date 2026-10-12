"""Bounded program diagnostic; no model inference, training or external requests."""
import ast
import copy
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import tomsup.ktom_functions as kt
import tomsup.agent as agents
from tomsup.payoffmatrix import PayoffMatrix

ROOT = Path(__file__).resolve().parent
PARAMS = {'volatility': -2, 'b_temp': -1, 'bias': 0}
MATRIX = PayoffMatrix(name='custom', predefined=np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def scalar_state(s, name):
    node = ast.parse(s, mode='eval').body
    for key in ('internal_states', 'own_states', name):
        assert isinstance(node, ast.Dict)
        found = [v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant) and k.value == key]
        assert len(found) == 1
        node = found[0]
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        assert node.func.value.id == 'np' and node.func.attr in ('float64', 'float32')
        assert len(node.args) == 1 and not node.keywords
        node = node.args[0]
    value = float(ast.literal_eval(node))
    assert np.isfinite(value) and 0 <= value <= 1
    return value

def parse_history(record):
    text = record['prompt'].split('Complete past observations:\n')[1]
    if text == 'none (first round)':
        history = []
    else:
        history = []
        for i, line in enumerate(text.splitlines(), start=1):
            m = re.fullmatch(r'round=(\d+); self_action=([01]); opponent_action=([01]); self_payoff=(-?1)', line)
            assert m and int(m[1]) == i
            history.append(tuple(map(int, m.groups()[1:])))
    assert len(history) == record['decision_round'] - 1
    return history

def replay(history, level, b_role):
    state = kt.init_k_tom(PARAMS, level)
    for a, b, payoff in history:
        state = kt.learning_function(state, PARAMS, b, a, level, b_role, MATRIX)
    logit_self, belief = kt.decision_function(state, PARAMS, b_role, level, MATRIX)
    return state, float(belief), float(kt.inv_logit(logit_self))

def scalar_collision():
    examples = []
    for variance in (0.1, 4.0):
        mean = float(kt.logit(0.7) * np.sqrt(1 + 0.36 * variance))
        logvar = float(np.log(variance))
        current = float(kt.p_op0_fun(mean, logvar))
        outcomes = {}
        for observed in (0, 1):
            nv = kt.p_op_var0_update(mean, logvar, PARAMS['volatility'])
            nm = kt.p_op_mean0_update(mean, nv, observed)
            outcomes[str(observed)] = float(kt.p_op0_fun(nm, nv))
        examples.append({'mean_logit': mean, 'variance': variance, 'log_variance': logvar,
                         'current_belief': current, 'next_belief_by_A_action': outcomes})
    assert abs(examples[0]['current_belief'] - examples[1]['current_belief']) < 1e-14
    gaps = {str(a): abs(examples[0]['next_belief_by_A_action'][str(a)] - examples[1]['next_belief_by_A_action'][str(a)]) for a in (0, 1)}
    assert max(gaps.values()) > 1e-8
    return {'constructed_not_default_reachability_claim': True, 'states': examples, 'next_gaps': gaps}

def reachable_near_collision():
    rows = []
    for seq in itertools.product((0, 1), repeat=10):
        mean, var = 0.0, 0.0
        for a in seq:
            var = kt.p_op_var0_update(mean, var, -2)
            mean = kt.p_op_mean0_update(mean, var, a)
        p = float(kt.p_op0_fun(mean, var))
        nv = kt.p_op_var0_update(mean, var, -2)
        pn = float(kt.p_op0_fun(kt.p_op_mean0_update(mean, nv, 1), nv))
        rows.append({'A_history': list(seq), 'current': p, 'next_after_A1': pn,
                     'mean_logit': float(mean), 'log_variance': float(var)})
    best, count = None, 0
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a['A_history'][-1] != b['A_history'][-1]:
                continue
            gap = abs(a['current'] - b['current'])
            if gap <= .001:
                count += 1
                ng = abs(a['next_after_A1'] - b['next_after_A1'])
                if best is None or ng > best['next_gap']:
                    best = {'a': a, 'b': b, 'current_gap': gap, 'next_gap': ng}
    return {'n_histories': len(rows), 'length': 10, 'matching_pairs': count,
            'tolerance': .001, 'same_last_A_action': True, 'next_A_action': 1,
            'max_gap_example': best, 'is_exact_collision_proof': False}

def main():
    dest = ROOT / 's1_program_reference_20260915.json'
    assert not dest.exists(), 'Preserve prior results'
    source = ROOT / 'eval_job_12878223/corrected_prior_full_history/rounds.csv'
    frozen = json.loads((ROOT / 'g02_inputs.json').read_text(encoding='utf-8'))
    assert sha(source) == frozen['source_hashes']['rounds.csv']
    with source.open(encoding='utf-8', newline='') as f:
        original = list(csv.DictReader(f))
    keyed = {(r['match_id'], int(r['decision_round'])): r for r in original}
    assert len(keyed) == len(original)
    output = []
    for record in frozen['records']:
        mid, t = record['match_id'], record['decision_round']
        orig = keyed[(mid, t)]
        assert orig['opponent_name'] == record['opponent'] and orig['llm_role'] == record['role']
        llm_role = int(orig['llm_player_index'])
        assert llm_role == (0 if record['role'] == 'seeker' else 1)
        b_role = 1 - llm_role
        history = parse_history(record)
        for i, (a, b, payoff) in enumerate(history, start=1):
            past = keyed[(mid, i)]
            assert a == int(past[f'choice_agent{llm_role}'])
            assert b == int(past[f'choice_agent{b_role}'])
            assert payoff == int(float(past['llm_payoff']))
            expected = (1 if a == b else -1) * (1 if llm_role == 0 else -1)
            assert payoff == expected
        archive = orig[f'history_agent{b_role}']
        target_self, target_belief = scalar_state(archive, 'p_self'), scalar_state(archive, 'p_op')
        assert abs(target_belief - record['belief_target']) < 1e-14
        level = int(record['opponent'][-1])
        state, belief, pself = replay(history, level, b_role)
        mapping = float(kt.softmax(kt.expected_payoff_fun(belief, b_role, MATRIX), PARAMS))
        candidates = {str(k): replay(history, k, b_role)[1] for k in range(3)}
        output.append({'match_id': mid, 'round': t, 'role': record['role'], 'level': level,
                       'target_belief': target_belief, 'replay_belief': belief,
                       'target_action_probability': target_self, 'replay_action_probability': pself,
                       'belief_error': abs(belief-target_belief), 'action_probability_error': abs(pself-target_self),
                       'mapping_error': abs(mapping-pself), 'candidate_beliefs': candidates,
                       'candidate_range': max(candidates.values())-min(candidates.values())})
    assert len(output) == 60
    summary = {'n': len(output), 'matches': len({r['match_id'] for r in output})}
    for field in ('belief_error', 'action_probability_error', 'mapping_error'):
        summary['max_' + field] = max(r[field] for r in output)
        summary['mean_' + field] = float(np.mean([r[field] for r in output]))
    summary['replay_passed'] = max(summary['max_belief_error'], summary['max_action_probability_error']) <= 1e-10
    summary['candidate_range_mean'] = float(np.mean([r['candidate_range'] for r in output]))
    summary['candidate_range_max'] = max(r['candidate_range'] for r in output)
    summary['candidate_range_above_0.1_count'] = sum(r['candidate_range'] > .1 for r in output)
    summary['constant_belief_mae_60'] = float(np.mean([abs(r['target_belief']-.5) for r in output]))
    files = [Path(__file__), Path(kt.__file__), Path(agents.__file__), source, ROOT/'g02_inputs.json',
             ROOT.parent/'Hide_and_Seek/LLM_vs_Agents_vllm/run_online_binary_belief_gate.py']
    result = {'summary': summary, 'parameters': PARAMS, 'hashes': {str(p): sha(p) for p in files},
              'rows': output, 'scalar_collision': scalar_collision(), 'reachable_near_collision': reachable_near_collision(),
              'limits': ['Known true level and fixed parameters are privileged relative to G02.',
                         'Same implementation replay, not independent algorithm validation.',
                         'Candidate ranges are not posterior intervals or irreducible error bounds.',
                         'Constructed exact collisions do not prove default-init reachable collisions.']}
    with dest.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('summary', 'scalar_collision', 'reachable_near_collision')}, indent=2))
    assert summary['replay_passed'], 'Saved failure evidence; investigate before interpretation'

if __name__ == '__main__':
    main()
