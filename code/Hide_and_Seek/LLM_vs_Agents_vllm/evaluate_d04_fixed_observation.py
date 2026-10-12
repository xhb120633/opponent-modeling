"""Missing A/C arms only; B is immutable audited live output. Development only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from belief_transition_prompt import build_belief_transition_prompt
from build_full_history_transition_sft import append_history


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prompt(row, prior, history, full=True):
    obs = history[-1] if history else (None, None, None)
    base = build_belief_transition_prompt(role=row.llm_role,
        decision_round=int(row.decision_round), previous_belief_p1=prior,
        previous_self_action=obs[0], previous_opponent_action=obs[1], previous_payoff=obs[2])
    return append_history(base, history) if full else base


def load_source(source):
    for line in (source / 'output_sha256.txt').read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert digest(source / Path(filename).name) == expected
    rows = pd.read_csv(source / 'rounds.csv')
    oracle = pd.read_csv(source / 'oracle_rounds.csv')
    rows = rows.merge(oracle[['match_id', 'decision_round', 'oracle_tom_p_self']],
        on=['match_id', 'decision_round'], validate='one_to_one')
    traces = pd.read_csv(source / 'prompt_traces.csv').set_index(['match_id', 'decision_round'])
    assert len(rows) == len(traces) == 1800 and rows.match_id.nunique() == 180
    assert set(rows.seed_base) == {61050001}
    assert set(rows.llm_role) == {'hider', 'seeker'}
    assert set(rows.opponent_name) == {'ToM0', 'ToM1', 'ToM2'}
    assert not rows.duplicated(['match_id', 'decision_round']).any() and traces.index.is_unique
    for mid, group in rows.groupby('match_id', sort=False):
        history, prior = [], .5
        assert list(group.decision_round) == list(range(1, 11))
        for row in group.itertuples():
            trace = traces.loc[(mid, row.decision_round)]
            assert trace['prompt'] == prompt(row, prior, history)
            assert abs(trace.prior_belief_opponent_action_1_probability - prior) < 1e-12
            assert abs(trace.predicted_opponent_action_1_probability - row.predicted_opponent_action_1_probability) < 1e-12
            prior = row.predicted_opponent_action_1_probability
            history.append(observation(row))
    return rows


def observation(row):
    return (int(getattr(row, f'choice_agent{int(row.llm_player_index)}')),
            int(row.opponent_action), int(row.llm_payoff))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--stage', choices=['parent', 'corrected'], required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    rows = load_source(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / (args.stage + '.jsonl')
    assert not dest.exists(), 'Never overwrite a run'
    hashes = {name: digest(args.source / name) for name in
              ['rounds.csv', 'oracle_rounds.csv', 'prompt_traces.csv', 'output_sha256.txt']}
    parents = {}
    if args.stage == 'corrected':
        parent_manifest = json.loads((args.output / 'parent_manifest.json').read_text())
        assert parent_manifest['source_hashes'] == hashes
        assert parent_manifest['smoke'] == args.smoke
        assert digest(args.output / 'parent.jsonl') == parent_manifest['output_sha256']
        for line in (args.output / 'parent.jsonl').read_text().splitlines():
            r = json.loads(line)
            parents[(r['match_id'], r['decision_round'])] = r
        assert len(parents) == 1800
    if args.smoke:
        def predict(text):
            return .25 + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) / 0xffffff * .5, {}
        generator = None
    else:
        from vllm_binary_batcher import BinaryVLLMBatchedGenerator
        generator = BinaryVLLMBatchedGenerator(model_name='Qwen/Qwen2.5-7B-Instruct',
            lora_path=args.adapter, max_num_seqs=1, max_batch_size=1, max_model_len=4096)
        predict = generator.predict_one
    count, calls, selected_cells = 0, 0, set()
    with dest.open('x', encoding='utf-8') as f:
        for mid, group in rows.groupby('match_id', sort=False):
            history, parent_prior, own_prior, oracle_prior = [], .5, .5, .5
            cell = (group.iloc[0].llm_role, group.iloc[0].opponent_name)
            selected = cell not in selected_cells
            selected_cells.add(cell)
            for row in group.itertuples():
                conditions = [('parent', parent_prior, False)] if args.stage == 'parent' else [
                    ('A_parent_prior', parent_prior, True), ('C_oracle_prior', oracle_prior, True),
                    ('B_reused_self', own_prior, True)]
                for arm, prior, full in conditions:
                    text = prompt(row, prior, history, full)
                    if arm == 'B_reused_self':
                        prob, diag = row.predicted_opponent_action_1_probability, {'reused_job': 12878223}
                    else:
                        prob, diag = predict(text)
                        calls += 1
                    assert math.isfinite(prob) and 0 <= prob <= 1
                    record = dict(match_id=mid, decision_round=int(row.decision_round),
                        llm_role=row.llm_role, opponent_name=row.opponent_name, arm=arm,
                        prior=float(prior), prompt=text, p=float(prob), oracle=float(row.oracle_tom_p_self),
                        opponent_action=int(row.opponent_action), diagnostics=diag, repeat=0)
                    f.write(json.dumps(record) + '\n')
                    count += 1
                    if args.stage == 'corrected' and selected and row.decision_round == 5:
                        for rep in range(1, 4):
                            rp, rd = predict(text)
                            assert math.isfinite(rp) and 0 <= rp <= 1
                            f.write(json.dumps(dict(record, p=float(rp), diagnostics=rd, repeat=rep)) + '\n')
                            count += 1
                            calls += 1
                    if arm == 'parent':
                        parent_prior = prob
                if args.stage == 'corrected':
                    parent_record = parents[(mid, row.decision_round)]
                    assert parent_record['prompt'] == prompt(row, parent_prior, history, False)
                    parent_prior = parent_record['p']
                own_prior = row.predicted_opponent_action_1_probability
                oracle_prior = row.oracle_tom_p_self
                history.append(observation(row))
            f.flush()
    if generator:
        generator.close()
    assert count == (1800 if args.stage == 'parent' else 5454)
    manifest = dict(stage=args.stage, smoke=args.smoke, source_hashes=hashes,
        output_sha256=digest(dest), rows=count, inference_calls=calls,
        adapter=args.adapter, final_opened=False, training=False,
        parent_interface='scalar_previous_observation_recursive', corrected_interface='full_observable_history',
        repeats='first match per role/opponent cell, round5, 3 fresh calls per A/B/C')
    if not args.smoke:
        manifest['adapter_hashes'] = {name: digest(Path(args.adapter) / name)
            for name in ['adapter_model.safetensors', 'adapter_config.json']}
    (args.output / (args.stage + '_manifest.json')).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
