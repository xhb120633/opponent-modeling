"""Append all past observable rounds to the existing model-prior interface."""
import argparse
import json
from pathlib import Path

import pandas as pd
from build_belief_transition_sft_dataset import build_examples, sha256

VERSION = 'model_prior_full_observable_history_v1'


def append_history(base_prompt, history):
    lines = ['Complete observable history (past rounds only):']
    lines.extend(f'round={i}; self_action={a}; opponent_action={b}; payoff={r}'
                 for i, (a, b, r) in enumerate(history, 1))
    if not history:
        lines.append('none (first decision)')
    return base_prompt + '\n' + '\n'.join(lines)


def build(source, horizon=10):
    assert 1 <= horizon <= 50
    assert set(source.opponent_name) == {'ToM0', 'ToM1', 'ToM2'}
    assert set(source.llm_role) == {'hider', 'seeker'}
    assert set(source.seed_base.astype(int)) == {61010001}, 'Only approved development source'
    subset = source[source.decision_round.le(horizon)].copy()
    examples = build_examples(subset, state_source='model')
    lookup = {(row['metadata']['match_id'], row['metadata']['decision_round']): row
              for split in examples.values() for row in split}
    for match_id, match in subset.groupby('match_id'):
        history = []
        for _, row in match.sort_values('decision_round').iterrows():
            record = lookup[(str(match_id), int(row.decision_round))]
            record['prompt'] = append_history(record['prompt'], history)
            record['messages'][0]['content'] = record['prompt']
            record['metadata'].update(prompt_version=VERSION, observable_history_length=len(history),
                prior_source='frozen_parent_model_rollout', no_oracle_prior=True)
            assert len(history) == int(row.decision_round)-1
            history.append((int(row[f'choice_agent{int(row.llm_player_index)}']),
                            int(row.opponent_action), int(row.llm_payoff)))
    train = {r['metadata']['match_id'] for r in examples['train']}
    val = {r['metadata']['match_id'] for r in examples['val']}
    assert not train.intersection(val)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--tokenizer-check', action='store_true')
    args = parser.parse_args()
    source = pd.read_csv(args.source)
    examples = build(source, args.horizon)
    # Current/future actions and probabilities cannot affect the current prompt.
    changed = source.copy()
    changed.loc[changed.decision_round.ge(args.horizon), 'opponent_action'] = 1-changed.loc[changed.decision_round.ge(args.horizon), 'opponent_action']
    changed.loc[changed.decision_round.ge(args.horizon), 'predicted_opponent_action_1_probability'] = .123
    check = build(changed, args.horizon)
    assert all(a['prompt'] == b['prompt'] for s in examples for a,b in zip(examples[s], check[s]))
    lengths = []
    if args.tokenizer_check:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
        for records in examples.values():
            for row in records:
                rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False)
                length = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
                lengths.append(length)
                assert length < 4096, 'Reject truncation of observable history'
    output = args.output / 'soft_realized_action'
    output.mkdir(parents=True, exist_ok=False)
    for split, records in examples.items():
        with (output / f'{split}.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'prompt_version': VERSION, 'source_sha256': sha256(args.source),
        'source': str(args.source), 'horizon': args.horizon, 'final_opened': False,
        'prior_source': 'frozen parent predictions, not oracle or new on-policy predictions',
        'split': 'complete matches, validation iff repeat_idx % 5 == 0',
        'target': 'oracle soft probability clipped to [0.05,0.95], unchanged',
        'summary': {s: {'rows':len(rs), 'matches':len({r['metadata']['match_id'] for r in rs})} for s,rs in examples.items()},
        'current_future_prompt_invariance_passed': True,
        'max_tokens': max(lengths) if lengths else None,
        'output_sha256': {s:sha256(output / f'{s}.jsonl') for s in examples}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
