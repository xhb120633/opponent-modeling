"""G03: frozen G02 prompts, base/parent/belief-SFT; no targets in inference file."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ['TOM_PACE_ROOT'])
MODULE = ROOT / 'LLM_towards_ToM_bridge_v1_20260808/LLM_vs_Agents_vllm'
PARENT = ROOT / 'LLM_towards_ToM/Hide_and_Seek/LLM_vs_Agents_vllm/checkpoints/sft_locked_v1_qwen25_7b_structured_tom_progress10_ep5/checkpoint-300'
BRIDGE = MODULE / 'checkpoints/predictive_state_bridge_steps200_v1'
MODEL = ROOT / 'hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
KEY = 'opponent_belief_my_action_1_probability'
SCHEMA = {'type': 'object', 'properties': {KEY: {'type': 'number'}}, 'required': [KEY], 'additionalProperties': False}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def prepare():
    source = Path(__file__).with_name('g02_inputs.json')
    records = json.loads(source.read_text(encoding='utf-8'))['records']
    assert len(records) == 60
    rows = [{k: r[k] for k in ('match_id', 'decision_round', 'prompt')} for r in records]
    assert len({(r['match_id'], r['decision_round']) for r in rows}) == 60
    dump(source.with_name('g03_prompts.json'), {'g02_sha256': sha(source), 'records': rows})

def run(preflight):
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm.sampling_params as sp
    data = json.loads(Path('g03_prompts.json').read_text(encoding='utf-8'))
    assert data['g02_sha256'] == 'b194258629e16411779f5bed107084196c0374263356d08b88fa6a65afd2159c'
    rows = data['records']
    assert len(rows) == 60 and len({(r['match_id'], r['decision_round']) for r in rows}) == 60
    assert all(set(r) == {'match_id', 'decision_round', 'prompt'} for r in rows)
    adapters = {'base': None, 'parent': PARENT, 'belief_sft_s100': BRIDGE / 'checkpoint-100'}
    provenance = {'versions': {p: importlib.metadata.version(p) for p in ('vllm', 'transformers', 'torch', 'peft')},
                  'model_snapshot': str(MODEL), 'input_sha256': sha('g03_prompts.json'), 'code_sha256': sha(__file__),
                  'source_g02_sha256': data['g02_sha256'], 'job_id': os.environ.get('SLURM_JOB_ID'), 'models': {}}
    for name, path in adapters.items():
        files = list(MODEL.glob('*.json')) if path is None else [path / 'adapter_config.json', path / 'adapter_model.safetensors']
        assert files and all(p.is_file() for p in files)
        provenance['models'][name] = {'path': str(path or MODEL), 'hashes': {p.name: sha(p) for p in files}}
    manifest = json.loads((BRIDGE / 'bridge_run_manifest.json').read_text())
    assert manifest['target_mode'] == 'predictive_state'
    assert Path(manifest['init_adapter_path']) == PARENT
    provenance['training_manifest'] = manifest
    train = MODULE / 'data_predictive_state_bridge_v1/predictive_state/train.jsonl'
    with train.open() as f:
        example = json.loads(next(f))
    answer = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
    assert 'opponent_belief_my_action_bin' in answer
    provenance['first_training_assistant'] = answer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in rows]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    assert max(lengths) + 4096 <= 8192
    provenance['prompt_tokens'] = lengths
    provenance['rendered_prompt_sha256'] = [hashlib.sha256(p.encode()).hexdigest() for p in prompts]
    kwargs = dict(temperature=0, top_p=1, max_tokens=4096, seed=20260915)
    if 'structured_outputs' in inspect.signature(SamplingParams).parameters:
        kwargs['structured_outputs'] = sp.StructuredOutputsParams(json=SCHEMA)
        provenance['schema_interface'] = 'structured_outputs'
    else:
        kwargs['guided_decoding'] = sp.GuidedDecodingParams(json=SCHEMA)
        provenance['schema_interface'] = 'guided_decoding'
    sampling = SamplingParams(**kwargs)
    provenance['sampling'] = str(sampling)
    if preflight:
        dump('preflight.json', provenance)
        print('PREFLIGHT_OK', max(lengths), flush=True)
        return
    out = Path('outputs')
    out.mkdir(exist_ok=False)
    dump(out / 'provenance.json', provenance)
    llm = LLM(model=str(MODEL), dtype='bfloat16', tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=8,
              enforce_eager=True, enable_lora=True, max_loras=1, max_cpu_loras=2, max_lora_rank=16)
    with (out / 'responses.jsonl').open('x', encoding='utf-8') as f:
        for idx, (name, path) in enumerate(adapters.items()):
            lora = None if path is None else LoRARequest(name, idx, str(path))
            results = llm.generate(prompts, sampling, lora_request=lora)
            assert len(results) == len(rows)
            for i, (r, result) in enumerate(zip(rows, results)):
                assert result.prompt == prompts[i]
                completion = result.outputs[0]
                p, error = None, None
                try:
                    obj = json.loads(completion.text)
                    assert set(obj) == {KEY}
                    p = obj[KEY]
                    assert isinstance(p, (float, int)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1
                except Exception as exc:
                    p, error = None, repr(exc)
                row = {'model': name, 'match_id': r['match_id'], 'round': r['decision_round'], 'p': p,
                       'text': completion.text, 'parse_error': error, 'finish_reason': completion.finish_reason,
                       'output_tokens': len(completion.token_ids)}
                f.write(json.dumps(row) + '\n')
                f.flush()
            print('MODEL_COMPLETED', name, len(results), flush=True)
    dump(out / 'completion.json', {'rows': 180, 'responses_sha256': sha(out / 'responses.jsonl'), 'provenance_sha256': sha(out / 'provenance.json')})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run(args.preflight)
