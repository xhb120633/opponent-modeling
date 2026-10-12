"""Dependency-free reanalysis of retained original outputs, not new inference."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,io,json,hashlib
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
entries={r['path']:r for r in manifest['files']}
plotted=list(csv.DictReader((ROOT/'data/supplement_plotted_values.csv').open()))
count=0; matched_s1=0; matched_s2=0
for p in sorted((ROOT/'results/supplement_audits').glob('*.json')):
    d=json.loads(p.read_text());key=d['key'];cell=d['cell']
    assert key not in {f'original_reasoning_{n}' for n in range(6)}
    raw=ROOT/'raw/supplement'/key/'rounds.csv.gz'
    meta=entries[raw.relative_to(ROOT).as_posix()]
    data=gzip.decompress(raw.read_bytes())
    assert hashlib.sha256(data).hexdigest()==meta['release_uncompressed_sha256']
    assert meta['source_sha256']==d['csv_sha256']
    rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    assert len(rows)==d['n_rows']==1000
    scores=defaultdict(float);rounds=defaultdict(set)
    for row in rows:
        repeat=int(row['repeat_idx']);round_id=int(row['round'])
        assert round_id not in rounds[repeat],(key,repeat,round_id)
        rounds[repeat].add(round_id);payoff=float(row['llm_payoff'])
        llm_index=int(row['llm_player_index']);opponent_index=int(row['opponent_player_index'])
        a=int(row[f'choice_agent{llm_index}']);b=int(row[f'choice_agent{opponent_index}'])
        expected=(1 if a==b else -1) if cell['role']=='seeker' else (1 if a!=b else -1)
        assert payoff==expected,(key,repeat,round_id)
        scores[repeat]+=payoff
    assert sorted(scores)==list(range(20)) and all(len(r)==50 for r in rounds.values())
    assert [scores[k] for k in range(20)]==d['match_scores']
    mean=sum(scores.values())/20
    assert abs(mean-d['mean_final_payoff'])<1e-9
    candidates=[r for r in plotted if r['role']==cell['role'] and r['opponent']==cell['opponent'] and key+'.json' in r['source']]
    for r in candidates:
        if r['figure']=='S2':
            assert abs(float(r['structured_payoff'])-mean)<1e-9
            assert abs(float(r['value'])-(mean-float(r['baseline_payoff'])))<1e-9
            matched_s2+=1
        elif r['figure']=='S1':
            assert abs(float(r['value'])-mean)<1e-9
            if cell['opponent']!='ToM2':matched_s1+=1
    count+=1
assert count==143 and matched_s2==48 and matched_s1==20,(count,matched_s1,matched_s2)
print('PASS: 143 retained cells / 143,000 original decisions; 48 new S2 contrasts and 20 new S1 hider means verified.')
print('Archived arithmetic/provenance checks, not model reruns or causal mentalizing validation.')
