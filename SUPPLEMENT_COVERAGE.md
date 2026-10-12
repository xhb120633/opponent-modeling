# Completed supplementary coverage

The current S1 has standard/no-thinking models and Gemma thinking/GPT-OSS effort settings across six opponents in each role. S2 has seven Qwen2.5 sizes (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B), both roles and six opponents. All retained conditions are included regardless of effect direction. Historical baseline comparisons are descriptive and not paired causal interventions.

`raw/supplement/<cell>/rounds.csv.gz` contains original round-level model outputs with numerical observations unchanged. `results/supplement_audits/<cell>.json` gives model/runtime identity, prompt-history/parser/checkpoint checks, match scores, fallback counts and raw CSV hashes. Private machine paths are replaced by portable placeholders, with original and released hashes retained separately in the manifest. The six excluded Qwen thinking conditions are not packaged.

`data/supplement_plotted_values.csv` is the current figure table. `data/supplement_plotted_values_previous.csv` retains the earlier table for historical provenance. `python reanalyse_supplement.py` verifies unique match/round keys, role/payoff semantics, complete 20-by-50 coverage, match scores and means for 143 retained cells, and matches all 48 fresh S2 structured means and 20 newly covered S1 hider means to their plotted values. It does not rerun model generation or certify every historical baseline trajectory.

Retained fallback counts include 84/6,000 for Gemma thinking hider, 87/15,000 for non-ToM2 GPT-OSS hider, and 1,184/12,000 for new 0.5B structured scaling. Do not remove these decisions post hoc: changing live actions would change subsequent histories.
