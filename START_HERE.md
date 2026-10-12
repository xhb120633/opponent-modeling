> Extended edition: [EXTENDED_CONTENTS.md](EXTENDED_CONTENTS.md) supersedes the older subset descriptions below concerning excluded weights/training data and newly added raw evidence.

# Choose a reproducibility starting point

You do not need to run the entire project to inspect one result. Pick the level and panel relevant to your question. Commands below run from the repository root with Python 3.12; only the Figure 2B score bootstrap needs `pip install -r requirements.txt`.

| Start here | Command or location | What it establishes | What it does not establish |
|---|---|---|---|
| Check package integrity | `python verify_artifact.py` | Every released source/data/figure hash and Python syntax | That a model experiment ran correctly |
| Inspect a displayed point | `data/plotted_values.csv`, `data/supplement_plotted_values.csv`, `figures/` | Exact plotted value, label, source pointer and final vector PDF | Independent reanalysis from model output |
| Recalculate Figure 2A/B/C live play | `python reanalyse_live_raw.py` | Figure 2A 14 Qwen2.5 match means, four Figure 2B ToM2 hider cells, and GPT-5.4 seeker/hider action-only versus structured summaries | A new model run or independent simulator-state audit |
| Recalculate four Figure 2B live cells | `python reproduce_audited_cells.py` | Mean payoff, match-bootstrap interval and correct-action rate from released per-match scores | Original replies, parsing or semantic provenance |
| Recalculate Figure 3a/b/d | `python reanalyse_selected_raw.py --panel 3a` (substitute `3b` or `3d`) | Prediction/belief MSE from the included T04 response audit rows | Model inference or independently regenerated simulator targets |
| Recalculate Figure 3e/f | `python reanalyse_selected_raw.py --panel 3e` (or `3f`) | R5 fresh-cohort MSE and opponent-specific point differences from included row-level predictions | New training, target-generation audit or live-play transfer |
| Recalculate Figure 4d/e | `python reanalyse_selected_raw.py --panel 4d` (or `4e`) | Forecast MSE or expected-regret differences from included scored decisions; checks hashes of request/response files | A new API run or the separate best-action agreement audit |
| Check every included row-level panel | `python reanalyse_selected_raw.py --panel all` | All seven panel-level point-estimate checks above | The full paper from raw outputs |
| Trace an experiment back to code | [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) | Historical training/evaluation/plotting entrypoints and prerequisites | That the historical data and weights are all distributed |

The released raw subset is listed in [RAW_DATA_SCOPE.md](RAW_DATA_SCOPE.md). Figure 2A includes paper-visible match-level scores; Figure 2B/2C compressed records include per-round replies and prompts. The selected Figure 3/4 JSONL files preserve model outputs and, for R5 and the GPT intervention, prompts. Paired audit JSONs retain source and consistency checks. A panel not listed for row-level reanalysis can still be inspected in the plotted-value table and final PDF, but this archive does not contain enough observation-level material to recompute it independently.

To start at model inference or training instead, follow the relevant script in `code/` and its prerequisites in [MODEL_ENVIRONMENT_SNAPSHOT.md](MODEL_ENVIRONMENT_SNAPSHOT.md) and [DATA_AND_LIMITS.md](DATA_AND_LIMITS.md). Such a rerun needs the named model/API access, external simulator and, for some historical studies, excluded training splits or adapter weights. It should be labeled a new prospective run rather than a byte-identical reproduction of the archived one.
