# Audited result records available in this archive

The Figure 2A file contains 20 match scores for each of the 14 Qwen2.5 seeker/hider cells plotted against ToM2. It is a filtered match-level table, not a full raw trajectory. The `results/audited_live_cells/` directory contains byte-preserved audit JSONs for four completed hider-versus-ToM2 live-play cells: GPT-OSS low, medium and high effort, and Gemma 4 thinking. Each record includes 20 per-match payoffs, the published mean and whole-match bootstrap interval, parser/fallback counts, semantic audit counts, and a hash of the original full CSV. The complete audited round CSVs are released as compressed files in `raw/figure2b_tom2_hider/`; model weights are not included.

The released per-match scores support a self-contained numerical check of the corresponding Figure 2B values:

```bash
python -m pip install -r requirements.txt
python verify_artifact.py
python reproduce_audited_cells.py
```

Figure 2C GPT-5.4 seeker/hider action-only and structured-output matches are also included with per-round CSVs, request journals and audits in `raw/figure2c_gpt54_{seeker,hider}/`. Use `python reanalyse_live_raw.py` to check these and the Figure 2B cells from rounds. Selected row-level development outputs for Figure 3a/b/d/e/f and Figure 4d/e are also included under `raw/`; use `python reanalyse_selected_raw.py --panel 3e` (or another listed panel). See `RAW_DATA_SCOPE.md` for exact files and limits.

`reproduce_audited_cells.py` recomputes mean payoff, the original 20,000-draw whole-match bootstrap interval, and the role-correct action fraction implied by the 50-round zero-sum payoff. It checks these against the released audit records and Figure 2B plotted values. The seeded calculation was tested with Python 3.12.4 and NumPy 1.26.4. The added `reanalyse_live_raw.py` recomputes point estimates from the released full rounds. Neither script reruns inference or repeats the original semantic audit. For those boundaries and the original CSV hashes, see the individual audit records and `DATA_AND_LIMITS.md`.

No incomplete or invalid condition is replaced with a zero-valued score. The `status` column of `data/supplement_plotted_values.csv` records the provenance category for each retained cell; it is not a scientific-success flag.
