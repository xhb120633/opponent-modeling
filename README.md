# Theory of mind in strategic interaction

**Beyond Successful Play: Diagnosing Opponent Modeling in Large Language Models**  
Hanbo Xie, Yue Zhao and Robert C. Wilson

Code and data for the preprint on recursive opponent modeling in strategic interaction. This repository follows the paper's sequence: live behavior, prediction and belief diagnostics, supervised learning, then reinforcement-learning interventions. It includes original model outputs, numerical figure inputs, experiment entrypoints and 18 evaluated SFT/RL LoRA adapters. It does not include base-model weights or proprietary API access.

## Choose your starting point

1. **Read the results:** start with `figures/` and `data/` alongside the preprint. Figures S1/S2 include the completed six-opponent reasoning comparison and seven-size Qwen2.5 structured-output comparison.
2. **Check files:** run `python verify_artifact.py`. It checks the released hashes and Python syntax without model calls.
3. **Reanalyse original outputs:** run `python reanalyse_live_raw.py`, `python reanalyse_selected_raw.py --panel all`, `python reanalyse_extended.py`, and `python reanalyse_supplement.py`. These recompute supported statistics from model outputs, not just plotted tables. See [raw-data scope](RAW_DATA_SCOPE.md) and [supplement coverage](SUPPLEMENT_COVERAGE.md).
4. **Recover observations/training examples:** run `python materialize.py evidence fixed_history`; available groups also include `training`, `live_play`, `standard_live`, `behavioral_sft`, `rl`, and `configs`. Recent original CSVs are gzip-compressed under `raw/supplement/`. The manifest records original hashes and any path-metadata substitutions.
5. **Evaluate an adapter:** run `python materialize.py list`, then `python materialize.py adapter mixed_200` (or another listed checkpoint). Lossless 4 MiB pieces reconstruct a normal PEFT adapter; no precision conversion was performed.
6. **Run a new model experiment:** use [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) and [MODEL_ENVIRONMENT_SNAPSHOT.md](MODEL_ENVIRONMENT_SNAPSHOT.md). Configure model/API access and data/checkpoint paths. Historical scripts are not a universally tested one-command GPU installer.

## Environments and limits

Offline standard-library reanalysis needs Python 3.10 or newer. Install `requirements.txt` for the NumPy-based bootstrap helper. `requirements-model-reference.txt` and the runtime snapshot document the separate historical GPU stacks; installing the offline requirements does not install model runtimes or CUDA.

Original outputs and supervised targets are distinct. Archived oracle intervention arms are explicitly labeled and are not autonomous model predictions. Historical controls can be unpaired; live histories depend on actions. Random fallback decisions remain in deployed-system scores. Qwen thinking is excluded from the reported comparisons and this release. An integrity or statistical check does not establish causal mentalizing, training-seed stability, or byte-identical GPU reproduction.

The preprint PDF/source is distributed separately; its code-availability link will be updated after this repository is published. Base models and external dependencies retain their own terms. No blanket license for third-party weights is implied; a code license is not assigned in this preparation.

## Provenance

`SOURCE_MANIFEST.json` covers code, data, model pieces and figures. `results/supplement_audits/` preserves the audit evidence for 143 retained live conditions (143,000 decisions), including the recent scaling/reasoning additions. Some older comparator and training/runtime evidence has narrower provenance than these recent output audits; the paper and contents documents state those boundaries.
