> Extended edition: [EXTENDED_CONTENTS.md](EXTENDED_CONTENTS.md) supersedes the older subset descriptions below concerning excluded weights/training data and newly added raw evidence.

# Model runtime environment snapshot

The following versions were read from the PACE `scaling_up_RL` environment on 25 September 2026. The frozen live-play launchers use that environment. This is an observed environment snapshot, not a universal lockfile for the paper's historically separate runs. A fresh installation also needs compatible NVIDIA drivers, CUDA libraries, access to the named model revision, and the simulator's own dependencies. No credentials or model weights are distributed.

| Component | Observed version |
|---|---|
| Python | 3.10.20 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| matplotlib | 3.10.9 |
| tomsup | 1.4.1 |
| torch | 2.10.0 |
| transformers | 5.5.3 |
| vllm | 0.19.0 |
| accelerate | 1.8.1 |
| datasets | 4.8.4 |
| peft | 0.18.1 |
| trl | 1.3.0.dev0 |
| openai | 2.24.0 |

The reader-facing `requirements.txt` is intentionally narrower: it pins the NumPy version on which `reproduce_audited_cells.py` was tested. It supports an independent statistical recalculation from the released per-match scores, not a GPU rerun. Do not use the model-runtime snapshot to claim byte-identical reconstruction of every historical experiment.
