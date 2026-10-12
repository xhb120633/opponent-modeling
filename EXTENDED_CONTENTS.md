# Extended contents and provenance

## Adapter map

All adapters use the Qwen2.5-7B-Instruct base. See `models/index.json` for full
hashes and the provenance level of each checkpoint; model loading is not claimed.

| Name | Paper use |
|---|---|
| behavioral_action_2400 / behavioral_structured_300 | Behavioral SFT endpoints, Figure 2/S3; structured parent for later runs |
| five_field_base_200 / five_field_parent_100 | Figure 3d five-field SFT, initial versus behavioral initialization |
| probability_only_500 / state_probability_selected | Figure 3c probability-only versus state-plus-probability |
| known_200 / mixed_200 | Figure 3e/f known-information versus mixed-exposure continuation |
| offline_grpo_100 / 300 / 600 | Figure S4 offline GRPO trajectory |
| iterative_k4_75 / 150 / 225, iterative_k16_75 / 150 / 225 | Figure S5 iteration endpoints |
| recursive_v3_25 | Figure S6 recursive-reward intervention prefix |

Seven SFT tensor files match hashes recorded in the earlier experiment audits.
For historical behavioral action-only and RL adapters, the source is identified
by the archived launcher/evaluation checkpoint path; the release records the
current file hash, not an independently recovered original runtime tensor hash.
Base-model paths in adapter configs are rebound to the public model identifier;
the tensor bytes are unchanged. No tokenizer or full base-model weights are needed
inside the adapter package; obtain the base tokenizer with the base model.

## Evidence groups

- `live_play`: original historical Qwen scaling and standard hider/structured
  model-output CSVs. Files may retain adjacent historical conditions; use the
  paper's plotted-value/source mapping rather than averaging every file together.
- `standard_live`: complete model-output CSVs for 53 standard conditions, each
  hash-bound to its audit; per-match payoffs are recalculated during packaging.
- `behavioral_sft`: original main/opponent-transfer/GPT-transfer endpoint CSVs.
- `fixed_history`: frozen R3/R4/R5 data packages and outputs; GPT supplied-information,
  chain, depth and history packages. These are diagnostic cohorts, not sealed-final data.
- `training`: behavioral action/structured and five-field train/validation examples.
- `rl`: archived round-level light exports, match summaries, iterative evaluation
  exports and recursive intervention rows. Light exports are not full prompt journals.
- `configs`: historical RL training configuration files.

The existing `raw/` directory retains audit-bound T04 and R5 response journals,
GPT intervention requests/replies, ToM2 reasoning hider outputs and GPT live-play
records, with runnable statistical checks. Earlier scope documents describe this
original subset; this document describes the added `evidence/` and `models/` layers.

`SOURCE_MANIFEST.json` records original and released hashes and metadata-only
substitutions for new evidence. Historical audit JSONs may contain original hashes:
those are provenance, not hashes of the anonymized release copy. Identifying account
and storage prefixes are replaced; predictions, targets and numerical results are
not intentionally transformed. Simulator/oracle target fields must not be treated
as original model predictions. The paper-specific audit scripts preserve this distinction.

## Reproduction boundaries

Released train/validation and diagnostic cohorts do not include the sealed-final
partition. A frozen dataset alone is not proof that its training was rerun.
Exact historical environments vary; `MODEL_ENVIRONMENT_SNAPSHOT.md` records one
observed environment, not a universal lockfile. `requirements-model-reference.txt`
is a reference list, not a validated installation recipe. The development TRL
version is not represented as a readily installable PyPI release.

Some API studies and historical RL runs lack full archived prompts or immutable
runtime attestations. The included plotting tables/PDFs cover all paper figures,
but the package does not claim that every figure can be rebuilt end-to-end by a
single command. Use the checkpoint map and source pointers for partial reruns.
