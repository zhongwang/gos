# GOS detector assets

`gos_detector_distilled_genomeocean_100m.pt` is the shipped distilled
**GenomeOcean-100M v1.2** student checkpoint, tracked with Git LFS. Its weights
are unchanged by the package cleanup (SHA-256
`74b4087d09` is the prefix of its LFS object ID).

| Property | Value |
| --- | --- |
| Checkpoint schema | `gos-v4-stage2-distillation-benchmark/1` |
| Candidate | `genomeocean_100m_v12_saturation` |
| Parameter count | 116,411,905 |
| Selected epoch | 6 |
| Head | Linear(768, 1), after attention-mask mean pooling |
| Maximum tokens | 200 |
| target_mean | −0.000002191271897027036 |
| target_std | 11.3706368339268 |

The base architecture and tokenizer come from `DOEJGI/GenomeOcean-100M-v1.2`,
revision `2326d7b3d02476cb014a768e9f2de617007385ab`. Pass its local snapshot
as `--model`; the checkpoint's embedded training-location metadata is ignored.
`GenomeOceanStudent` loads the base with `local_files_only=True` and
`trust_remote_code=True`, then strictly loads the shipped state dictionary.
The tokenizer pads with `[PAD]`. No training or runtime workflow data is needed.

## Decision head

`decision_head.json` exports only the required values from the production
`distillation_targets.npz` arrays: `global_coef`, `global_scaler_mean`,
`global_scaler_scale`, and `global_intercept`. It includes the source archive's
SHA-256 and the original feature order for provenance.

The original vector has 3248 elements. The complete 47-feature CPU slice starts
at index **3201**, with `multiscale_available`. The feature at index 3207 is
`multiscale_mean_dimer_entropy`, the seventh entry in that slice. The separate
availability contribution uses element 0 and a fixed feature value of 1.0.
The JSON retains float64 values without coefficient rounding. See the root
README for the exact fusion formula.

## Threshold derivation

The production `threshold_at_fpr` function was executed against
`global_probability` and `labels` in the same source archive. For each distinct
natural score it considers that score and its next representable float toward
positive infinity, then selects the smallest threshold whose inclusive `>=`
false-positive rate is at most 0.05. The selected probability is
`0.01060348416857509`: 100 of 2000 natural records, among 5000 total reference
records, meet that threshold. The production clipped logit transform gives
`-4.535912535032772` (clipping is inactive for this value).

This is the production reference threshold, not a threshold fitted on the six
examples and not a newly certified student false-positive rate. Runtime
`router_confidence` is `sigmoid(base_logit + observer_logit)`.

The [runnable example](../../examples/README.md) uses CPU float32. Production's
CUDA bfloat16 inference is available through `--device cuda --dtype bfloat16`;
rounding can differ while the decision formula stays the same.
