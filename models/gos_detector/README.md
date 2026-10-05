# GOS detector assets

The complete distilled **GenomeOcean-100M v1.2** student observer is published
as [DOEJGI/GenomeOcean-Sentinel](https://huggingface.co/DOEJGI/GenomeOcean-Sentinel),
revision `7ad672818571cb63f7646539d60414e2d271b90b`, and is the **single source
of truth** for the model weights, tokenizer, and metadata. No checkpoint is
stored in this git repository.

The original `.pt` file that the standalone release was built from (SHA-256
prefix `74b4087d09`, 116,411,905 parameters, schema
`gos-v4-stage2-distillation-benchmark/1`, candidate
`genomeocean_100m_v12_saturation`, selected epoch 6) is preserved for provenance
in the gitignored `models_archive/` directory next to this repo; it is not
tracked and not required at runtime.

| Property | Value |
| --- | --- |
| Standalone model | `DOEJGI/GenomeOcean-Sentinel` |
| Pinned revision | `7ad672818571cb63f7646539d60414e2d271b90b` |
| Architecture | `GOSStudentForObserver` (Mistral encoder + head) |
| Parameter count | 116,411,905 |
| Head | Linear(768, 1), after attention-mask mean pooling |
| Maximum tokens | 200 |
| target_mean | −0.000002191271897027036 |
| target_std | 11.3706368339268 |

`GenomeOceanStudent.from_checkpoint` loads this standalone observer with
`AutoModel.from_pretrained(..., trust_remote_code=True)`; the normalization
scalars and max length come from the published `config.json`, so no base
snapshot or in-repo `.pt` is consulted. `--model` optionally selects a local
standalone snapshot or a compatible Hub repository; the default is the pinned
release above. A first run downloads the standalone model and tokenizer; cached
runs can use `HF_HUB_OFFLINE=1`.

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
