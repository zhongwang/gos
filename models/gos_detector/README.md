# GOS detector — distilled GenomeOcean-100M v1.2 student

This is the **production observer** for the GenomeOcean-Sentinel (GOS) v4
detector: a distilled **GenomeOcean-100M v1.2** student (`genomeocean_100m_v12_saturation`)
that replaces the larger NT-2.5B teacher used in early experiments, enabling
high-throughput detection.

## File

`gos_detector_distilled_genomeocean_100m.pt` — a `torch` `state_dict` (Git LFS):

| Key | Value |
|---|---|
| `schema_version` | `gos-v4-stage2-distillation-benchmark/1` |
| `candidate.kind` | `genomeocean` |
| `candidate.label` | `GenomeOcean-100M v1.2 saturation` |
| params | 116,411,905 |
| `max_length` | 200 |
| `target_mean` / `target_std` | observation-logit standardization scalars |

## Loading

The checkpoint records the distilled student's weights on top of the **base
GenomeOcean-100M v1.2** architecture. Build the observer with
`GenomeOceanStudent` (see `production_pipeline_service.py`), load
`model_state_dict`, and apply the scalars. The base weights are on Hugging Face
as `DOEJGI/GenomeOcean-100M-v1.2`.

Because this distilled student replaces the NT-2.5B observer, it **does not
reference** `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`; that teacher
is only used to produce distillation targets and is not part of the released
detector.
