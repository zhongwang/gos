# GOS

GenomeOcean-Sentinel (GOS) classifies FASTA records as `AI` or `Natural` using
47 multiscale CPU features and a **distilled GenomeOcean-100M v1.2 student
observer**. The complete observer is published as
[DOEJGI/GenomeOcean-Sentinel](https://huggingface.co/DOEJGI/GenomeOcean-Sentinel).
The original student checkpoint and production decision weights remain in
[models/gos_detector/](models/gos_detector/). Sequences are treated as
input strings; this package performs inference only.

## Quickstart

Use Python 3.10 or newer, Git LFS, and a Python environment with the runtime
dependencies. Run from the repository root:

```bash
git lfs pull
python -m pip install -r requirements.txt

# First run downloads the standalone observer and tokenizer from Hugging Face.
python -m src.gos.cli \
  --input examples/example.fasta \
  --checkpoint models/gos_detector/gos_detector_distilled_genomeocean_100m.pt \
  --output examples/example_output.json \
  --device cpu --dtype float32
```

The default loader uses `AutoModel.from_pretrained(..., trust_remote_code=True)`
with the standalone model's complete backbone, mask-mean pooling, and observation
head. Its tokenizer comes from the same Hub repository. The release is pinned to
revision `0d91aff1293ad48e71335a6963316a2062c090d5`, including the custom code.
No base-model snapshot or in-repo config/tokenizer copy is needed.
The retained `.pt` supplies normalization and maximum-length metadata; inference
weights come from the standalone model's safetensors.

`--model` (alias `--model-dir`) can select another compatible standalone model
or a downloaded standalone snapshot directory. After the initial download,
`HF_HUB_OFFLINE=1` uses the cached standalone release. An uncached offline first
run requires downloading that release first; the `.pt` alone lacks a tokenizer.
`--decision-head` optionally selects a decision JSON; the default is
`decision_head.json` beside the checkpoint. `--batch-size` defaults to 16.
For the production precision mode, use `--device cuda --dtype bfloat16`.
CPU float32 is the portable default; device, precision, and dependency versions
are recorded in output and can affect numerical scores slightly.

## Decision rule

The CPU extractor preserves the production 500-character windows, 250-character
stride, terminal-window inclusion, and 47-feature ordering. It casts features
to float32 before the decision head standardizes them in float64. The original
3248-feature coefficient vector places this block at **index 3201**, beginning
with `multiscale_available`; `multiscale_mean_dimer_entropy` is at index 3207.
The shipped JSON contains exactly the required slices and availability scalar:

```text
base_logit = intercept + ((1 - mean[0]) / scale[0]) * coef[0]
             + ((cpu_features - mean[3201:]) / scale[3201:]) @ coef[3201:]
observer_logit = student_head(mask_mean(backbone_hidden)) * target_std + target_mean
total_logit = base_logit + observer_logit
router_confidence = sigmoid(total_logit)
call = AI if total_logit >= threshold_logit else Natural
```

The observer uses a Linear(768, 1) head, `[PAD]` padding, and truncation to
**200 tokens**. CPU features cover all eligible fragments; the observer sees
the truncated tokenized record.

The default threshold is **−4.535912535032772** in logit space
(probability **0.01060348416857509**). It was derived with the production
`threshold_at_fpr` procedure from the stored `global_probability` and `labels`
arrays, targeting FPR 0.05: 100 of 2000 natural reference records meet the
inclusive threshold. This reuses the existing production reference calibration;
it is **not an independent FPR certification for the student or new datasets**.
See [model documentation](models/gos_detector/README.md) for provenance.

`router_confidence` is the AI score even when the call is `Natural`; it is not
the probability that the selected label is correct. Calls use the calibrated
threshold, not 0.5. The package implements the specified binary decision path;
it does not add a fallback service or an inconclusive routing stage.

## Inputs and output

The FASTA reader removes formatting whitespace and uppercases record strings.
CPU preprocessing splits on characters outside its supported four-character
alphabet and excludes fragments shorter than 500 characters. Records without
an eligible fragment cause an explicit error; no score is invented. Empty
records, duplicate IDs, invalid labels, and non-finite scores are rejected.

A header may contain `true_label=0` (Natural) or `true_label=1` (AI). Labels
are copied to output solely for evaluation and never affect inference; absent
labels produce `true_label: null`. Each output record includes `call`, numeric
`prediction`, `router_confidence`, `reason`, and the component logits.
[The example](examples/README.md) includes labels verified against the source
manifest and actual output from this CLI.

## Python API

```python
from src.gos import GOSDetector

detector = GOSDetector(
    checkpoint="models/gos_detector/gos_detector_distilled_genomeocean_100m.pt",
)
result = detector.scan(sequence)  # your input string
print(result["call"], result["router_confidence"])
```

## Layout and license

- `src/gos/`: student, CPU features, decision head, detector, and CLI.
- `models/gos_detector/`: existing LFS checkpoint and small decision-weight JSON.
- `examples/`: FASTA, verified label manifest, and real inference output.

GOS uses the [Lawrence Berkeley National Laboratory Non-Commercial Use Only
License](LICENSE). Copyright (c) 2026, The Regents of the University of
California, through Lawrence Berkeley National Laboratory. See [NOTICE](NOTICE)
for DOE contract attribution and [LICENSE](LICENSE) for the full terms and
commercial licensing contact. The standalone Hugging Face release includes
its complete weights, configuration, tokenizer, custom modeling code, and the
same LICENSE/NOTICE.
