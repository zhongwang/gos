# GenomeOcean-Sentinel (GOS)

GOS classifies DNA sequences in FASTA files as `AI` or `Natural`. Use it to
assess whether each record appears AI-generated or natural, with an AI score
and a classification for each sequence. This package runs inference only.

## Quickstart

Requires **Python 3.10+** and the dependencies in `requirements.txt`.
Run from the repository root:

```bash
python -m pip install -r requirements.txt

python -m src.gos.cli \
  --input examples/example.fasta \
  --output examples/example_output.json \
  --device cpu --dtype float32
```

The first run downloads the model and tokenizer from
[DOEJGI/GenomeOcean-Sentinel](https://huggingface.co/DOEJGI/GenomeOcean-Sentinel),
the **single source of truth** for the observer. No model checkpoint is stored
in this repository. The default release is pinned to revision
`7ad672818571cb63f7646539d60414e2d271b90b`, including its custom modeling code,
which the loader runs with `trust_remote_code=True`.

## Usage

Use `--input` to select a FASTA file and `--output` to choose the JSON output
path. The output directory must already exist; an existing output file is
replaced.

| Option | Values and default |
| --- | --- |
| `--device` | `cpu` (default), `cuda`, or `cuda:N` |
| `--dtype` | `float32` (default) or `bfloat16`; `bfloat16` requires CUDA |
| `--batch-size` | Positive integer; default `16` |
| `--model` / `--model-dir` | Compatible standalone Hugging Face model ID or local snapshot directory; default `DOEJGI/GenomeOcean-Sentinel` |
| `--decision-head` | Decision JSON path; default `models/gos_detector/decision_head.json` |

For CUDA inference with bfloat16, use `--device cuda --dtype bfloat16`.
After downloading the default release, set `HF_HUB_OFFLINE=1` to use its cached
files offline. Device, precision, and dependency versions can slightly change
scores; the output records these runtime settings.

### Python API

Scan the first record in the example FASTA:

```python
from src.gos import GOSDetector
from src.gos.cli import read_fasta

_, sequence, _ = next(read_fasta("examples/example.fasta"))
detector = GOSDetector()  # loads DOEJGI/GenomeOcean-Sentinel by default
result = detector.scan(sequence)
print(result["call"], result["router_confidence"])
```

## Decision rule

GOS combines **47 CPU features** with a distilled GenomeOcean observer.
The CPU features summarize eligible fragments across the sequence; the
observer's tokenized input is truncated to **200 tokens**.

```text
total_logit = base_logit + observer_logit
router_confidence = sigmoid(total_logit)
call = AI if total_logit >= threshold_logit else Natural
```

The default `threshold_logit` is **-4.535912535032772**, equivalent to an AI
score threshold of **0.01060348416857509**. This calibration does not guarantee
a false-positive rate on new datasets.

`router_confidence` is the AI score, not the probability that the chosen label
is correct. Classification uses the threshold above, not 0.5.

## Inputs and output

The FASTA reader removes formatting whitespace and uppercases sequences.
CPU preprocessing splits sequences at characters outside `A`, `C`, `G`, and
`T`, then excludes fragments shorter than **500 bases**. Each record must
contain at least one eligible fragment. Empty records, duplicate IDs, invalid
labels, and non-finite scores cause errors.

Headers may include `true_label=0` (Natural) or `true_label=1` (AI). Labels are
copied to the output for evaluation and **never affect inference**. Missing
labels appear as `true_label: null`.

The output JSON includes model, threshold, and runtime metadata, plus a
`records` array. Each record contains:

| Field | Meaning |
| --- | --- |
| `id`, `length` | FASTA ID and sequence length after whitespace removal |
| `true_label` | Optional input label: `0`, `1`, or `null` |
| `call`, `prediction` | `Natural` / `0` or `AI` / `1` |
| `router_confidence` | AI score from the combined logit |
| `reason` | `total_logit_below_threshold` or `total_logit_at_or_above_threshold` |
| `base_logit`, `observer_logit`, `total_logit` | CPU contribution, observer contribution, and their sum |

Example record copied from the saved
[example output](examples/example_output.json), which was generated with an
earlier model revision:

```json
{
  "id": "record_1",
  "true_label": 0,
  "length": 1395,
  "prediction": 0,
  "call": "Natural",
  "router_confidence": 3.833126964630078e-09,
  "reason": "total_logit_below_threshold",
  "base_logit": -0.13143123665181666,
  "observer_logit": -19.248153686523438,
  "total_logit": -19.379584923175255
}
```

## License

GOS uses the [Lawrence Berkeley National Laboratory Non-Commercial Use Only
License](LICENSE). Copyright (c) 2026, The Regents of the University of
California, through Lawrence Berkeley National Laboratory. See [NOTICE](NOTICE)
for DOE contract attribution and [LICENSE](LICENSE) for the full terms and
commercial licensing contact.
