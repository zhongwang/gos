# GenomeOcean-Sentinel (GOS)

GenomeOcean-Sentinel (GOS) is a high-throughput detector for AI-generated and
synthetic DNA sequences. It uses a frozen biological language model as an
observer and measures the **positional attention variance (pAV)** of a sequence.
AI-generated DNA shows a characteristic flattening of pAV relative to evolved
(natural) sequences, which GOS exploits to flag synthetic sequences with high
sensitivity and low false-positive rates.

This repository contains the GOS **v4 detector** source, an example, and the
documentation needed to run it. Trained model checkpoints are published
separately (see [Models](#models)).

## License

GenomeOcean-Sentinel (GOS) is licensed under the
[Lawrence Berkeley National Laboratory Non-Commercial Use Only License](LICENSE).
Copyright (c) 2026, The Regents of the University of California, through Lawrence
Berkeley National Laboratory ("Berkeley Lab"), subject to receipt of any required
approvals from the U.S. Dept. of Energy. All rights reserved. Redistribution and
use are permitted only for **non-commercial** purposes under the conditions in
[LICENSE](LICENSE). A separate commercial use license is available from Berkeley
Lab at IPO@lbl.gov. See [NOTICE](NOTICE) for the DOE contract attribution.

## Installation

```bash
pip install -r requirements.txt
```

Runtime dependencies: `numpy`, `scikit-learn`, `joblib`, `torch`, `transformers`.

## Quickstart

The v4 detector scans FASTA records and writes a JSON or TSV projection:

```bash
python -m src.v4.detect \
  --input examples/example.fasta \
  --model-dir /path/to/checkpoint \
  --format json \
  --out results.json
```

Each record is labelled `AI`, `natural`, or `inconclusive`.

### Getting a checkpoint

A trained GOS v4 checkpoint is a directory containing the saved detector
artifacts (`sequence.joblib`, `cds.joblib`, `router_cpu.joblib`, and, for the
full detector, `pav/model.joblib` and `router_full.joblib`). Checkpoints are
published on Hugging Face; see the model card for the download and the exact
`--model-dir` to use. For a CPU-only run, use a CPU checkpoint. For the full
detector (with the pAV rescue observer), use a full checkpoint and ensure a
CUDA device and `transformers` are available.

## Python API

```python
from src.v4 import GOSV4Detector

# Instantiate from a saved checkpoint directory.
detector = GOSV4Detector.from_checkpoint("/path/to/checkpoint")
result = detector.scan("ATGCTAGCTAGCTAGCTAGCTAGCTAGC...", source_id="seq1")
print(result.prediction)  # 'AI' | 'natural' | 'inconclusive'
```

## Repository layout

```
src/v4/            GOS v4 detector source (GOSV4Detector + CLI)
examples/          Example FASTA input and usage notes
LICENSE            Berkeley Lab Non-Commercial Use Only License
NOTICE             DOE contract attribution
```

See [examples/README.md](examples/README.md) for a worked example.
