# Example

`example.fasta` is a small synthetic input used to demonstrate the v4 detector
CLI. It contains two placeholder records. Replace them with real sequences of
interest.

```bash
python -m src.v4.detect \
  --input examples/example.fasta \
  --model-dir /path/to/checkpoint \
  --format json
```

The CLI writes one result per record (JSON by default). Add `--format tsv` for a
compact table. See the top-level [README](../README.md) for checkpoint setup.
