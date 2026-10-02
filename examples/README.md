# Example

`example.fasta` is a **random sample of the held-out GOS v4 test set** (66 test
records; 24 natural OOD, 42 AI across seven generator families). The sample is
deterministic (`seed=42`, recorded in `example_manifest.json`) and balanced: two
natural (Harvard Forest, GTDB-25k) and four AI (GenomeOcean-500M,
GenomeOcean-100M, GENERATOR-v2, Evo-1) records, full-length sequences.

## Run it

```bash
python -m src.v4.detect \
  --input examples/example.fasta \
  --model-dir <checkpoint-dir> \
  --format json \
  --allow-uncertified
```

`--allow-uncertified` is only needed for exploratory CPU checkpoints; the
production checkpoint does not require it. Use `--format tsv` for a compact
table.

## Example output

Saved in [`example_output.json`](example_output.json) (real run against the
`gos_v4_cpu` checkpoint). Summary of the decision calls:

| Record | Length | Call | Reason | Router confidence |
| --- | ---: | --- | --- | ---: |
| record_1 | 1395 | inconclusive | insufficient_sequence_score_coverage | 0.137 |
| record_2 | 5000 | natural | router_natural | 0.071 |
| record_3 | 861 | AI | router_ai | 0.930 |
| record_4 | 878 | AI | router_ai | 0.926 |
| record_5 | 3072 | AI | router_ai | 0.924 |
| record_6 | 2501 | AI | router_ai | 0.948 |

`prediction`: `1` = AI, `0` = natural, `null` = inconclusive. `router_confidence`
is the detector's calibrated decision score (0–1). `record_1` (a short natural
fragment) is genuinely inconclusive because too little of the sequence was
scorable, so no router decision was made — this is the intended fail-open
behaviour, not a false call.

See [../README.md](../README.md) for checkpoint setup.
