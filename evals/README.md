# evals/ for CrackRetinexMamba

Minimum evaluation scaffold for `CrackRetinexMamba`.

## Layout

```text
evals/
  gold/seed_gold.jsonl
  gold/full_gold.jsonl
  gold/annotation_template.csv
  rubrics/rubric.yaml
  scripts/README.md
  fixtures/
  results/
```

## Contract

- Eval type: `cv_segmentation`
- Target full gold size: `500+ images`
- Seed cases: `3`
- Metrics: miou, dice_f1, precision, recall, boundary_f_score, fps, vram_mb

Replace placeholders with real fixtures/records before using any result as portfolio evidence.
