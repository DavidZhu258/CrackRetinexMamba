# Evaluation Results
> Last updated: 2026-06-09

This repository now includes a lightweight, GitHub-visible seed-gold evaluation scaffold.

## What is tested now

- `evals/gold/seed_gold.jsonl` contains 3 real repository-grounded seed cases.
- Every seed case references files that exist in this repository.
- At least one seed case is a hard-negative/safety/refusal case.
- `evals/scripts/validate_seed.py` writes `evals/results/latest_seed_validation.json`.
- GitHub Actions workflow `.github/workflows/eval-seed-validation.yml` runs the same validator on pushes and pull requests touching `evals/**`.

## Current limitation

This seed validation proves traceability and feasibility scaffolding. It does **not** yet prove final model/product accuracy. Full quality claims require expanding `evals/gold/full_gold.jsonl` and adding project-specific model/system runners.

## Local command

```bash
python evals/scripts/validate_seed.py
```

## Result artifact

See `evals/results/latest_seed_validation.json` after running the validator locally or in GitHub Actions.
