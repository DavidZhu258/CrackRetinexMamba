# Gold Testset Plan: CrackRetinexMamba
> Generated: 2026-06-08  
> Tier: A-core  
> Project bucket: Computer vision research / crack segmentation  
> Priority score: 90  
> GitHub: https://github.com/DavidZhu258/CrackRetinexMamba

## Evaluation Goal

验证 Retinex+Mamba 结构在低光裂缝分割上是否比 baseline 更稳。

## Target Gold Set

- Target size: **500+ images**
- Eval type: `cv_segmentation`
- Seed cases created now: **3**
- First next step from matrix: 补 scripts/eval.py 和 baseline 对照表，固定 test split 与 seeds。

## Test-set Design

公共裂缝数据集 + 自建低光 holdout：正常光、低光、噪声、细裂缝、背景纹理干扰。

## Metrics

Accuracy metrics:

```text
mIoU; Dice/F1; precision; recall; boundary F-score; low-light subset delta; cross-dataset generalization
```

Feasibility metrics:

```text
training reproducibility; inference FPS; VRAM; model size; ONNX export; one-command eval
```

Rubric seed metrics:

- miou
- dice_f1
- precision
- recall
- boundary_f_score
- fps
- vram_mb

## Required Hard Cases

低对比度裂缝、阴影、岩石纹理、细裂缝断裂、非裂缝线状物。

## Build Plan

1. Replace the 3 placeholder seed cases in `evals/gold/seed_gold.jsonl` with real examples.
2. Fill `evals/gold/annotation_template.csv` with expected labels, evidence references, and reviewer status.
3. Run a manual seed evaluation and save raw output in `evals/results/`.
4. Only after the seed suite is stable, expand `evals/gold/full_gold.jsonl` toward the target size.
5. Publish evidence only when the report includes both accuracy and feasibility metrics.

## Acceptance Bar

For portfolio use, the project must pass all seed hard negatives, have a reproducible fresh-run path, and show at least one saved result artifact under `evals/results/`.

## Evidence to Add

定量表、可视化对比图、失败案例、模型大小和 FPS。
