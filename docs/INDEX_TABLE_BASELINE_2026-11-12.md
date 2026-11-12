# One Embedding Matrix per Class — preliminary A→B→A baseline

Date: 2026-11-12. Experiment is real, not hypothetical.

## Motivation

Replace the prior 128→128→128→2 MLP with **one 2D index-position weight table per output class** to support a future 4D lattice representation.

This is **not the exact original paper's architecture**. The attached JCP draft, Section 4.3, uses F parallel matrices \(W_f\) for features plus a separate linear classifier. This adaptation assigns one matrix \(W_c\) to each class and produces its logit directly, eliminating both hidden layers and the classification head.

## Input and architecture

- Original UCR TwoPatterns train 1,000 / test 4,000, length 128.
- Series-wise z-normalization, then fixed 64 bins on [-4,4] (a necessary adaptation for continuous TwoPatterns values, not the paper's exact integer vocabulary).
- Per binary class: \(W_c\) of shape [64, 128], one learned bias.
- Two tables total: 2×64×128+2 = **16,386 parameters**.
- Score: \(z_c=\\sum_r W_c[v_r,r]/\\sqrt{128}+b_c\).
- Cross-entropy + Adam, lr 0.01, 20 epochs for A then 20 for B; seed=42.
- A labels 0/1; B original labels 2/3 are mapped to binary 0/1, so **both learning stages update the same two tables**.
- No replay, no task ID at inference, no hidden MLP, no linear head.

## Measured outcome

| Trained | Test A | Test B |
|---|---:|---:|
| A | 68.57% | 62.59% |
| B | 68.57% | 71.39% |

- A forgetting: **0.00 percentage points** measured at these two checkpoints.
- This table-only model is relatively weak on both tasks; its A score is substantially lower than the previous MLP 94.33%.
- **Do not interpret 0% forgetting as solving lifelong learning**: the model might lack the capacity to encode the desired classes effectively, and this is a single seed on one task split.
- The unchanged aggregate A accuracy does not mean its predictions or weights are unchanged: the A confusion matrix changes substantially.
- For a meaningful lattice comparison we should first establish a per-class table model that learns A reliably, or use a dataset more amenable to one index-position table per class.

## Reproduce

```bash
python run_index_table_binary.py --device cpu --epochs-a 20 --epochs-b 20 --seed 42 --bins 64 --lr 0.01
```

Files: `results/one_table_per_class_seed42/metrics.json`, `accuracy_matrix.csv`, `loss.csv`, and `train.log`.
GitHub Actions: https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43182027479
