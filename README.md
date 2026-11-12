# Lifelong 1D TwoPatterns

A minimal **non-LLM** continual-learning experiment for **real 1D sequence inputs**:
train Task A, then Task B with the same fixed-size MLP, then test forgetting on A.

**[中文说明 / Chinese README](README.zh-CN.md)** · **[Experiment protocol](docs/PROTOCOL.zh-CN.md)**

## Dataset already included

The original UCR TwoPatterns splits are checked into this repository:

- [TRAIN (1,000 sequences)](data/TwoPatterns/TwoPatterns_TRAIN.ts)
- [TEST (4,000 sequences)](data/TwoPatterns/TwoPatterns_TEST.ts)
- [Source URL, sizes, and SHA-256 checksums](data/TwoPatterns/SOURCE.json)

Each sample contains 128 scalar values, with 4 labels. This is a published, simulated 1D time-series benchmark, **not flattened images**.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_baseline.py --epochs-a 20 --epochs-b 20 --seed 42
```

Outputs: `results/baseline_seed42/` with accuracy matrix, loss history, confusion matrix, metrics JSON, and two weight checkpoints.

Task A = first two label classes; Task B = remaining two. **No replay, no model reset, no task ID during testing.** We do not claim benchmark accuracies until training has actually been run.

To re-download the authentic UCR data:

```bash
python scripts/fetch_twopatterns.py
```

Dataset reference: Dau et al. (2019), *The UCR Time Series Archive*, DOI: 10.1109/JAS.2019.1911747.
