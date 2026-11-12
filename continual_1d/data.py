"""Load original UCR TwoPatterns .ts data and define two disjoint tasks."""
from pathlib import Path
import numpy as np

def _load_ts(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    try:
        i = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data")
    except StopIteration as exc:
        raise ValueError(f"Invalid .ts file without @data: {path}") from exc
    X, y = [], []
    for line in lines[i + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values, label = line.rsplit(":", 1)
        seq = [float(x) for x in values.split(",")]
        if len(seq) != 128:
            raise ValueError(f"Expected 128 values, got {len(seq)} in {path}")
        X.append(seq)
        y.append(label.strip())
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=str)

def normalize_per_sample(X):
    """Each series independently normalized; no future-task statistics leaked."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / np.maximum(sd, 1e-8)).astype(np.float32)

def load_twopatterns(data_dir="data"):
    root = Path(data_dir) / "TwoPatterns"
    paths = [root / "TwoPatterns_TRAIN.ts", root / "TwoPatterns_TEST.ts"]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(
            "Original UCR data missing. Run: python scripts/fetch_twopatterns.py")
    X_train, raw_train = _load_ts(paths[0])
    X_test, raw_test = _load_ts(paths[1])
    labels = sorted(set(raw_train), key=lambda s: float(s))
    if len(labels) != 4:
        raise ValueError(f"Expected 4 categories, got {labels}")
    label_map = {name: idx for idx, name in enumerate(labels)}
    if any(y not in label_map for y in raw_test):
        raise ValueError("Test labels not found in train labels")
    y_train = np.array([label_map[y] for y in raw_train], dtype=np.int64)
    y_test = np.array([label_map[y] for y in raw_test], dtype=np.int64)
    assert X_train.shape == (1000, 128), X_train.shape
    assert X_test.shape == (4000, 128), X_test.shape
    return (normalize_per_sample(X_train), y_train,
            normalize_per_sample(X_test), y_test, labels)

def get_task_indices(y, classes):
    return np.flatnonzero(np.isin(y, classes))
