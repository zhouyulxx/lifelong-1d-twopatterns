"""Sequential A -> B fixed-size MLP baseline; four-way task-agnostic evaluation."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import get_task_indices, load_twopatterns
from .metrics import evaluate
from .model import SequenceMLP

TASKS = {"A": (0, 1), "B": (2, 3)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_task(model, train_x, train_y, *, stage, epochs, batch_size, lr, device, seed):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    # Fresh optimizer state per stage, but the model parameters are NEVER reset.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        avg = total / n
        history.append({"stage": stage, "epoch": epoch, "train_loss": avg})
        print(f"Task {stage}: epoch {epoch}/{epochs}: loss={avg:.5f}", flush=True)
    return history


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(*, output_dir="results/baseline_seed42", data_dir="data", seed=42,
        epochs_a=20, epochs_b=20, lr=1e-3, batch_size=64,
        hidden_dim=128, device="auto"):
    if min(epochs_a, epochs_b, batch_size, hidden_dim) < 1 or lr <= 0:
        raise ValueError("Invalid hyperparameters")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    X_train, y_train, X_test, y_test, original_labels = load_twopatterns(data_dir)
    train_idx = {k: get_task_indices(y_train, v) for k, v in TASKS.items()}
    test_idx = {k: get_task_indices(y_test, v) for k, v in TASKS.items()}
    if any(not len(ids) for ids in [*train_idx.values(), *test_idx.values()]):
        raise ValueError("Empty class task")
    print(f"Train counts: { {k: len(v) for k,v in train_idx.items()} }")
    print(f"Test counts: { {k: len(v) for k,v in test_idx.items()} }")
    model = SequenceMLP(hidden_dim=hidden_dim).to(resolved_device)
    history, scores, stage_metrics = [], [], {}
    final_predictions = np.full(len(y_test), -1, dtype=np.int64)

    for stage_num, stage in enumerate(("A", "B")):
        idx = train_idx[stage]
        print(f"\n=== Training {stage}: classes {TASKS[stage]} ===")
        history += train_task(
            model, X_train[idx], y_train[idx], stage=stage,
            epochs=epochs_a if stage == "A" else epochs_b,
            batch_size=batch_size, lr=lr, device=resolved_device,
            seed=seed + stage_num)
        torch.save({"model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim, "stage": stage,
                    "original_labels": original_labels},
                   output / f"model_after_{stage}.pt")
        stage_metrics[stage] = {}
        for test_stage in ("A", "B"):
            idx = test_idx[test_stage]
            result = evaluate(model, X_test[idx], y_test[idx],
                              TASKS[test_stage], resolved_device)
            record = {
                "after_training": stage, "eval_task": test_stage,
                "global_accuracy": result["global_accuracy"],
                "task_aware_accuracy_diagnostic": result["task_aware_accuracy_diagnostic"],
                "n_test": int(len(idx))}
            scores.append(record)
            stage_metrics[stage][test_stage] = {
                "global_accuracy": record["global_accuracy"],
                "task_aware_accuracy_diagnostic": record["task_aware_accuracy_diagnostic"],
                "n_test": record["n_test"]}
            print(f"After {stage} -> test {test_stage}: "
                  f"global={record['global_accuracy']:.4f}; "
                  f"task-aware diagnostic={record['task_aware_accuracy_diagnostic']:.4f}")
            if stage == "B":
                final_predictions[idx] = result["predicted"]

    forgetting = (stage_metrics["A"]["A"]["global_accuracy"]
                  - stage_metrics["B"]["A"]["global_accuracy"])
    confusion = np.zeros((4, 4), dtype=np.int64)
    for target, pred in zip(y_test, final_predictions):
        confusion[int(target), int(pred)] += 1
    write_csv(output / "accuracy_matrix.csv", scores,
              ["after_training", "eval_task", "global_accuracy",
               "task_aware_accuracy_diagnostic", "n_test"])
    write_csv(output / "train_history.csv", history,
              ["stage", "epoch", "train_loss"])
    write_csv(output / "confusion_after_B.csv",
              [{"actual": row, **{f"pred_{col}": int(confusion[row,col]) for col in range(4)}}
               for row in range(4)],
              ["actual"] + [f"pred_{col}" for col in range(4)])
    report = {
        "dataset": "UCR TwoPatterns", "split": "original UCR TRAIN/TEST",
        "tasks": TASKS, "original_label_order": original_labels,
        "protocol": "class-incremental A then B, four global outputs, no task ID",
        "replay": False, "preprocessing": "per-series z-normalization",
        "seed": seed, "parameters": {
            "epochs_a": epochs_a, "epochs_b": epochs_b, "lr": lr,
            "batch_size": batch_size, "hidden_dim": hidden_dim,
            "device": resolved_device},
        "train_counts": {k: int(len(v)) for k,v in train_idx.items()},
        "test_counts": {k: int(len(v)) for k,v in test_idx.items()},
        "stage_metrics": stage_metrics,
        "forgetting_A_global": float(forgetting),
        "note": "Task-aware scores are DIAGNOSTIC only; not a reproduction of prior published figures."
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\nForgetting A: {forgetting:.4f}. Results saved in {output}")
    return report
