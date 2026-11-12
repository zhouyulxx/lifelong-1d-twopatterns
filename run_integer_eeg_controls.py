"""Control: train Task B alone, with identical architecture, split and epochs."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch

from run_integer_eeg_tables import (
    TASK_CLASSES, load_integer_eeg, split_by_label, GrowingVocab,
    PerClassPositionTable, train_stage, evaluate
)


def run(seed=42, window=30, lr=.005, epochs=20, batch=128,
        out_path="results/eeg_integer_B_only_seed42"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    X, y = load_integer_eeg()
    tr, te = split_by_label(X, y, seed=seed)
    bclasses = TASK_CLASSES["B"]
    tr = tr[np.isin(y[tr], bclasses)]
    te = te[np.isin(y[te], bclasses)]
    x_train, label_train = X[tr], (y[tr]==bclasses[1]).astype(np.int64)
    x_test, label_test = X[te], (y[te]==bclasses[1]).astype(np.int64)
    vocab = GrowingVocab()
    vocab.extend(x_train)
    model = PerClassPositionTable(vocab_size=len(vocab.values), window=window)
    histories = train_stage(
        model, vocab.encode(x_train), label_train, task="B-only",
        epoch_count=epochs, batch_size=batch, lr=lr, seed=seed, device="cpu")
    score = evaluate(model,vocab.encode(x_test),label_test,"cpu")
    result = {
        "dataset":"UCI Epileptic Seizure Recognition",
        "source":"data/eeg_integer/SOURCE.json",
        "model":"One integer-index-position table per class",
        "protocol":"Train B only from scratch, no Task A",
        "task_B_original_classes":bclasses,
        "window":window,"lr":lr,"epochs":epochs,"seed":seed,
        "train_count":len(tr),"test_count":len(te),
        "test_B_accuracy":score["accuracy"],
        "test_B_confusion":score["confusion"],
        "vocab_size":len(vocab.values)
    }
    out = Path(out_path);out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"train_history.json").write_text(json.dumps(histories,indent=2)+"\n")
    print("B only control",json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--output",default="results/eeg_integer_B_only_seed42")
    args=parser.parse_args()
    run(seed=args.seed,epochs=args.epochs,out_path=args.output)
