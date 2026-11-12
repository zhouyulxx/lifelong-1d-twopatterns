"""Native integer-valued EEG A->B->A continual-learning table benchmark.

One 2D index-position embedding matrix per binary output class. No CNN, MLP,
linear classification head, or numeric binning. Two class-tables are reused
sequentially on Tasks A and B, with dynamic vocabulary growth at Stage B.

This is a NEW custom task-incremental split of a published EEG classification
dataset, not an official UCI continual-learning protocol.
"""
import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TASK_CLASSES = {"A": (1, 5), "B": (2, 4)}


def load_integer_eeg(path="data/eeg_integer/data.csv.gz"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Missing verified EEG file; see data/eeg_integer/SOURCE.json")
    opener = gzip.open if str(path).endswith(".gz") else open
    names = [f"X{i}" for i in range(1, 179)]
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(names + ["y"]).issubset(reader.fieldnames or []):
            raise ValueError("Expected X1..X178 and y columns")
        rows, labels = [], []
        for line in reader:
            vals = [int(line[k]) for k in names]
            rows.append(vals)
            labels.append(int(line["y"]))
    X = np.asarray(rows, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if X.shape != (11500, 178):
        raise ValueError(f"Unexpected source shape: {X.shape}")
    if set(y.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five UCI classes")
    return X, y


def split_by_label(X, y, *, seed=42, train_fraction=.8):
    """Reproducible class-stratified sample split. No patient identity available."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in sorted(set(y.tolist())):
        ids = np.flatnonzero(y == label)
        rng.shuffle(ids)
        n_train = int(len(ids) * train_fraction)
        train_idx.extend(ids[:n_train].tolist())
        test_idx.extend(ids[n_train:].tolist())
    return (np.asarray(train_idx), np.asarray(test_idx))


class GrowingVocab:
    def __init__(self):
        self.token_to_id = {}
        self.values = []

    def extend(self, x):
        new = sorted(set(np.asarray(x).ravel().tolist()) - set(self.token_to_id))
        for v in new:
            self.token_to_id[v] = len(self.values)
            self.values.append(int(v))
        return len(new)

    def encode(self, x):
        result = np.fromiter(
            (self.token_to_id.get(int(v), -1) for v in np.asarray(x).ravel()),
            dtype=np.int64, count=x.size).reshape(x.shape)
        return result


class PerClassPositionTable(nn.Module):
    def __init__(self, *, vocab_size, window=30, num_classes=2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, vocab_size, window))
        nn.init.normal_(self.W, std=0.025)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.window = window

    def grow(self, additional_rows):
        if additional_rows <= 0: return
        with torch.no_grad():
            new = self.W.new_empty((self.W.shape[0], additional_rows, self.window))
            nn.init.normal_(new, std=0.025)
            self.W = nn.Parameter(torch.cat([self.W.detach(), new], dim=1))

    def forward(self, tokens):
        # tokens: batch x sequence length, -1 for unseen OOV values.
        if tokens.ndim != 2 or tokens.shape[1] < self.window:
            raise ValueError("Tokens must be [batch, length>=window]")
        view = tokens.unfold(1, self.window, 1)  # [batch, #windows, window]
        position = torch.arange(self.window, device=tokens.device)
        # Appended zero-weight row for OOV: no trainable OOV contribution.
        weights = torch.cat([torch.tanh(self.W),
                             torch.zeros_like(self.W[:, :1, :])], dim=1)
        ids = torch.where(view < 0, weights.shape[1] - 1, view)
        selected = weights[:, ids, position]  # [classes, batch, #windows, window]
        return selected.sum(-1).permute(1, 2, 0) / math.sqrt(self.window) + self.bias


def evaluate(model, encoded, labels, device):
    model.eval()
    true, predicted = [], []
    with torch.no_grad():
        for offset in range(0, len(labels), 64):
            x = torch.from_numpy(encoded[offset:offset+64]).to(device)
            window_prob = model(x).softmax(-1).mean(dim=1)
            predicted.extend(window_prob.argmax(-1).cpu().numpy().tolist())
            true.extend(labels[offset:offset+64].tolist())
    actual = np.asarray(true)
    pred = np.asarray(predicted)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(actual, pred):
        cm[int(t), int(p)] += 1
    return {"accuracy": float(np.mean(pred == actual)),
            "n": int(len(actual)), "confusion": cm.tolist()}


def train_stage(model, x, y, *, task, epoch_count, batch_size, lr, seed, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epoch_count+1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            target = by[:, None].expand(-1, logits.shape[1]).reshape(-1)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 2), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(by)
        avg_loss = total / len(y)
        history.append({"task":task,"epoch":epoch,"window_ce_loss":avg_loss})
        print(f"Task {task}: epoch {epoch}/{epoch_count}, loss={avg_loss:.5f}", flush=True)
    return history


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    X, labels = load_integer_eeg(args.input)
    train_idx, test_idx = split_by_label(X, labels, seed=args.seed)
    outputs = {}
    for task, pair in TASK_CLASSES.items():
        a, b = pair
        tr = train_idx[np.isin(labels[train_idx], pair)]
        te = test_idx[np.isin(labels[test_idx], pair)]
        outputs[task] = ((X[tr], (labels[tr] == b).astype(np.int64)),
                         (X[te], (labels[te] == b).astype(np.int64)))
        print(f"Task {task} original labels={pair} train={len(tr)} test={len(te)}",
              flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vocab = GrowingVocab()
    vocab.extend(outputs["A"][0][0])
    model = PerClassPositionTable(vocab_size=len(vocab.values), window=args.window).to(device)
    print(f"Stage A vocabulary size={len(vocab.values)}", flush=True)

    all_metrics = {}
    histories = []
    for stage_idx, task in enumerate(("A","B")):
        if task == "B":
            n = vocab.extend(outputs["B"][0][0])
            model.grow(n)
            print(f"Stage B added {n} integer values, total vocab={len(vocab.values)}", flush=True)
        tr_X, tr_y = outputs[task][0]
        histories += train_stage(
            model, vocab.encode(tr_X), tr_y,
            task=task, epoch_count=args.epochs_a if task=="A" else args.epochs_b,
            batch_size=args.batch, lr=args.lr, seed=args.seed+stage_idx, device=device)
        torch.save({"weights":model.state_dict(),"vocabulary":vocab.values,
                    "stage":task,"window":args.window}, out / f"model_after_{task}.pt")
        all_metrics[task] = {}
        for eval_task in ("A","B"):
            x, y = outputs[eval_task][1]
            enc = vocab.encode(x)
            oov = float((enc<0).mean())
            m = evaluate(model, enc, y, device)
            m["oov_ratio"] = oov
            all_metrics[task][eval_task] = m
            print(f"After {task}, test {eval_task}: accuracy={m['accuracy']:.4%}"
                  f"; OOV={oov:.4%}", flush=True)

    forgetting = 100 * (all_metrics["A"]["A"]["accuracy"] -
                        all_metrics["B"]["A"]["accuracy"])
    report = {
        "dataset":"UCI Epileptic Seizure Recognition (integer-valued signals)",
        "source":"data/eeg_integer/SOURCE.json",
        "split":"Stratified 80/20 at EEG chunk level; no patient-level grouping",
        "tasks":TASK_CLASSES,
        "protocol":"Task-wise binary A -> B -> re-test A, same 2 output tables, no replay",
        "method":"One exact integer-index-by-window-position matrix per binary class",
        "window":args.window, "epochs_a":args.epochs_a,"epochs_b":args.epochs_b,
        "lr":args.lr,"seed":args.seed,"batch_size":args.batch,
        "initial_vocabulary_size":len(set(outputs["A"][0][0].ravel().tolist())),
        "final_vocabulary_size":len(vocab.values),
        "final_parameter_count":sum(p.numel() for p in model.parameters()),
        "metrics":all_metrics,"forgetting_A_percentage_points":forgetting,
        "notes":"UCI is a published EEG classification dataset. This A/B split is researcher-defined; "
                "no clipping, rounding, normalization, or quantization of input signals. "
                "Unseen test integers receive zero contribution. "
                "EEG segments from the same original recording may occur in train and test; not clinical validation."
    }
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (out/"accuracy_matrix.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["after","test_task","accuracy","n","oov_ratio"])
        for s in ("A","B"):
            for t in ("A","B"):
                m=all_metrics[s][t]
                w.writerow([s,t,m["accuracy"],m["n"],m["oov_ratio"]])
    with (out/"train_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task","epoch","window_ce_loss"]);w.writeheader();w.writerows(histories)
    print(f"FORGETTING A: {forgetting:.4f} percentage points", flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/eeg_integer/data.csv.gz")
    p.add_argument("--output",default="results/eeg_integer_table_seed42")
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs-a",type=int,default=20)
    p.add_argument("--epochs-b",type=int,default=20)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=0.005)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    main(p.parse_args())
