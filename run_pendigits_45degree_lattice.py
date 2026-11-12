"""Measured 0/45/90/135-degree 3D-lattice comparison on native integer UCI PenDigits.

ONE new class per stage (0,1,2,3), positive-only learning, one 101x16
Index-Position log-likelihood matrix per class.  No task ID at inference.

Two controls sharing the identical class-estimation objective and samples:
- shared_axis: four rotated class planes all meet at (v,0,0); last trained
               class overwrites the shared r=0 weights. All other cells
               are disjoint.
- isolated: same angles, origins translated so no class cells overlap.
            Mathematically equal to independent class tables.
- dummy_axis: add one unused geometric column r=0 at the rotation axis;
              original position r is stored at geometric r+1. No intersection,
              no loss of information, no task/class-specific translation.

This code never quantizes or rearranges integer data/axes.  It truly reads
all class weights from the 3D coordinate store during inference.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

VOCAB = 101
LENGTH = 16
CLASSES = [0, 1, 2, 3]
# Integer lattice primitive vectors, each exactly 45 degrees apart.
DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1)]
ANGLES_DEGREES = [0, 45, 90, 135]

def read_data(filename):
    path = Path("data/pendigits") / (filename + ".gz")
    with gzip.open(path, "rt") as stream:
        values = np.asarray(
            [[int(x.strip()) for x in line] for line in csv.reader(stream) if line],
            dtype=np.int64,
        )
    expected = 7494 if filename.endswith(".tra") else 3498
    assert values.shape == (expected, 17), values.shape
    assert values[:, :LENGTH].min() >= 0 and values[:, :LENGTH].max() < VOCAB
    assert values[:, LENGTH].min() >= 0 and values[:, LENGTH].max() <= 9
    return values[:, :LENGTH], values[:, LENGTH]

def train_single_class(X, alpha=1.0):
    """Same per-class Dirichlet-smoothed log probabilities as previous model."""
    count = np.zeros((VOCAB, LENGTH), dtype=np.float64)
    for r in range(LENGTH):
        count[:, r] = np.bincount(X[:, r], minlength=VOCAB)
    return np.log((count + alpha) / (len(X) + alpha * VOCAB))

def coordinate(c, v, r, *, mode):
    """v is always integer vocabulary axis, r is always position axis."""
    dy, dz = DIRECTIONS[c]
    if mode == "isolated":
        # Keep planes more than 2*L apart so no accidental lattice overlap.
        origin = c * 4 * LENGTH
        return (v, origin + dy * r, origin + dz * r)
    if mode == "shared_axis":
        return (v, dy * r, dz * r)
    if mode == "dummy_axis":
        # Geometric column 0 is deliberately vacant (not trained / not read).
        # 16 real feature columns occupy geometric positions 1..16.
        return (v, dy * (r+1), dz * (r+1))
    raise ValueError(mode)

def materialized_tables(storage, classes, mode):
    # At inference, query the physical spatial coordinates, not cached class
    # matrices; shared coordinates reference exactly the same scalar.
    result = []
    for c in classes:
        table = np.empty((VOCAB, LENGTH), dtype=np.float64)
        for v in range(VOCAB):
            for r in range(LENGTH):
                table[v, r] = storage[coordinate(c, v, r, mode=mode)]
        result.append(table)
    return result

def classify(tables, X):
    pos = np.arange(LENGTH)
    score = np.column_stack([W[X, pos].sum(axis=1) for W in tables])
    return score.argmax(axis=1)

def run(mode, output):
    train_x, train_y = read_data("pendigits.tra")
    test_x, test_y = read_data("pendigits.tes")
    storage = {}
    learned = []
    original = []
    history = []
    overlap_by_stage = []
    for c in CLASSES:
        X = train_x[train_y == c]
        class_table = train_single_class(X)
        original.append(class_table)
        shared_overwrites = 0
        for v in range(VOCAB):
            for r in range(LENGTH):
                key = coordinate(c, v, r, mode=mode)
                shared_overwrites += int(key in storage)
                storage[key] = float(class_table[v, r])
        learned.append(c)
        tables = materialized_tables(storage, learned, mode)
        # The first-class closed-set accuracy is tautologically 100%; omit it.
        mask = np.isin(test_y, learned)
        true = test_y[mask]
        pred = classify(tables, test_x[mask])
        confusion = np.zeros((len(learned), len(learned)), dtype=int)
        for actual, predicted in zip(true, pred):
            confusion[int(actual), int(predicted)] += 1
        seen_acc = float(np.mean(pred == true)) if len(learned) > 1 else None
        per_class = {
            str(c0): float(np.mean(pred[true == c0] == c0))
            for c0 in learned
        } if len(learned) > 1 else None
        stage = {
            "learned_class": int(c),
            "angle_degrees": ANGLES_DEGREES[c],
            "seen_classes": learned.copy(),
            "new_training_samples": int(len(X)),
            "tested_samples": int(len(true)),
            "all_seen_class_accuracy": seen_acc,
            "per_class_accuracy": per_class,
            "confusion": confusion.tolist() if len(learned) > 1 else None,
            "lattice_scalar_count": len(storage),
            "new_class_cells_overwriting_old_cells": shared_overwrites,
        }
        history.append(stage)
        overlap_by_stage.append(shared_overwrites)
        print("STAGE",json.dumps(stage),flush=True)
    # True four-class geometry and correct equality for isolated controls.
    expected_shared = (len(CLASSES) - 1) * VOCAB
    n_naive = len(CLASSES) * VOCAB * LENGTH
    assert len(storage) == n_naive - (expected_shared if mode == "shared_axis" else 0)
    assert overlap_by_stage == (
        [0, VOCAB, VOCAB, VOCAB] if mode == "shared_axis" else [0, 0, 0, 0]
    )
    if mode in ("isolated", "dummy_axis"):
        for original_table, read_back in zip(
            original,materialized_tables(storage,learned,mode)
        ):
            assert np.array_equal(original_table,read_back)
        assert abs(history[-1]["all_seen_class_accuracy"] - 0.8591450595655221) < 1e-12
        if mode == "dummy_axis":
            assert all(coord[1:] != (0,0) for coord in storage)
    else:
        r0 = materialized_tables(storage,learned,mode)
        for W in r0[1:]:
            assert np.array_equal(W[:, 0],r0[0][:, 0])
        for c, W in enumerate(r0):
            assert np.array_equal(W[:, 1:],original[c][:, 1:])
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    metrics = {
        "dataset":"UCI PenDigits; official train/test; native integer 0-100",
        "model":"one positive-only learned 101x16 index-position log likelihood table per class",
        "mode":mode,
        "class_order":CLASSES,
        "rotation_angles_degrees":ANGLES_DEGREES,
        "row_axis":"integer token index v",
        "column_axis":"position r",
        "geometry":"shared_axis: (v,r*dy,r*dz); isolated: translated origin; dummy_axis: (v,(r+1)*dy,(r+1)*dz)",
        "direction_vectors":DIRECTIONS,
        "logp_smoothing_alpha":1.0,
        "scores":"sum of 16 position-specific class log probabilities; global argmax; no task ID",
        "overlap_semantics":"shared_axis overwrites real r=0 weights; dummy_axis leaves geometric r=0 unused (real columns r+1) and has no overlap; isolated translates origins",
        "shared_cells_across_classes":expected_shared if mode=="shared_axis" else 0,
        "allocated_feature_columns":LENGTH,
        "geometric_columns_per_class":LENGTH+1 if mode=="dummy_axis" else LENGTH,
        "unused_geometric_axis_column":0 if mode=="dummy_axis" else None,
        "unique_active_spatial_cells":len(storage),
        "all_stages":history,
        "caveat":"Different from original cross-entropy discriminative baseline. Rotation alone cannot prove transfer."
    }
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    # Save a compact, fully explicit geometry for reproducibility.
    with (out/"stage_accuracy.csv").open("w") as stream:
        stream.write("n_classes,mode,accuracy,scalar_count\n")
        for i,x in enumerate(history,1):
            stream.write(f"{i},{mode},{x['all_seen_class_accuracy']},{x['lattice_scalar_count']}\n")
    print("DONE",mode,"four-class-acc",history[-1]["all_seen_class_accuracy"],flush=True)
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["shared_axis","isolated","dummy_axis"],required=True)
    p.add_argument("--output",default="")
    args=p.parse_args()
    run(args.mode,args.output or f"results/pendigits_45degree_{args.mode}")
