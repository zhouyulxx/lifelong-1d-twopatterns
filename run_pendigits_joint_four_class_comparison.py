"""Joint four-class control for PenDigits: one M versus CE class-table baseline.

Both use the *same official training and test partitions* and the same four
labels 0,1,2,3. All four classes are available at once to BOTH methods.

The 3D M is updated from a MIXED training batch (not staged class-by-class).
Each example contributes to its label's rotated direction. One M, one blank
rotation-axis column; read and score all directions without task ID.

The CE baseline is the previously measured `joint` run (30 epochs, Adam).
It uses a different parameter-learning objective, so the difference is not
an isolated effect of the rotation geometry.
"""
import json
from pathlib import Path
import numpy as np
from run_pendigits_single_M_direct import (
    SingleRotatedMatrix, ORDER, ANGLES, LENGTH, VOCAB, OFFSET, load_split
)

ROOT = Path("results")
CE_PATH=ROOT/"pendigits_joint_seed42"/"metrics.json"
INCREMENTAL_PATH=ROOT/"pendigits_one_shared_M_dummy_axis"/"metrics.json"

def read_baseline(path):
    return json.loads(path.read_text())

def confusion_matrix(y, pred):
    return np.array([[(int(((y==c)&(pred==p)).sum()))
                     for p in ORDER] for c in ORDER],dtype=np.int64)

def split_accuracy(y, predicted, classes):
    mask=np.isin(y,classes)
    return float(np.mean(y[mask]==predicted[mask]))

def main():
    X,Y=load_split("pendigits.tra")
    XT,YT=load_split("pendigits.tes")
    train_mask=np.isin(Y,ORDER)
    test_mask=np.isin(YT,ORDER)
    X=X[train_mask].copy()
    Y=Y[train_mask].copy()
    XT=XT[test_mask]
    YT=YT[test_mask]

    # Shuffle all 4 labels together before training, rather than grouping by
    # class. Integer count updates are commutative and do not depend on order.
    rng=np.random.default_rng(42)
    shuffled=rng.permutation(len(Y))
    X,Y=X[shuffled],Y[shuffled]
    assert set(Y.tolist())==set(ORDER)

    lattice=SingleRotatedMatrix()
    lattice.angles={c:ANGLES[c] for c in ORDER}
    lattice.positions={c:lattice.coords(c) for c in ORDER}
    lattice.n_by_class={c:int(np.sum(Y==c)) for c in ORDER}
    yy=np.asarray([lattice.positions[c][0] for c in ORDER])
    zz=np.asarray([lattice.positions[c][1] for c in ORDER])

    # One mixed training batch: for each observed (value, position, class)
    # directly increment exactly one voxel in the ONE global 3D M.
    for pos in range(LENGTH):
        np.add.at(lattice.M, (X[:,pos], yy[Y,pos], zz[Y,pos]), 1)

    assert not np.any(lattice.M[:,OFFSET,0]), "The unused axis must remain blank"
    assert np.count_nonzero(lattice.M[:,OFFSET,0])==0
    assert all(int(lattice.M[:,*lattice.positions[c]].sum())==
               LENGTH*lattice.n_by_class[c] for c in ORDER)
    assert int(lattice.M.sum())==LENGTH*len(X)

    predicted=lattice.infer(XT)
    conf=confusion_matrix(YT,predicted)
    joint={
        "accuracy":float(np.mean(predicted==YT)),
        "n":len(YT),
        "A_accuracy":split_accuracy(YT,predicted,(0,1)),
        "B_accuracy":split_accuracy(YT,predicted,(2,3)),
        "per_class_accuracy":{str(c):split_accuracy(YT,predicted,(c,))
                              for c in ORDER},
        "confusion":conf.tolist(),
    }

    ce=read_baseline(CE_PATH)["stages"][0]["global_AB"]
    ce_result={
        "accuracy":ce["accuracy"],
        "n":ce["n"],
        "A_accuracy":float((ce["confusion"][0][0]+ce["confusion"][1][1])/
                           (sum(ce["confusion"][0])+sum(ce["confusion"][1]))),
        "B_accuracy":float((ce["confusion"][2][2]+ce["confusion"][3][3])/
                           (sum(ce["confusion"][2])+sum(ce["confusion"][3]))),
        "per_class_accuracy":ce["per_class_accuracy"],
        "confusion":ce["confusion"]
    }

    # Full proof: joint learning in one M is EXACTLY the same predictor
    # as the previous incremental one-class frequency learner. Every
    # per-class label, prediction, confusion and aggregate must match.
    earlier=read_baseline(INCREMENTAL_PATH)["stages"][-1]
    assert conf.tolist()==earlier["confusion"],(conf,earlier["confusion"])
    assert joint["accuracy"]==earlier["global_accuracy"]
    assert joint["A_accuracy"]==earlier["A_accuracy_after_B"]
    assert joint["B_accuracy"]==earlier["B_accuracy_after_B"]

    report={
      "dataset":"UCI PenDigits, original integer 16-feature official split",
      "classes":list(ORDER),
      "A":[0,1],
      "B":[2,3],
      "training":"Both models receive A+B labeled data from the beginning",
      "train_count":len(X),
      "test_count":len(YT),
      "joint_CE_baseline":{
          "train_method":"30 epoch Adam lr 0.01, global four-class cross entropy",
          **ce_result
      },
      "joint_single_M_rotated":{
          "train_method":"single mixed-data pass of positive class integer frequency updates directly into one 3D M",
          "M_shape":list(lattice.M.shape),
          "unused_rotation_axis":True,
          "orientations_degrees":list(ANGLES),
          **joint
      },
      "joint_CE_minus_joint_M_percentage_points":100*(ce_result["accuracy"]-joint["accuracy"]),
      "joint_M_exactly_matches_one_class_incremental_frequency_method":True,
      "scientific_caution":"Same split and same four class labels, but different training objectives. No demonstrated rotation-related accuracy gain.",
      "source_CE_metrics":str(CE_PATH),
      "source_incremental_M_metrics":str(INCREMENTAL_PATH)
    }
    output=ROOT/"pendigits_joint_four_class_comparison"
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print("CE JOINT 4:",json.dumps(ce_result),flush=True)
    print("ONE-M JOINT 4:",json.dumps(joint),flush=True)
    print("CE minus one-M, percentage points:",
          report["joint_CE_minus_joint_M_percentage_points"],flush=True)
    print("PASS: mixed-training joint four-way single M equals incremental model exactly",flush=True)

if __name__=="__main__":
    main()
