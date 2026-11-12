"""Single-class incremental EEG: native integer Index-Position 2D tables.

This is a *generative* adaptation of Index-Position Embedding, not the original
discriminative cross-entropy objective from Zhou et al. Each new class is learned
from positive-only data, then all learned class likelihoods compete at inference.

No task identity, class labels, or unknown-class samples are used at inference.
The first stage with only one learned class has no meaningful closed-set accuracy;
we report an out-of-class ranking AUROC *diagnostic* instead.

No rounding, z-normalization, or quantization of source EEG values.
The fixed offset is an exact bijection for the bounded integer domain.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_integer_eeg_tables import load_integer_eeg, split_by_label

# Fixed input alphabet selected independently of the test split. 4096 offset
# is exact integer indexing, not binning (1 integer -> 1 distinct ID).
MIN_VALUE=-4096
MAX_VALUE=4095
VOCAB=MAX_VALUE-MIN_VALUE+1
ORDER=[1,5,2,4,3]

def encode_exact(X):
    a=np.asarray(X,dtype=np.int64)
    if a.min()<MIN_VALUE or a.max()>MAX_VALUE:
        raise ValueError("Observed integer outside fixed domain; no clipping permitted")
    return (a-MIN_VALUE).astype(np.int32)


def position_counts(values, window):
    n,length=values.shape
    n_windows=length-window+1
    counts=np.zeros((VOCAB,window),dtype=np.uint32)
    for r in range(window):
        c=np.bincount(values[:,r:r+n_windows].ravel(),minlength=VOCAB)
        counts[:,r]=c
    assert (counts.sum(axis=0)==n*n_windows).all()
    return counts


def fit_class(values,window,alpha):
    """One scalar weight per (integer index, window position).

    Dirichlet-smoothed P(v|r,class) with fixed vocabulary support.
    Smooth weights are real logits, learned entirely from this class.
    """
    cnt=position_counts(values,window).astype(np.float64)
    denom=cnt.sum(axis=0,keepdims=True)+alpha*VOCAB
    logp=np.log(cnt+alpha)-np.log(denom)
    return logp.astype(np.float32)


def eval_scores(models, values, window, batch=64):
    # models: ordered list of [vocab,position] log P(v|r,class) tables
    predictions=[]
    all_scores=[]
    for i in range(0,len(values),batch):
        batch_x=values[i:i+batch]
        B,L=batch_x.shape
        # shape: [B, L-window+1, window]
        wins=np.lib.stride_tricks.sliding_window_view(batch_x,window,axis=1)
        r=np.arange(window)
        scores=np.column_stack([
            model[wins,r].mean(axis=(1,2)) for model in models
        ])
        all_scores.append(scores)
        predictions.append(scores.argmax(1))
    return np.concatenate(predictions),np.concatenate(all_scores)


def auc_rank(positive,negative):
    """Exact AUC with ties via sorted pooled ranks; positive higher is better."""
    values=np.concatenate([positive,negative])
    n1=len(positive);n0=len(negative)
    idx=np.argsort(values,kind="stable")
    sorted_values=values[idx]
    ranks=np.empty(len(values),dtype=np.float64)
    i=0
    while i<len(values):
        j=i+1
        while j<len(values) and sorted_values[j]==sorted_values[i]: j+=1
        ranks[idx[i:j]]=(i+j+1)/2.0
        i=j
    return float((ranks[:n1].sum()-n1*(n1+1)/2)/(n1*n0))


def geometry(class_index, position, window):
    """Conservative 3D lattice coordinate; row axis (v) remains fixed.

    Step between class centers makes planes nonintersecting. Rotations are
    a *storage re-indexing*: they do not create new learning behavior.
    """
    q=class_index%4
    if q==0: dy,dz=position,0
    elif q==1: dy,dz=0,position
    elif q==2: dy,dz=-position,0
    else: dy,dz=0,-position
    base=4*window*class_index
    return base+dy,base+dz


def main(args):
    X,y=load_integer_eeg()
    tr,te=split_by_label(X,y,seed=args.seed)
    X=encode_exact(X)
    tables=[];seen=[];stages=[]
    for stage,cls in enumerate(ORDER[:args.classes],1):
        t=tr[y[tr]==cls]
        table=fit_class(X[t],args.window,args.alpha)
        tables.append(table)
        seen.append(cls)
        testing=te[np.isin(y[te],seen)]
        pred,scores=eval_scores(tables,X[testing],args.window)
        global_target=np.asarray([seen.index(int(z)) for z in y[testing]],dtype=np.int64)
        confusion=np.zeros((len(seen),len(seen)),dtype=np.int64)
        for a,b in zip(global_target,pred): confusion[a,b]+=1
        class_acc={str(k):float(np.mean(pred[global_target==j]==j))
                   for j,k in enumerate(seen)}
        stage_data={
            "stage":stage,"new_class":cls,"seen_classes":seen.copy(),
            "training_samples_for_new_class":int(len(t)),
            "global_accuracy":None if stage==1 else float(np.mean(pred==global_target)),
            "n_seen_test":int(len(testing)),
            "per_class_accuracy":None if stage==1 else class_acc,
            "confusion":None if stage==1 else confusion.tolist(),
            "table_parameter_count":len(tables)*VOCAB*args.window,
            "class_plane_angle_degrees":{str(c):90*k for k,c in enumerate(seen)}
        }
        if stage==1:
            ood=te[y[te]==ORDER[1]]
            _,scores_unseen=eval_scores(tables,X[ood],args.window)
            stage_data["one_class_vs_next_unseen_AUROC_diagnostic"] = auc_rank(
                scores[:,0],scores_unseen[:,0])
            stage_data["one_class_accuracy"]="not defined (one seen class)"
        stages.append(stage_data)
        print("STAGE",stage,
              "new class",cls,
              "global_seen_accuracy",stage_data["global_accuracy"],
              "class_accuracies",stage_data["per_class_accuracy"],
              "first_stage_AUROC",stage_data.get("one_class_vs_next_unseen_AUROC_diagnostic"),
              flush=True)

    # Geometry is exactly lossless because each (class,v,r) has one unique point.
    # Validate no collision of nontrivial sampled coordinates across stages.
    coords=set()
    for c in range(len(seen)):
        for r in range(args.window):
            yz=geometry(c,r,args.window)
            for v in (0,1024,4096,VOCAB-1):
                coord=(v,*yz)
                assert coord not in coords,(c,r,coord)
                coords.add(coord)
    # The prediction operation never reads geometry coordinates; it reads
    # the class-local logp through a reversible map, so rotated == stacked.
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    report={
        "dataset":"UCI Epileptic Seizure Recognition, original integer samples",
        "protocol":"one new class per stage; known classes compete globally; no task ID",
        "learning_objective":"one-class Dirichlet-smoothed index-position log likelihood",
        "explicit_caveat":"Not the original discriminative cross-entropy method; this is a generative single-class adaptation.",
        "order":seen,"seed":args.seed,
        "window":args.window,"alpha":args.alpha,
        "integer_encoding":{"minimum":MIN_VALUE,"maximum":MAX_VALUE,"vocabulary_size":VOCAB,
                            "mapping":"exact integer +4096, NOT quantization"},
        "geometry":{"spatial_axes":["integer_id","y","z"],
                    "class_angles_degrees":[90*i for i in range(len(seen))],
                    "class_centers":"(4*window*class_index,4*window*class_index)",
                    "rotated_equivalent_to_independent_storage":True,
                    "overlapping_cells":0},
        "stages":stages,
        "limits":[
            "At stage one, accuracy over one seen class is trivially 100% and is not reported.",
            "No labels are needed at inference, but the current closed-set evaluation excludes unseen classes.",
            "The AUROC at stage 1 is an OOD discrimination diagnostic using held-out test labels only for evaluation.",
            "No training on previous classes when fitting a new class.",
            "Rotation is an exact coordinate permutation: it cannot change predictions without overlap or another shared mechanism.",
            "The UCI split is segment level; it does not guarantee patient independence."
        ]
    }
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    # Save learned 2D class tables as reusable weights.
    np.savez_compressed(out/"class_tables.npz",**{f"class_{v}":tables[i] for i,v in enumerate(seen)})
    print("DONE",out,flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--alpha",type=float,default=1.0)
    p.add_argument("--classes",type=int,default=5)
    p.add_argument("--output",default="results/generative_single_class_seed42")
    a=p.parse_args()
    if a.classes<2 or a.classes>5: raise ValueError("classes must be 2 to 5")
    main(a)
