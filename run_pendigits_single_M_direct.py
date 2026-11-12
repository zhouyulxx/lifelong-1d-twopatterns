"""Single physical 3D matrix M: learn classes directly at rotated coordinates.

PenDigits official integer train/test, four classes 0->1->2->3.
One unused geometric column r=0 (the shared rotation axis).
Class angles 0,45,90,135 degrees, row index always v in 0..100.

NO per-class learned matrices and NO weight copying:
  M[v, y(theta,r), z(theta,r)] stores that class/position's learned
  integer frequency. New-class samples update M directly at its angle.
  Class likelihood scores are computed directly from M at inference,
  checking all angles without task ID.

The underlying positive-only Dirichlet-smoothed Index-Position objective
is intentionally identical to the previous independent-table experiment.
This is not cross-entropy discriminative training.
"""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

VOCAB=101
LENGTH=16
ORDER=(0,1,2,3)
ANGLES=(0,45,90,135)
DIRECTIONS=((1,0),(1,1),(0,1),(-1,1))
OFFSET=16
ALPHA=1.

def load_split(name):
    with gzip.open(Path("data/pendigits")/(name+".gz"),"rt") as file:
        raw=np.asarray([[int(v.strip()) for v in row] for row in csv.reader(file) if row],dtype=np.int64)
    target=7494 if name.endswith("tra") else 3498
    assert raw.shape==(target,17),raw.shape
    assert raw[:,:LENGTH].min()>=0 and raw[:,:LENGTH].max()<VOCAB
    return raw[:,:LENGTH],raw[:,-1]

class SingleRotatedMatrix:
    """ONLY ONE parameter store; all classes access rotated slices of M."""
    def __init__(self):
        # Reserve enough positions to hold all four angles without realloc.
        self.M=np.zeros((VOCAB,33,17),dtype=np.int32)
        self.n_by_class={}
        self.angles={}
        self.positions={}

    @staticmethod
    def coords(c):
        dy,dz=DIRECTIONS[c]
        r=np.arange(1,LENGTH+1,dtype=np.int64)
        return OFFSET+dy*r,dz*r

    def learn_class(self,c,training):
        if c!=len(self.angles):
            raise ValueError("Only incremental order 0,1,2,3 supported")
        y,z=self.coords(c)
        # Before learning a new class, explicitly snapshot every previously
        # learned class slice and make sure all values are preserved exactly.
        before={old:self.M[:,*self.coords(old)].copy() for old in self.angles}
        assert all(np.count_nonzero(self.M[:,yy,zz])==0 for yy,zz in zip(y,z))
        for r,(yy,zz) in enumerate(zip(y,z)):
            histogram=np.bincount(training[:,r],minlength=VOCAB)
            self.M[:,yy,zz]+=histogram.astype(np.int32)
            assert self.M[:,yy,zz].sum()==len(training)
        self.n_by_class[c]=len(training)
        self.angles[c]=ANGLES[c]
        self.positions[c]=(y.copy(),z.copy())
        old_unchanged=all(np.array_equal(arr,self.M[:,*self.coords(old)])
                          for old,arr in before.items())
        assert old_unchanged
        # The whole rotation-axis column must always remain unused.
        assert not np.any(self.M[:,OFFSET,0])
        return old_unchanged

    def scores(self,X):
        # All directions tried for every input, no known class/task supplied.
        scores=[]
        for c in self.angles:
            yy,zz=self.positions[c]
            # Native integer value v selects first axis; position r selects
            # the rotated (y,z) address in the ONE stored matrix M.
            cnt=self.M[X, yy, zz]
            score=np.log(cnt.astype(np.float64)+ALPHA).sum(axis=1)
            score-=LENGTH*np.log(self.n_by_class[c]+ALPHA*VOCAB)
            scores.append(score)
        return np.column_stack(scores)

    def infer(self,X):
        return np.argmax(self.scores(X),axis=1)

def main():
    train_x,train_y=load_split("pendigits.tra")
    test_x,test_y=load_split("pendigits.tes")
    lattice=SingleRotatedMatrix()
    stages=[]
    for c in ORDER:
        samples=train_x[train_y==c]
        assert lattice.learn_class(c,samples)
        seen=np.isin(test_y,list(lattice.angles))
        labels=test_y[seen]
        predicted=lattice.infer(test_x[seen])
        K=len(lattice.angles)
        confusion=np.zeros((K,K),dtype=int)
        for t,p in zip(labels,predicted):
            confusion[int(t),int(p)]+=1
        stats={
            "learned":list(lattice.angles),
            "class_just_learned":c,
            "angle_degrees":lattice.angles[c],
            "training_samples":len(samples),
            "test_samples_seen":len(labels),
            "global_accuracy":float(np.mean(labels==predicted)) if K>=2 else None,
            "old_weights_preserved_exactly":True,
            "confusion":confusion.tolist() if K>=2 else None,
            "class_accuracy":{str(cl):float(np.mean(predicted[labels==cl]==cl))
                              for cl in lattice.angles} if K>=2 else None,
        }
        if K==4:
            a=(labels==0)|(labels==1)
            b=(labels==2)|(labels==3)
            stats["A_accuracy_after_B"]=float(np.mean(predicted[a]==labels[a]))
            stats["B_accuracy_after_B"]=float(np.mean(predicted[b]==labels[b]))
        stages.append(stats)
        print("STAGE",json.dumps(stats),flush=True)
    # Compare with the prior independent model at all tested stages, not
    # just overall 4-class accuracy.
    old={
        2:0.9669876203576341,
        3:0.84051329055912,
        4:0.8591450595655221
    }
    for stage in stages[1:]:
        assert abs(stage["global_accuracy"]-old[len(stage["learned"])])<1e-12
    # Verify each class's original log-probabilities by recomputing
    # strictly as a READ-ONLY consistency check, never as stored weights.
    for c in lattice.angles:
        yy,zz=lattice.positions[c]
        train=train_x[train_y==c]
        for r in range(LENGTH):
            expected=np.bincount(train[:,r],minlength=VOCAB)
            assert np.array_equal(lattice.M[:,yy[r],zz[r]],expected)
    assert np.count_nonzero(lattice.M[:,OFFSET,0])==0
    assert len({(v,int(yy),int(zz)) for c in ORDER
                for v in range(VOCAB)
                for yy,zz in zip(*lattice.positions[c])})==4*VOCAB*LENGTH

    result={
        "dataset":"UCI PenDigits native integer data, original split",
        "architecture":"Exactly one fixed 3D storage M[v,y,z]; no class matrix copies",
        "storage_dtype":"int32 counters; log-probability weights computed from M on demand",
        "shape_M":list(lattice.M.shape),
        "physical_total_allocated_scalar_cells":int(lattice.M.size),
        "used_weight_coordinates":4*VOCAB*LENGTH,
        "unused_rotation_axis":"M[:,16,0] all zero; effective position r stored at r+1",
        "angles_deg":list(ANGLES),
        "directions_yz":[list(z) for z in DIRECTIONS],
        "learning":"positive-only incremental count update DIRECTLY into M at rotated coordinates",
        "prediction":"global argmax of all learned direction log-probabilities; task unknown",
        "per_class_count_metadata":lattice.n_by_class,
        "smoothing_alpha":ALPHA,
        "extra_parameters_per_new_class":"Only new directions' previously unused lattice cells; class count & angle are metadata",
        "all_old_weight_slices_unchanged":True,
        "equal_independent_probability_table_baseline":True,
        "stages":stages,
        "scope":"4-class PenDigits; generative index-position objective, NOT discriminative CE objective"
    }
    out=Path("results/pendigits_one_shared_M_dummy_axis")
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(out/"single_M_checkpoint.npz",M=lattice.M,
                         n_by_class=np.asarray([lattice.n_by_class[c] for c in ORDER],dtype=np.int64),
                         angles=np.asarray(ANGLES,dtype=np.int32))
    print("VERIFIED one-M four-class accuracy:",stages[-1]["global_accuracy"],flush=True)

if __name__=="__main__":
    main()
