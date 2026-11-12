"""UCI PenDigits native-integer ordered trajectory benchmark.

Each input is 8 temporally ordered pen points (x,y): 16 native integer
attributes (0..100). This is a 1D ordered input with alternating x/y axes,
NOT a single scalar sensor stream. UCI original train/test splits unchanged.

Models:
  generative: learn one independent 101x16 table per new class from positives
              only; no task ID, global argmax over seen classes.
  discriminative: global 4-output learned index-position weights, A=(0,1)
                  then B=(2,3), no replay; no task ID at inference.
  B_only / joint: controls on exactly the same architecture and dataset.
"""
import argparse,csv,gzip,io,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader

VOCAB=101
LENGTH=16
ORDER=list(range(10))

def read_split(name):
    with gzip.open(Path("data/pendigits")/(name+".gz"),"rt") as f:
        rows=[[int(z.strip()) for z in row] for row in csv.reader(f) if row]
    a=np.asarray(rows,dtype=np.int64)
    n=7494 if name.endswith("tra") else 3498
    assert a.shape==(n,17) and (a[:,:16]>=0).all() and (a[:,:16]<101).all()
    assert (a[:,16]>=0).all() and (a[:,16]<10).all()
    return a[:,:16],a[:,16]

def one_class_table(X,alpha=1.):
    counts=np.zeros((VOCAB,LENGTH),dtype=np.float64)
    for pos in range(LENGTH):
        counts[:,pos]=np.bincount(X[:,pos],minlength=VOCAB)
    return np.log((counts+alpha)/(len(X)+alpha*VOCAB))

def predict_generative(tables,X):
    r=np.arange(LENGTH)
    scores=np.column_stack([table[X,r].sum(axis=1) for table in tables])
    return scores.argmax(1),scores

def auc_binary(pos,neg):
    """P(pos score > neg score) with half-credit on ties."""
    a=np.concatenate([pos,neg])
    sort=np.argsort(a,kind="stable");sorted_a=a[sort]
    rank=np.empty(len(a),dtype=float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and sorted_a[j]==sorted_a[i]:j+=1
        rank[sort[i:j]]=(i+j+1)/2
        i=j
    n=len(pos)
    return float((rank[:n].sum()-n*(n+1)/2)/(n*len(neg)))

def geometry(c,v,r):
    # Fixed integer value axis (v). Rotate position axis by 90 degrees
    # per class around it; translate centers apart to forbid overwrite.
    base=c*4*LENGTH
    k=c%4
    if k==0:dy,dz=r,0
    elif k==1:dy,dz=0,r
    elif k==2:dy,dz=-r,0
    else:dy,dz=0,-r
    return (v,base+dy,base+dz)

def generative(seed,output):
    trainX,trainy=read_split("pendigits.tra")
    testX,testy=read_split("pendigits.tes")
    classes=[];tables=[];stages=[]
    geometry_store={}
    for cls in ORDER:
        X=trainX[trainy==cls]
        table=one_class_table(X)
        cid=len(classes)
        tables.append(table)
        classes.append(cls)
        # Materialize each class matrix in a sparse 3D lattice.
        for v in range(VOCAB):
            for r in range(LENGTH):
                coord=geometry(cid,v,r)
                assert coord not in geometry_store
                geometry_store[coord]=table[v,r]
        known=np.isin(testy,classes)
        labels=testy[known]
        pred,scores=predict_generative(tables,testX[known])
        acc_by_class={str(c):float(np.mean(pred[labels==c]==i)) for i,c in enumerate(classes)}
        result={"class":cls,"n_train":len(X),"n_test_seen":int(known.sum()),
                "accuracy":None if len(classes)==1 else float(np.mean(pred==labels)),
                "per_class_accuracy":None if len(classes)==1 else acc_by_class,
                "total_table_cells":len(geometry_store),
                "class_plane_orientation_deg":(cid%4)*90,
                "single_class_test_accuracy_defined":len(classes)>1}
        if len(classes)==1:
            positive=scores[:,0]
            out_x=testX[testy==1]
            _,unknown_score=predict_generative(tables,out_x)
            result["first_class_vs_class_1_OOD_AUROC"]=auc_binary(positive,unknown_score[:,0])
        stages.append(result)
        print("GEN",json.dumps(result),flush=True)
    # Physically re-read every matrix through 3D lattice coordinates.
    for cls_idx,table in enumerate(tables):
        for v in range(VOCAB):
            for r in range(LENGTH):
                assert geometry_store[geometry(cls_idx,v,r)] == table[v,r]
    report={
        "dataset":"UCI Pen-Based Recognition of Handwritten Digits (official writer-disjoint train/test)",
        "native_input":"8 ordered pen XY points, flattened to 16 integer coordinates",
        "protocol":"one new class at each stage, global argmax, unknown task, no replay",
        "model":"one 101x16 2D index-position log likelihood table per class",
        "learned_objective":"positive-only smoothed per-value-per-position log probability",
        "stage1":"one-class closed-set accuracy undefined; OOD AUROC diagnostic only",
        "order":ORDER,"alpha":1.0,
        "rotation_rule":"90-degree turns about integer-index axis; spatially translated class centers",
        "rotated_lattice_exactly_matches_independent_class_tables":True,
        "geometry_overlapping_weights":0,
        "seed":seed,"stages":stages,
        "caveats":[
            "Exact integers, not binned values.",
            "Alternating X/Y channels, not a single-channel sensor time series.",
            "Generative density may fail for classes with different entropy even when old matrices remain frozen.",
            "Rotation with isolated spatial addresses is a re-indexing; it does not establish transfer or superior learning."
        ]
    }
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(out/"tables.npz",**{f"class_{c}":tables[i] for i,c in enumerate(classes)})
    return report


class ClassTable(nn.Module):
    def __init__(self,nclasses=4):
        super().__init__()
        self.weights=nn.Parameter(torch.empty(nclasses,VOCAB,LENGTH))
        nn.init.normal_(self.weights,0,.025)
        self.bias=nn.Parameter(torch.zeros(nclasses))
    def forward(self,x):
        pos=torch.arange(LENGTH,device=x.device)
        # [classes, batch, position] -> [batch, classes]
        return self.weights[:,x,pos].sum(dim=-1).T+self.bias


def classification(model,X,y,eligible):
    model.eval()
    with torch.no_grad():
        prediction=np.concatenate([
            model(torch.from_numpy(X[i:i+256]))[:,:eligible].argmax(1).cpu().numpy()
            for i in range(0,len(X),256)
        ])
    return {"accuracy":float(np.mean(prediction==y)),
            "n":len(y),
            "per_class_accuracy":{str(c):float(np.mean(prediction[y==c]==c))
                                  for c in np.unique(y).tolist()},
            "confusion":np.asarray([
                [int(((y==c)&(prediction==k)).sum()) for k in range(eligible)]
                for c in range(eligible)]).tolist()}


def fit(model,X,y,nclasses,epochs,lr,seed):
    loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),
                      batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for xx,yy in loader:
            opt.zero_grad(set_to_none=True)
            logits=model(xx)[:,:nclasses]
            loss=nn.functional.cross_entropy(logits,yy)
            loss.backward();opt.step()
            total+=float(loss)*len(yy)
        print("FIT",epoch,nclasses,"loss",round(total/len(y),5),flush=True)


def discriminative(mode,seed,epochs,lr,output):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed)
    X,y=read_split("pendigits.tra")
    XT,Y=read_split("pendigits.tes")
    model=ClassTable()
    stages=[]
    idxA=np.isin(y,[0,1])
    idxB=np.isin(y,[2,3])
    testA=np.isin(Y,[0,1]);testB=np.isin(Y,[2,3])
    if mode=="sequential":
        fit(model,X[idxA],y[idxA],2,epochs,lr,seed)
        stages.append({"stage":"after_A","A":classification(model,XT[testA],Y[testA],2)})
        fit(model,X[idxB],y[idxB],4,epochs,lr,seed+1)
        stages.append({
            "stage":"after_B",
            "A":classification(model,XT[testA],Y[testA],4),
            "B":classification(model,XT[testB],Y[testB],4),
            "global_AB":classification(model,XT[testA|testB],Y[testA|testB],4)
        })
    elif mode=="joint":
        select=idxA|idxB
        fit(model,X[select],y[select],4,epochs,lr,seed)
        stages.append({"stage":"joint","global_AB":classification(model,XT[testA|testB],Y[testA|testB],4)})
    elif mode=="B_only":
        fit(model,X[idxB],y[idxB]-2,2,epochs,lr,seed)
        stages.append({"stage":"B_only","B":classification(model,XT[testB],Y[testB]-2,2)})
    else:raise ValueError(mode)
    result={"dataset":"UCI PenDigits original writer-disjoint train/test",
            "protocol":"2-class stage A (digits 0/1), stage B (2/3); global output evaluation",
            "model":"one trainable 101x16 matrix per output class, no hidden layers",
            "mode":mode,"seed":seed,"epochs":epochs,"lr":lr,
            "num_parameters":sum(p.numel() for p in model.parameters()),
            "stages":stages}
    if mode=="sequential":
        result["forgetting_A_percentage_points"]=100*(stages[0]["A"]["accuracy"]-
                                                          stages[1]["A"]["accuracy"])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    torch.save(model.state_dict(),out/"weights_after.pt")
    print("RESULT",json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["generative","sequential","joint","B_only"],default="generative")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--lr",type=float,default=.01)
    p.add_argument("--output",default=None)
    args=p.parse_args()
    if args.output is None:args.output=f"results/pendigits_{args.mode}_seed{args.seed}"
    if args.mode=="generative":generative(args.seed,args.output)
    else:discriminative(args.mode,args.seed,args.epochs,args.lr,args.output)
