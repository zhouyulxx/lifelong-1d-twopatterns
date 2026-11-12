"""Four-global-class incremental discriminative Index-Position baseline.

Task A = EEG classes (1,5), Task B = (2,4); 4 logits always exist.
Train Task A with only old two outputs; then Task B with all four outputs
but ONLY B training examples. Inference uses global argmax, NO task ID.

Joint-data oracle and B-only control use the same model, preprocessing and
per-window architecture. Values are exact native integers: no binning.

This is a standard two-class-per-experience class-incremental experiment.
"""
import argparse,json,math,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from run_integer_eeg_tables import load_integer_eeg,split_by_label
from run_single_class_incremental import encode_exact,VOCAB,ORDER


class DiscriminativeIndexPosition(nn.Module):
    def __init__(self,window=30,num_classes=4):
        super().__init__()
        self.window=window
        self.embedding=nn.Embedding(VOCAB*window,num_classes,sparse=True)
        nn.init.normal_(self.embedding.weight,0,0.025)
        self.bias=nn.Parameter(torch.zeros(num_classes))

    def forward(self,x):
        b,n=x.shape
        p=torch.arange(self.window,device=x.device)
        ids=x.unfold(1,self.window,1)*self.window+p
        # Per-window class scores, shared across every sequence
        scores=self.embedding(ids).sum(-2)/math.sqrt(self.window)
        # Sequence-level average-logit classification
        return scores.mean(dim=1)+self.bias


def get_accuracy(model,x,y,classes,all_labels):
    true=np.asarray([all_labels.index(int(z)) for z in y])
    predictions=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),64):
            logits=model(torch.from_numpy(x[start:start+64].astype(np.int64)))
            # Expose only outputs learned by this stage; class IDs are
            # in original global label order (first classes of all_labels).
            mask=logits[:, :classes]
            predictions.extend(mask.argmax(-1).cpu().numpy().tolist())
    p=np.asarray(predictions)
    cm=np.zeros((classes,classes),dtype=int)
    for a,b in zip(true,p):
        if a<classes:cm[a,b]+=1
    per_class={str(label):float((p[true==i]==i).mean())
               for i,label in enumerate(all_labels[:classes]) if (true==i).any()}
    return {"accuracy":float((true==p).mean()),
            "per_class_accuracy":per_class,"n":len(y),
            "confusion":cm.tolist()}


def train(model,x,y,available,epochs,lr,batch,seed):
    opt_e=torch.optim.SparseAdam(model.embedding.parameters(),lr=lr)
    opt_b=torch.optim.Adam([model.bias],lr=lr)
    loader=DataLoader(TensorDataset(
        torch.from_numpy(x.astype(np.int64)),
        torch.from_numpy(y.astype(np.int64))),
        batch_size=batch,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    history=[]
    for e in range(1,epochs+1):
        model.train();running=0.
        for xx,yy in loader:
            opt_e.zero_grad(set_to_none=True);opt_b.zero_grad(set_to_none=True)
            logits=model(xx)[:,:available]
            loss=nn.functional.cross_entropy(logits,yy)
            loss.backward();opt_e.step();opt_b.step()
            running+=float(loss)*len(yy)
        mean=running/len(y)
        print(f"epoch={e}/{epochs} loss={mean:.5f}, global_classes={available}",flush=True)
        history.append(mean)
    return history


def main(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    X,y=load_integer_eeg()
    X=encode_exact(X)
    tr,te=split_by_label(X,y,seed=args.seed)
    model=DiscriminativeIndexPosition(window=args.window)
    A_labels=ORDER[:2];B_labels=ORDER[2:4]
    Atrain=tr[np.isin(y[tr],A_labels)]
    Btrain=tr[np.isin(y[tr],B_labels)]
    Atest=te[np.isin(y[te],A_labels)]
    Btest=te[np.isin(y[te],B_labels)]
    allt=np.concatenate([Atest,Btest])
    report={
        "dataset":"UCI integer EEG",
        "protocol":"4 global output class-incremental; A=(1,5), B=(2,4), no task ID",
        "model":"one learned integer-index-position weight matrix per global class",
        "mode":args.mode,"seed":args.seed,"window":args.window,
        "lr":args.lr,"epochs":args.epochs,"batch":args.batch,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "stages":[]
    }
    history={}
    if args.mode=="sequential":
        stages=[
            ("after_A",Atrain,2,args.epochs),
            ("after_B",Btrain,4,args.epochs)]
    elif args.mode=="joint":
        stages=[("joint_AB",np.concatenate([Atrain,Btrain]),4,args.epochs)]
    elif args.mode=="B_only":
        stages=[("B_only",Btrain,4,args.epochs)]
    else: raise ValueError(args.mode)

    # In B_only, train only class output indices 2,3; keep others
    # inaccessible, and evaluate with B-only logits via custom branch.
    if args.mode=="B_only":
        # relabel 2/3 into 0/1 and use two-class slots of same model
        B_local=np.asarray([0 if y[idx]==B_labels[0] else 1 for idx in Btrain])
        stages=[("B_only",Btrain,2,args.epochs)]
    for st,indices,available,ep in stages:
        if args.mode=="B_only":
            targets=B_local
        else:
            targets=np.asarray([ORDER.index(int(y[idx])) for idx in indices])
        history[st]=train(model,X[indices],targets,available,ep,args.lr,args.batch,args.seed+len(history))
        result={"stage":st,"trained_on_original_labels":
                [1,5] if st=="after_A" else [2,4] if st in ("after_B","B_only") else [1,5,2,4]}
        if args.mode=="B_only":
            # B mapped to outputs 0 and 1 for this control.
            result["B_test"]=get_accuracy(model,X[Btest],
                np.asarray([B_labels[0] if y[k]==B_labels[0] else B_labels[1] for k in Btest]),
                2,B_labels)
        else:
            result["seen_test"]=get_accuracy(model,X[np.concatenate([
                Atest, Btest]) if available==4 else X[Atest],
                y[np.concatenate([Atest,Btest])] if available==4 else y[Atest],
                available,ORDER)
            result["A_test"]=get_accuracy(model,X[Atest],y[Atest],available,ORDER)
            if available==4:
                result["B_test"]=get_accuracy(model,X[Btest],y[Btest],available,ORDER)
        print("STAGE_RESULTS",json.dumps(result),flush=True)
        report["stages"].append(result)
    if args.mode=="sequential":
        a1=report["stages"][0]["A_test"]["accuracy"]
        a2=report["stages"][1]["A_test"]["accuracy"]
        report["forgetting_A_percentage_points"]=100*(a1-a2)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"train_losses.json").write_text(json.dumps(history,indent=2)+"\n")
    torch.save(model.state_dict(),out/"weights_after.pt")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["sequential","joint","B_only"],default="sequential")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--lr",type=float,default=.005)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    if a.output is None: a.output=f"results/discriminative_global_{a.mode}_seed{a.seed}"
    main(a)
