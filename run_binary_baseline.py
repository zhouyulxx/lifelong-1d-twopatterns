"""Task-wise binary A -> B -> re-test A: same parameters and same two output neurons."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from continual_1d.data import load_twopatterns, get_task_indices

TASKS={"A":(0,1),"B":(2,3)}
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(128,hidden),nn.ReLU(),
                                  nn.Linear(hidden,hidden),nn.ReLU(),
                                  nn.Linear(hidden,2))
    def forward(self,x):
        return self.layers(x)

def evaluate(model,x,y,device):
    model.eval()
    with torch.no_grad():
        prediction=np.concatenate([
            model(torch.from_numpy(x[i:i+256]).to(device)).argmax(1).cpu().numpy()
            for i in range(0,len(x),256)])
    cm=np.zeros((2,2),dtype=np.int64)
    for t,p in zip(y,prediction): cm[int(t),int(p)]+=1
    return {"accuracy":float(np.mean(prediction==y)), "count":int(len(y)),
            "confusion":cm.tolist()}

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else args.device
    if device=="auto":device="cpu"
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    Xtr,Ytr,Xte,Yte,labels=load_twopatterns(args.data_dir)
    dataset={}
    for name,(c0,c1) in TASKS.items():
        it=get_task_indices(Ytr,(c0,c1))
        iv=get_task_indices(Yte,(c0,c1))
        # Reuse *same two output neurons*, so B classes 2/3 become target 0/1
        dataset[name]=((Xtr[it],(Ytr[it]==c1).astype(np.int64)),
                       (Xte[iv],(Yte[iv]==c1).astype(np.int64)))
        print(f"Task {name}: TRAIN={len(it)}, TEST={len(iv)}",flush=True)
    model=MLP(args.hidden).to(device)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    results={};history=[]
    for j,(task,epochs) in enumerate([("A",args.epochs_a),("B",args.epochs_b)]):
        x,y=dataset[task][0]
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),
                          batch_size=args.batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed+j))
        # Only optimizer resets; neural-network parameters and outputs never reset.
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        for epoch in range(1,epochs+1):
            model.train();total=0.
            for bx,by in loader:
                bx,by=bx.to(device),by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(bx),by)
                loss.backward();optimizer.step()
                total+=float(loss.item())*len(bx)
            train_loss=total/len(x)
            history.append((task,epoch,train_loss))
            print(f"Train {task} epoch {epoch}/{epochs}: loss={train_loss:.6f}",flush=True)
        torch.save({"state_dict":model.state_dict(),"stage":task,
                    "original_labels":labels},out/f"model_after_{task}.pt")
        results[task]={}
        for name in TASKS:
            xx,yy=dataset[name][1]
            results[task][name]=evaluate(model,xx,yy,device)
            print(f"After {task}, test {name}: {100*results[task][name]['accuracy']:.3f}%",flush=True)
    forgetting=100*(results["A"]["A"]["accuracy"]-results["B"]["A"]["accuracy"])
    report={"dataset":"UCR TwoPatterns","split":"original UCR TRAIN/TEST",
            "protocol":"same two-output MLP: train A, train B, re-test A; no replay",
            "original_labels":labels,"task_classes":TASKS,
            "binary_targets":{"A":"original 0->0, 1->1","B":"original 2->0, 3->1"},
            "seed":args.seed,"epochs_a":args.epochs_a,"epochs_b":args.epochs_b,
            "learning_rate":args.lr,"hidden":args.hidden,"device":device,
            "results":results,"forgetting_A_percentage_points":forgetting,
            "caveat":"Task-wise binary metric: not four-class class-incremental metric."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (out/"accuracy_matrix.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["after_training","test_task","accuracy","count"])
        for stage in ("A","B"):
            for task in ("A","B"):
                m=results[stage][task];writer.writerow([stage,task,m["accuracy"],m["count"]])
    with (out/"train_history.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["stage","epoch","loss"]);writer.writerows(history)
    print(f"FORGETTING A (percentage points): {forgetting:.3f}",flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data")
    parser.add_argument("--output-dir",default="results/binary_seed42")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs-a",type=int,default=20)
    parser.add_argument("--epochs-b",type=int,default=20)
    parser.add_argument("--hidden",type=int,default=128)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    run(parser.parse_args())
