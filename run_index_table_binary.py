"""Minimal one-index-position-embedding-matrix-per-class baseline for UCR TwoPatterns.

Adapted from the index-position tables in Zhou et al., JCP manuscript (2026).
Unlike the paper's F parallel feature tables + linear head, this deliberately
uses one table per class and no learned classification head.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from continual_1d.data import load_twopatterns, get_task_indices

CLASSES = {"A": (0, 1), "B": (2, 3)}

def quantize(x, bins=64, low=-4., high=4.):
    """Fixed bins, independent of future task B and of UCR test data."""
    if bins <= 1 or high <= low: raise ValueError("Invalid quantizer")
    clipped = np.clip(x, low, np.nextafter(np.float32(high), np.float32(low)))
    result = np.floor((clipped - low) * (bins / (high - low)))
    return np.clip(result, 0, bins - 1).astype(np.int64)


class ClassIndexTable(nn.Module):
    def __init__(self, classes=2, vocab=64, positions=128):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, vocab, positions))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.positions = positions
        self.vocab = vocab
        self.classes = classes

    def forward(self, index):
        if index.ndim != 2 or index.shape[1] != self.positions:
            raise ValueError("Expected input [batch, 128] token IDs")
        pos = torch.arange(self.positions, device=index.device)
        selected = self.weight[:, index, pos]  # [classes,batch,positions]
        return selected.sum(dim=-1).T / math.sqrt(self.positions) + self.bias


def accuracy(model, tokens, labels, device):
    model.eval()
    pred=[]
    with torch.no_grad():
        for start in range(0,len(tokens),256):
            x=torch.from_numpy(tokens[start:start+256]).to(device)
            pred.append(model(x).argmax(1).cpu().numpy())
    pred=np.concatenate(pred)
    cm=np.zeros((2,2),dtype=int)
    for t,p in zip(labels,pred): cm[int(t),int(p)]+=1
    return {"accuracy":float((pred==labels).mean()),"count":len(labels),"confusion":cm.tolist()}


def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    Xtr,Ytr,Xte,Yte,raw_labels=load_twopatterns(args.data_dir)
    Xtr=quantize(Xtr,args.bins)
    Xte=quantize(Xte,args.bins)
    tasks={}
    for name,(first,second) in CLASSES.items():
        it=get_task_indices(Ytr,(first,second))
        ie=get_task_indices(Yte,(first,second))
        tasks[name]=((Xtr[it],(Ytr[it]==second).astype(np.int64)),
                     (Xte[ie],(Yte[ie]==second).astype(np.int64)))
        print(f"Task {name}: train {len(it)}, test {len(ie)}",flush=True)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    model=ClassIndexTable(classes=2,vocab=args.bins,positions=128).to(device)
    metrics={};history=[]
    for stage_index,task in enumerate(["A","B"]):
        tokens,labels=tasks[task][0]
        loader=DataLoader(TensorDataset(torch.from_numpy(tokens),torch.from_numpy(labels)),
                          batch_size=args.batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed+stage_index))
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        epochs=args.epochs_a if task=="A" else args.epochs_b
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.
            for x,y in loader:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x),y)
                loss.backward()
                optimizer.step()
                loss_sum+=float(loss.item())*len(y)
            avg=loss_sum/len(tokens);history.append([task,epoch,avg])
            print(f"Task {task} epoch {epoch}/{epochs}: loss={avg:.5f}",flush=True)
        torch.save({"weight":model.state_dict(),"stage":task,
                    "vocab":args.bins,"positions":128},out/f"tables_after_{task}.pt")
        metrics[task]={}
        for eval_task in ("A","B"):
            features,y=tasks[eval_task][1]
            metrics[task][eval_task]=accuracy(model,features,y,device)
            print(f"After {task} test {eval_task}: accuracy={metrics[task][eval_task]['accuracy']:.4%}",flush=True)
    delta=100*(metrics["A"]["A"]["accuracy"]-metrics["B"]["A"]["accuracy"])
    payload={"data":"UCR TwoPatterns", "train_test":"original UCR",
             "model":"one index-position matrix per binary class, no MLP or head",
             "weight_shape":[2,args.bins,128],
             "num_parameters":sum(p.numel() for p in model.parameters()),
             "quantizer":{"kind":"fixed_zscore_bins","bins":args.bins,"low":-4.0,"high":4.0},
             "class_mapping":{"A":"original classes 0/1 => outputs 0/1",
                              "B":"original classes 2/3 => outputs 0/1"},
             "epoch_a":args.epochs_a,"epoch_b":args.epochs_b,
             "seed":args.seed,"lr":args.lr,"batch_size":args.batch_size,
             "metrics":metrics,"forgetting_A_percentage_points":delta,
             "notes":"same two trainable tables continually overwritten; this is task-wise binary (not 4-class class incremental)"}
    (out/"metrics.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    with (out/"accuracy_matrix.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["trained","eval","accuracy","count"])
        for s in ("A","B"):
            for t in ("A","B"):w.writerow([s,t,metrics[s][t]["accuracy"],metrics[s][t]["count"]])
    with (out/"loss.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["task","epoch","loss"]);w.writerows(history)
    print(f"FORGETTING_A_PP={delta:.4f}",flush=True)
    return payload

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data")
    parser.add_argument("--output-dir",default="results/one_table_per_class_seed42")
    parser.add_argument("--bins",type=int,default=64)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--epochs-a",type=int,default=20)
    parser.add_argument("--epochs-b",type=int,default=20)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    run(parser.parse_args())
