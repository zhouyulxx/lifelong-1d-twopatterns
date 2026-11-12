"""A spatial 3D lattice of class-specific 2D index-position matrices.

Fixed row axis: integer vocabulary ID v.
Class c in task A: (v, r, c*(m+1)).
Class c in task B: (v, 0, c*(m+1)+r) -- a 90-degree rotation
of the position axis around the shared vocabulary/row axis.

The A/B plane intersection is exactly position r=0 for a matching
class slot c and vocabulary row v.  This controlled shared-line
experiment is contrasted with an isolated version whose B plane is
translated to y=m.

Train A, then B on the same 3D lattice, then re-test A. Test-time
orientation uses the *task ID*, never the target class label.
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

from run_integer_eeg_tables import (
    TASK_CLASSES, load_integer_eeg, split_by_label, GrowingVocab
)

class RotatedLattice(nn.Module):
    def __init__(self, vocab_size, window=30, mode="intersection"):
        super().__init__()
        if mode not in ("intersection", "isolated"):
            raise ValueError(mode)
        self.window = window
        self.mode = mode
        self.coord_to_id = {}
        self.lattice = None
        self.bias = nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("lookup_A", torch.full((2,vocab_size,window),-1,dtype=torch.long))
        self.register_buffer("lookup_B", torch.full((2,vocab_size,window),-1,dtype=torch.long))
        self.register_buffer("a_seen", torch.zeros(vocab_size,dtype=torch.bool))
        self.register_buffer("b_seen", torch.zeros(vocab_size,dtype=torch.bool))

    def extend_vocab(self, new_size):
        old_size = self.lookup_A.shape[1]
        if new_size < old_size:
            raise ValueError("Vocab cannot shrink")
        if new_size == old_size:
            return
        delta = new_size-old_size
        for name in ("lookup_A","lookup_B"):
            current = getattr(self,name)
            extra = torch.full((2,delta,self.window),-1,dtype=torch.long,device=current.device)
            setattr(self,name,torch.cat([current,extra],dim=1))
        for name in ("a_seen","b_seen"):
            cur = getattr(self,name)
            setattr(self,name,torch.cat([cur,torch.zeros(delta,dtype=torch.bool,device=cur.device)]))

    def _coordinate(self, stage, c, v, r):
        anchor = c * (self.window+1)
        if stage == "A":
            return (v,r,anchor)
        if self.mode == "intersection":
            return (v,0,anchor+r)
        return (v,self.window,anchor+r)  # 90-degree turn plus offset: no intersection

    def register_plane(self, stage, active_vocab):
        """Register spatial locations and update a sparse shared weight store.

        Existing weight values are copied exactly. No reinitialization of
        Task A weights when Task B is registered.
        """
        if stage not in ("A","B"):raise ValueError(stage)
        table = getattr(self, "lookup_"+stage).clone()
        for v in sorted(set(map(int,active_vocab))):
            if v < 0 or v >= table.shape[1]:
                raise ValueError(f"Invalid token ID {v}")
            for c in range(2):
                for r in range(self.window):
                    coord = self._coordinate(stage,c,v,r)
                    if coord not in self.coord_to_id:
                        self.coord_to_id[coord] = len(self.coord_to_id)
                    table[c,v,r] = self.coord_to_id[coord]
        setattr(self,"lookup_"+stage,table)
        marker = getattr(self,stage.lower()+"_seen")
        marker[torch.as_tensor(sorted(set(active_vocab)),dtype=torch.long,device=marker.device)] = True
        old = self.lattice
        ncoords=len(self.coord_to_id)
        if old is None:
            self.lattice = nn.Embedding(ncoords,1,sparse=True)
            nn.init.normal_(self.lattice.weight,std=.025)
        elif ncoords>old.num_embeddings:
            replacement = nn.Embedding(ncoords,1,sparse=True,device=old.weight.device)
            nn.init.normal_(replacement.weight,std=.025)
            with torch.no_grad():
                replacement.weight[:old.num_embeddings].copy_(old.weight)
            self.lattice = replacement
        return ncoords-(old.num_embeddings if old is not None else 0)

    def forward(self,tokens,stage):
        if stage not in ("A","B"):raise ValueError(stage)
        v = tokens.unfold(1,self.window,1)
        ids = getattr(self,"lookup_"+stage)
        valid = (v>=0) & (v<ids.shape[1])
        clipped = v.clamp(0,ids.shape[1]-1)
        pos = torch.arange(self.window,device=tokens.device)
        ids_selected = ids[:,clipped,pos]     # [class,batch,windows,position]
        valid_cells = (ids_selected>=0)&valid.unsqueeze(0)
        safe_ids = ids_selected.clamp(min=0)
        selected = self.lattice(safe_ids).squeeze(-1).tanh()
        selected = selected*valid_cells.float()
        result = selected.sum(-1).permute(1,2,0)/math.sqrt(self.window)
        return result + self.bias[0 if stage=="A" else 1]

    def shared_coordinates(self):
        a = set(self.lookup_A[self.lookup_A>=0].tolist())
        b = set(self.lookup_B[self.lookup_B>=0].tolist())
        return len(a&b)


def eval_task(model,x,y,stage):
    model.eval()
    predictions=[]
    with torch.no_grad():
        for i in range(0,len(y),64):
            data=torch.from_numpy(x[i:i+64])
            p=model(data,stage).softmax(-1).mean(1).argmax(-1)
            predictions.extend(p.cpu().numpy().tolist())
    pred=np.asarray(predictions)
    cm=np.zeros((2,2),dtype=np.int64)
    for actual,guess in zip(y,pred):
        cm[int(actual),int(guess)]+=1
    return {"accuracy":float((pred==y).mean()),"count":int(len(y)),"confusion":cm.tolist()}


def train_stage(model,values,labels,stage,epochs,seed,lr,batch):
    loader=DataLoader(TensorDataset(torch.from_numpy(values),torch.from_numpy(labels)),
                      shuffle=True,batch_size=batch,
                      generator=torch.Generator().manual_seed(seed))
    # SparseAdam updates only lookup coordinates that are activated by this task.
    opt_w=torch.optim.SparseAdam(model.lattice.parameters(),lr=lr)
    opt_b=torch.optim.Adam([model.bias],lr=lr)
    history=[]
    for epoch in range(1,epochs+1):
        model.train()
        loss_total=0.
        for xx,yy in loader:
            opt_w.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            logits=model(xx,stage)
            desired=yy[:,None].expand(-1,logits.shape[1])
            loss=nn.functional.cross_entropy(logits.reshape(-1,2),desired.reshape(-1))
            loss.backward()
            opt_w.step()
            opt_b.step()
            loss_total+=float(loss.detach())*len(yy)
        loss_avg=loss_total/len(values)
        history.append({"stage":stage,"epoch":epoch,"loss":loss_avg})
        print(f"{model.mode}: {stage}, epoch {epoch}/{epochs}, loss={loss_avg:.5f}",flush=True)
    return history


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    X,y=load_integer_eeg()
    itr,ite=split_by_label(X,y,seed=args.seed)
    tasks={}
    for name,cls in TASK_CLASSES.items():
        tr=itr[np.isin(y[itr],cls)]
        te=ite[np.isin(y[ite],cls)]
        tasks[name]=((X[tr],(y[tr]==cls[1]).astype(np.int64)),
                     (X[te],(y[te]==cls[1]).astype(np.int64)))
    vocab=GrowingVocab()
    vocab.extend(tasks["A"][0][0])
    lattice=RotatedLattice(len(vocab.values),window=args.window,mode=args.mode)
    a_ids=np.unique(vocab.encode(tasks["A"][0][0]))
    lattice.register_plane("A",a_ids)
    count_a=lattice.lattice.num_embeddings
    print(f"Initialized A, vocabulary={len(vocab.values)}, coordinates={count_a}",flush=True)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    metrics={};logs=[]
    for iteration,stage in enumerate(("A","B")):
        if stage=="B":
            new=vocab.extend(tasks["B"][0][0])
            lattice.extend_vocab(len(vocab.values))
            b_ids=np.unique(vocab.encode(tasks["B"][0][0]))
            added=lattice.register_plane("B",b_ids)
            print(f"Rotated B by 90 degrees, vocab-added={new}, new coordinates={added}",flush=True)
        tx,ty=tasks[stage][0]
        logs+=train_stage(lattice,vocab.encode(tx),ty,stage,
                          args.epochs_a if stage=="A" else args.epochs_b,
                          args.seed+iteration,args.lr,args.batch)
        torch.save({
            "spatial_weights":lattice.lattice.weight.detach().clone(),
            "bias":lattice.bias.detach().clone(),
            "lookup_A":lattice.lookup_A.clone(),
            "lookup_B":lattice.lookup_B.clone(),
            "vocabulary":vocab.values,
            "coordinate_count":lattice.lattice.num_embeddings,
            "stage":stage,"mode":args.mode,
            "window":args.window
        },out/f"weights_after_{stage}.pt")
        metrics[stage]={}
        for test_task in ("A","B"):
            data,label=tasks[test_task][1]
            encoded=vocab.encode(data)
            metrics[stage][test_task]=eval_task(lattice,encoded,label,test_task)
            print(f"After {stage} test {test_task}: {metrics[stage][test_task]['accuracy']:.4%}",flush=True)
    forgetting=100*(metrics["A"]["A"]["accuracy"]-metrics["B"]["A"]["accuracy"])
    report={
        "dataset":"UCI Epileptic Seizure Recognition native integer EEG",
        "seed":args.seed,"mode":args.mode,
        "protocol":"known task orientation, A->B->retest A, binary two-class heads per task",
        "geometry":{
            "axes":["integer_vocabulary","sequence_position_or_perpendicular_axis","third_axis"],
            "A_class_c":"(v,r,c*(m+1))",
            "B_class_c":"(v,0,c*(m+1)+r)" if args.mode=="intersection" else "(v,m,c*(m+1)+r)",
            "shared_row_positions":0 if args.mode=="isolated" else 1,
            "shared_coordinate_count":lattice.shared_coordinates()
        },
        "epochs_A":args.epochs_a,"epochs_B":args.epochs_b,"lr":args.lr,
        "window":args.window,"vocab_size":len(vocab.values),
        "active_spatial_coordinates":lattice.lattice.num_embeddings,
        "weight_values_saved_per_coordinate":1,
        "parameter_count":sum(p.numel() for p in lattice.parameters()),
        "metrics":metrics,"forgetting_A_percentage_points":forgetting,
        "limitations":[
            "Task identity must be supplied to select orientation at inference; never inferred from target label.",
            "Separate task biases are used so that only genuine lattice intersections can interfere with A.",
            "3D rotation is a coordinate layout; no new feature-sharing learning algorithm is implied.",
            "80/20 split is at EEG segment level, not patient level."
        ]
    }
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"accuracy_matrix.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["after","test","accuracy"])
        for st in ("A","B"):
            for task in ("A","B"):w.writerow([st,task,metrics[st][task]["accuracy"]])
    with (out/"train_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["stage","epoch","loss"])
        w.writeheader();w.writerows(logs)
    print(f"MODE={args.mode}; Forgetting A: {forgetting:.3f} pp",flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--mode",choices=["intersection","isolated"],default="intersection")
    p.add_argument("--output",default=None)
    p.add_argument("--epochs-a",type=int,default=20)
    p.add_argument("--epochs-b",type=int,default=20)
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--lr",type=float,default=.005)
    p.add_argument("--batch",type=int,default=128)
    args=p.parse_args()
    if args.output is None:
        args.output=f"results/rotated_lattice_{args.mode}_seed{args.seed}"
    run(args)
