"""Busca de hiperparâmetros para o DQN, com SELEÇÃO NA VALIDAÇÃO e reporte no TESTE."""
import sys, json, itertools, statistics as st
sys.path.insert(0,'..')
import torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from train_utils import build_split_dataset, set_seed

DATA=Path("../data")
def loaders(bs):
    tr,vocab,idim = build_split_dataset(DATA,"train",use_refined_reward=True,refined_m2_boost=0.05)
    va,_,_        = build_split_dataset(DATA,"val",  use_refined_reward=True,refined_m2_boost=0.05)
    te,_,_        = build_split_dataset(DATA,"test", use_refined_reward=True,refined_m2_boost=0.05)
    return (DataLoader(tr,batch_size=bs,shuffle=True), DataLoader(va,batch_size=256),
            DataLoader(te,batch_size=256), len(vocab.task_to_idx), idim)

def make_net(idim,adim,hidden,depth,dropout):
    L=[nn.Linear(idim,hidden),nn.ReLU()]
    if dropout>0: L.append(nn.Dropout(dropout))
    for _ in range(depth-1):
        L += [nn.Linear(hidden,hidden),nn.ReLU()]
        if dropout>0: L.append(nn.Dropout(dropout))
    L.append(nn.Linear(hidden,adim))
    return nn.Sequential(*L)

@torch.no_grad()
def top1(model,loader):
    model.eval(); ok=n=0
    for s,a,r,ns,d in loader:
        p=model(s).argmax(1); ok+=(p==a).sum().item(); n+=len(a)
    return ok/max(1,n)

def run(cfg,seed):
    set_seed(seed)
    tr,va,te,adim,idim = loaders(cfg["bs"])
    model=make_net(idim,adim,cfg["hidden"],cfg["depth"],cfg["dropout"])
    target=make_net(idim,adim,cfg["hidden"],cfg["depth"],cfg["dropout"])
    target.load_state_dict(model.state_dict()); target.eval()
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
    best_va=-1; best_state=None; bad=0
    for ep in range(cfg["epochs"]):
        model.train()
        for s,a,r,ns,d in tr:
            q=model(s).gather(1,a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt=r + cfg["gamma"]*(1-d)*target(ns).max(1).values
            loss=nn.functional.smooth_l1_loss(q,tgt)
            opt.zero_grad(); loss.backward()
            if cfg["clip"]: nn.utils.clip_grad_norm_(model.parameters(),cfg["clip"])
            opt.step()
        if (ep+1)%cfg["tu"]==0: target.load_state_dict(model.state_dict())
        acc=top1(model,va)                      # <<< seleção por ACURÁCIA, não por perda TD
        if acc>best_va: best_va=acc; best_state={k:v.clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=cfg["patience"]: break
    model.load_state_dict(best_state)
    return best_va, top1(model,te)

if __name__=="__main__":
    base=dict(bs=32,gamma=0.95,lr=1e-3,tu=10,epochs=60,patience=8,hidden=128,depth=2,dropout=0.0,wd=0.0,clip=0.0)
    grid=dict(lr=[3e-4,1e-3,3e-3], gamma=[0.9,0.95,0.99], hidden=[128,256],
              depth=[2,3], dropout=[0.0,0.1], tu=[5,10])
    keys=list(grid); res=[]
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg=dict(base); cfg.update(dict(zip(keys,vals)))
        v,t = run(cfg,42)
        res.append((v,t,cfg))
    res.sort(key=lambda x:-x[0])
    print(f"{'val':>7s} {'test':>7s}  config")
    for v,t,c in res[:12]:
        print(f"{v:7.3f} {t:7.3f}  lr={c['lr']:<7g} gamma={c['gamma']:<5g} h={c['hidden']:<4d} d={c['depth']} drop={c['dropout']} tu={c['tu']}")
    json.dump([{"val":v,"test":t,"cfg":c} for v,t,c in res],open("sweep_resultados.json","w"),indent=1)
    print(f"\nconfigurações testadas: {len(res)}")
