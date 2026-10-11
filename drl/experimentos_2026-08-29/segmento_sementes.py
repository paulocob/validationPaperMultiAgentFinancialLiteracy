"""Decomposicao da acuracia Top-1 por segmento, replicada em cinco sementes.
Substitui a execucao unica da tabela q_segmento."""
import sys, json, statistics as st
sys.path.insert(0,'..')
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from train_utils import build_split_dataset, set_seed
from sweep import make_net, loaders

DATA=Path("../data")
SEG=["iniciante","intermediario","avancado"]

def segmentos_do_teste():
    """Recupera o segmento de cada transicao de teste, na ordem do dataset."""
    sp=json.load(open(DATA/"splits.json"))
    linhas={}
    for l in open(DATA/"interactions.jsonl"):
        d=json.loads(l); linhas[d["interaction_id"]]=d["state"]["profile"]["segment"]
    return [linhas[i] for i in sp["test"]]

@torch.no_grad()
def top1_por_segmento(model, te_loader, segs):
    model.eval(); pred=[]; alvo=[]
    for s,a,r,ns,d in te_loader:
        pred += model(s).argmax(1).tolist(); alvo += a.tolist()
    assert len(pred)==len(segs), f"{len(pred)} != {len(segs)}"
    out={}
    for g in SEG:
        idx=[i for i,x in enumerate(segs) if x==g]
        out[g]=sum(pred[i]==alvo[i] for i in idx)/max(1,len(idx))
    out["GERAL"]=sum(p==t for p,t in zip(pred,alvo))/len(pred)
    return out

def treina(seed, cfg):
    set_seed(seed)
    tr,va,te,adim,idim = loaders(cfg["bs"])
    model=make_net(idim,adim,cfg["hidden"],cfg["depth"],cfg["dropout"])
    target=make_net(idim,adim,cfg["hidden"],cfg["depth"],cfg["dropout"])
    target.load_state_dict(model.state_dict()); target.eval()
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
    import torch.nn as nn
    from sweep import top1
    best=-1; best_state=None; bad=0
    for ep in range(cfg["epochs"]):
        model.train()
        for s,a,r,ns,d in tr:
            q=model(s).gather(1,a.unsqueeze(1)).squeeze(1)
            with torch.no_grad(): tgt=r + cfg["gamma"]*(1-d)*target(ns).max(1).values
            loss=nn.functional.smooth_l1_loss(q,tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%cfg["tu"]==0: target.load_state_dict(model.state_dict())
        acc=top1(model,va)
        if acc>best: best=acc; best_state={k:v.clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=cfg["patience"]: break
    model.load_state_dict(best_state)
    return model, te

if __name__=="__main__":
    cfg=dict(bs=32,gamma=0.9,lr=3e-3,tu=5,epochs=80,patience=12,hidden=256,depth=2,dropout=0.1,wd=0.0,clip=0.0)
    segs=segmentos_do_teste()
    print("transicoes de teste por segmento:",{g:segs.count(g) for g in SEG})
    acum={g:[] for g in SEG+["GERAL"]}
    for seed in [42,43,44,45,46]:
        model,te = treina(seed,cfg)
        r=top1_por_segmento(model,te,segs)
        for g in acum: acum[g].append(r[g])
        print("  semente %d: %s"%(seed, {g:round(r[g],4) for g in SEG+["GERAL"]}))
    print()
    for g in SEG+["GERAL"]:
        print("  %-14s %.3f +/- %.3f"%(g, st.mean(acum[g]), st.pstdev(acum[g])))
    json.dump({g:acum[g] for g in acum}, open("segmento_sementes.json","w"), indent=2)
