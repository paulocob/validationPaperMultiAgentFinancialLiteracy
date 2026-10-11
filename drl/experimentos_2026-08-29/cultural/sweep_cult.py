"""DQN + regularizacao por comportamento (estilo TD3+BC): loss = TD + lambda*CE."""
import sys, json, itertools; sys.path.insert(0,'.')
import torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from train_utils import build_split_dataset, set_seed, compute_refined_reward, load_tasks
DATA=Path("data")

def loaders(bs,cw=0.0):
    tr,vocab,idim = build_split_dataset(DATA,"train",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    va,_,_ = build_split_dataset(DATA,"val",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    te,_,_ = build_split_dataset(DATA,"test",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    return (DataLoader(tr,batch_size=bs,shuffle=True),DataLoader(va,batch_size=256),
            DataLoader(te,batch_size=256),len(vocab.task_to_idx),idim,vocab)

def net(idim,adim,h,d):
    L=[nn.Linear(idim,h),nn.ReLU()]
    for _ in range(d-1): L+=[nn.Linear(h,h),nn.ReLU()]
    L.append(nn.Linear(h,adim)); return nn.Sequential(*L)

@torch.no_grad()
def metrics(model,loader,ks=(1,3,5)):
    model.eval(); n=0; hits={k:0 for k in ks}; mr_sum=0.0; mr_n=0
    for s,a,r,ns,d in loader:
        q=model(s); top=q.topk(max(ks),dim=1).indices
        for k in ks: hits[k]+=(top[:,:k]==a.unsqueeze(1)).any(1).sum().item()
        corr=(q.argmax(1)==a); mr_sum+=r[corr].sum().item(); mr_n+=corr.sum().item()
        n+=len(a)
    return {f"top{k}":hits[k]/n for k in ks} | {"reward": mr_sum/max(1,mr_n)}

def run(lam,seed,cw=0.0,h=256,d=2,lr=3e-3,gamma=0.9,tu=5,epochs=80,patience=12,bs=32):
    set_seed(seed)
    tr,va,te,adim,idim,_=loaders(bs,cw)
    m=net(idim,adim,h,d); t=net(idim,adim,h,d); t.load_state_dict(m.state_dict()); t.eval()
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    best=-1; bstate=None; bad=0
    for ep in range(epochs):
        m.train()
        for s,a,r,ns,dn in tr:
            logits=m(s); q=logits.gather(1,a.unsqueeze(1)).squeeze(1)
            with torch.no_grad(): tgt=r+gamma*(1-dn)*t(ns).max(1).values
            td=nn.functional.smooth_l1_loss(q,tgt)
            ce=nn.functional.cross_entropy(logits,a)
            loss=td+lam*ce
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%tu==0: t.load_state_dict(m.state_dict())
        acc=metrics(m,va)["top1"]
        if acc>best: best=acc; bstate={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    m.load_state_dict(bstate)
    return best, metrics(m,te)

if __name__=="__main__":
    print(f"{'lambda':>8s} {'val top1':>9s} {'test top1':>10s} {'test top3':>10s} {'reward':>8s}")
    print("-"*50)
    out=[]
    for lam in (0.0,0.1,0.3,0.5,1.0,2.0,5.0,10.0):
        vs=[];ts=[];r3=[];rw=[]
        for seed in (42,43,44):
            v,mt=run(lam,seed); vs.append(v); ts.append(mt["top1"]); r3.append(mt["top3"]); rw.append(mt["reward"])
        import statistics as st
        print(f"{lam:8.1f} {st.mean(vs):9.3f} {st.mean(ts):10.3f} {st.mean(r3):10.3f} {st.mean(rw):8.3f}")
        out.append({"lambda":lam,"val":st.mean(vs),"test_top1":st.mean(ts),"test_top1_sd":st.stdev(ts),
                    "test_top3":st.mean(r3),"reward":st.mean(rw)})
    json.dump(out,open("sweep_bc_resultados.json","w"),indent=1)
