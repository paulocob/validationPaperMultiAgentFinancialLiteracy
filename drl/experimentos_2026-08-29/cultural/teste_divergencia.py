"""Treina com pais no estado e mede se a politica DIVERGE por pais."""
import sys, json, statistics as st; sys.path.insert(0,'.')
import torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from train_utils import build_split_dataset, set_seed, load_tasks, PAIS_TO_IDX, PAISES
DATA=Path("data")

def net(i,a,h=256,d=2):
    L=[nn.Linear(i,h),nn.ReLU()]
    for _ in range(d-1): L+=[nn.Linear(h,h),nn.ReLU()]
    L.append(nn.Linear(h,a)); return nn.Sequential(*L)

@torch.no_grad()
def top1(m,l):
    m.eval(); ok=n=0
    for s,a,r,ns,d in l: ok+=(m(s).argmax(1)==a).sum().item(); n+=len(a)
    return ok/max(1,n)

def treinar(seed,cw,lam=0.05,epochs=80,patience=12):
    set_seed(seed)
    tr,vocab,idim=build_split_dataset(DATA,"train",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    va,_,_=build_split_dataset(DATA,"val",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    te,_,_=build_split_dataset(DATA,"test",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=cw)
    trl=DataLoader(tr,batch_size=32,shuffle=True); val=DataLoader(va,batch_size=256); tel=DataLoader(te,batch_size=256)
    adim=len(vocab.task_to_idx)
    m=net(idim,adim); t=net(idim,adim); t.load_state_dict(m.state_dict()); t.eval()
    opt=torch.optim.Adam(m.parameters(),lr=3e-3); best=-1; bs_=None; bad=0
    for ep in range(epochs):
        m.train()
        for s,a,r,ns,dn in trl:
            lg=m(s); q=lg.gather(1,a.unsqueeze(1)).squeeze(1)
            with torch.no_grad(): tg=r+0.9*(1-dn)*t(ns).max(1).values
            loss=nn.functional.smooth_l1_loss(q,tg)+lam*nn.functional.cross_entropy(lg,a)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%5==0: t.load_state_dict(m.state_dict())
        acc=top1(m,val)
        if acc>best: best=acc; bs_={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    m.load_state_dict(bs_)
    return m, top1(m,tel), vocab, idim

@torch.no_grad()
def divergencia(m, vocab, idim):
    """Para cada estado de teste, varia SO o one-hot de pais e conta mudancas de acao."""
    te,_,_=build_split_dataset(DATA,"test",use_refined_reward=True,refined_m2_boost=0.05,cultural_weight=0.15)
    loader=DataLoader(te,batch_size=256)
    base=idim-len(PAISES)
    muda=0; tot=0; pref={p:{} for p in PAISES}
    for s,a,r,ns,d in loader:
        for i in range(len(s)):
            acoes=[]
            for p,idx in PAIS_TO_IDX.items():
                x=s[i].clone(); x[base:]=0.0; x[base+idx]=1.0
                act=int(m(x.unsqueeze(0)).argmax(1))
                acoes.append(act); pref[p][act]=pref[p].get(act,0)+1
            if len(set(acoes))>1: muda+=1
            tot+=1
    return muda/tot, pref

if __name__=="__main__":
    print("PAÍS NO ESTADO (37 dims) + recompensa cultural — 5 sementes\n")
    t1=[]; dv=[]
    for seed in (42,43,44,45,46):
        m,acc,vocab,idim=treinar(seed,0.15)
        d,pref=divergencia(m,vocab,idim)
        t1.append(acc); dv.append(d)
    print(f"  Top-1 no teste          : {st.mean(t1):.3f} ± {st.stdev(t1):.3f}")
    print(f"  Estados em que a ação MUDA ao trocar só o país: {100*st.mean(dv):.1f}% ± {100*st.stdev(dv):.1f}")
    json.dump({"top1":st.mean(t1),"sd":st.stdev(t1),"divergencia":st.mean(dv),"div_sd":st.stdev(dv)},
              open("resultado_pais_no_estado.json","w"),indent=1)
