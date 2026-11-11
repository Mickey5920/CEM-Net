"""STraTS ss- adapter; author classes are loaded from the pinned MIT source."""
import argparse
import ast
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.cem_replacement_common import ROOT, RUN

def author_classes():
    ns=dict(torch=torch,nn=nn,F=F,np=np,argparse=argparse,Namespace=argparse.Namespace)
    folder=ROOT/'work'/RUN/'vendor/strats/src'
    for filename,names in [('models.py',{'TimeSeriesModel'}),('modeling_strats.py',{'CVE','FusionAtt','Transformer','Strats'})]:
        tree=ast.parse((folder/filename).read_text())
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        exec(compile(tree,str(folder/filename),'exec'),ns)
    return ns

AUTHOR=author_classes()

def safe_transformer(self,x,mask):
    # Preserve the author's unscaled attention and before-softmax dropout.
    # Saturating the mask prevents min_float + min_float -> -inf on padding.
    bsz,length,_=x.shape;minimum=torch.finfo(x.dtype).min
    mask=(1-mask[:,:,None]*mask[:,None,:])[:,None,:,:]*minimum
    for i in range(self.N):
        q=torch.einsum('bld,hde->bhle',x,self.Wq[i]);k=torch.einsum('bld,hde->bhle',x,self.Wk[i])
        v=torch.einsum('bld,hde->bhle',x,self.Wv[i]);A=torch.einsum('bhle,bhke->bhlk',q,k)
        layer_mask=mask
        if self.training:layer_mask=(mask+(torch.rand_like(A)<self.attention_dropout).float()*minimum).clamp_min(minimum)
        A=torch.softmax(A+layer_mask,dim=-1)
        v=torch.einsum('bhkl,bhle->bkhe',A,v).reshape(bsz,length,-1)
        x=(F.dropout(v@self.Wo[i],self.dropout,self.training)+x)/2
        z=F.gelu(x@self.W1[i]+self.b1[i])@self.W2[i]+self.b2[i]
        x=(F.dropout(z,self.dropout,self.training)+x)/2
    return x

class STraTS(AUTHOR['Strats']):
    def __init__(self,variables):
        args=argparse.Namespace(D=1,V=variables,hid_dim=64,num_layers=2,num_heads=16,
            dropout=.2,attention_dropout=.2,model_type='strats',pretrain=0,
            load_ckpt_path=None,pos_class_weight=1.)
        super().__init__(args)
        self.transformer.forward=types.MethodType(safe_transformer,self.transformer)
    def forward(self,types_,times,values):
        mask=(types_>0).to(values.dtype);varis=(types_-1).clamp_min(0)
        if self.training:
            dropped=torch.rand((len(values),self.V),device=values.device)<=self.dropout
            mask=mask*(~dropped.gather(1,varis))
        demo=torch.zeros((len(values),1),device=values.device)
        x=self.cve_time(times)+self.cve_value(values)+self.variable_emb(varis)
        x=F.dropout(x,self.dropout,self.training);x=self.transformer(x,mask)
        ts=(x*self.fusion_att(x,mask)[:,:,None]).sum(1)
        # No-observation histories remain in the cohort; use zero evidence.
        ts=ts*(mask.sum(1)>0)[:,None]
        return self.binary_head(torch.cat((ts,self.demo_emb(demo)),dim=-1))

def make_model(variant,seed,data):
    assert variant=='STraTS_supervised'
    torch.manual_seed(seed)
    return STraTS(data.input_dim)

class AuthorAdamW(torch.optim.Optimizer):
    """Legacy transformers AdamW used by the author (epsilon before bias correction)."""
    def __init__(self,params,lr=5e-4):
        super().__init__(params,dict(lr=lr,betas=(.9,.999),eps=1e-6,weight_decay=0.))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for g in self.param_groups:
            b1,b2=g['betas']
            for p in g['params']:
                if p.grad is None:continue
                s=self.state[p]
                if not s:s.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                s['step']+=1;s['exp_avg'].mul_(b1).add_(p.grad,alpha=1-b1)
                s['exp_avg_sq'].mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
                scale=g['lr']*(1-b2**s['step'])**.5/(1-b1**s['step'])
                p.addcdiv_(s['exp_avg'],s['exp_avg_sq'].sqrt().add_(g['eps']),value=-scale)
        return loss

def cpu_checks():
    torch.set_num_threads(2);torch.manual_seed(48);m=STraTS(9).eval()
    ref=AUTHOR['Strats'](m.args).eval();ref.load_state_dict(m.state_dict())
    ty=torch.tensor([[1,2,3],[2,5,8]]);t=torch.tensor([[.1,.4,.9],[0,.3,.4]]);v=torch.randn(2,3)
    a=m(ty,t,v).squeeze(-1).sigmoid();b=ref(v,t,ty-1,torch.ones_like(v),torch.zeros(2,1))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward();ga={n:p.grad.clone() for n,p in m.named_parameters()}
    b.sum().backward()
    for n,p in ref.named_parameters():torch.testing.assert_close(ga[n],p.grad,rtol=0,atol=0)
    ty2=F.pad(ty,(0,2));t2=F.pad(t,(0,2));v2=F.pad(v,(0,2))
    torch.testing.assert_close(m(ty2,t2,v2).squeeze(-1).sigmoid(),a,rtol=2e-6,atol=2e-7)
    assert torch.isfinite(m(torch.zeros_like(ty),t,v)).all()
    m.train()
    for _ in range(4):
        m.zero_grad();z=m(ty2,t2,v2);z.sum().backward()
        assert torch.isfinite(z).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    return ['author_inference_and_gradient_exact','padding_invariance','empty_history_finite','dropout_padding_gradients_finite']
