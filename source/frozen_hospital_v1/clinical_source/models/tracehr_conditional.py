"""Auditable components for summary-conditioned temporal correction."""
import importlib.util
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.build_tracehr_retention import sha


def official_module(folder, name):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['repo']=='shlee-labs/ATENet'
    assert manifest['commit']=='dc3149ca60f02dafd8e073b336479f280efa51e7'
    for file,digest in manifest['files'].items():assert sha(folder/file)==digest
    spec=importlib.util.spec_from_file_location('tracehr_official_ate_'+name,folder/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def device_mask(irr_mask):
    # Preserve the author's CPU Bernoulli stream; only fix hardcoded .cuda().
    ratio=random.randint(1,9)*.1
    temp=(1-torch.bernoulli(torch.full(irr_mask.shape[:2],ratio))).to(irr_mask.device)
    return irr_mask*temp.unsqueeze(-1)


class TemporalEncoder(nn.Module):
    def __init__(self,sources,device='cpu',mask_only=False):
        super().__init__();mod=official_module(sources,'models');self.losses=official_module(sources,'utils')
        self.core=mod.ATE(32,0,nhidden=64,embed_time=128,num_heads=1,learn_emb=True,device=device,n_classes=2,static=False)
        self.mask_only=mask_only

    def forward(self,x,times,augment=False):
        m=self.core;mask=x[:,:,32:]
        if self.mask_only:x=torch.cat([mask,mask],-1)
        query=m.learn_time_embedding(m.query.unsqueeze(0));key=m.learn_time_embedding(times)
        z=m.att_enc(query,key,x,torch.cat([mask,mask],-1))
        _,hidden=m.encoder(z.permute(1,0,2));h=hidden[0]  # Preserves batch=1.
        logits=m.classifier(h);nonempty=mask.sum((1,2))>0
        result=dict(logits=logits,h=h*nonempty[:,None],nonempty=nonempty)
        if augment:
            aug=device_mask(mask);z1=m.att_enc(query,key,x,torch.cat([aug,aug],-1))
            raw=x[:,:,:32]
            sx=torch.round(torch.sigmoid(torch.bmm(raw.transpose(-2,-1),raw)))
            sout=torch.sigmoid(torch.bmm(z.transpose(-2,-1),z))
            result.update(z=z,z1=z1,sx=sx,sout=sout)
        return result

    def loss(self,result,y):
        return (F.cross_entropy(result['logits'],y.long(),label_smoothing=.1)
            +.1*(self.losses.instance_contrastive_loss(result['z'],result['z1'])
                  +self.losses.temporal_contrastive_loss(result['z'],result['z1']))
            +.01*F.binary_cross_entropy(result['sout'],result['sx']))


class UnorderedEncoder(nn.Module):
    """A multiset of individual (variable,value) events; no co-time grouping."""
    def __init__(self):
        super().__init__();self.variable=nn.Embedding(32,16)
        self.token=nn.Sequential(nn.Linear(17,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.pool=nn.Sequential(nn.Linear(128,64),nn.GELU())
        self.classifier=nn.Linear(64,2)

    def forward(self,x,times=None,augment=False):
        observed=x[:,:,32:].bool();batch,time,variable=observed.nonzero(as_tuple=True)
        values=x[batch,time,variable].unsqueeze(-1)
        tokens=self.token(torch.cat([self.variable(variable),values],-1))
        sums=x.new_zeros((len(x),64)).index_add(0,batch,tokens)
        counts=observed.sum((1,2)).to(x.dtype);mean=sums/counts.clamp_min(1)[:,None]
        maximum=x.new_full((len(x),64),-torch.inf)
        maximum.scatter_reduce_(0,batch[:,None].expand(-1,64),tokens,reduce='amax',include_self=True)
        maximum=torch.where((counts>0)[:,None],maximum,torch.zeros_like(maximum))
        h=self.pool(torch.cat([mean,maximum],-1))
        h=h*(counts>0)[:,None]
        return dict(h=h,logits=self.classifier(h),nonempty=counts>0)

    def loss(self,result,y):return F.cross_entropy(result['logits'],y.long(),label_smoothing=.1)


class Nuisance(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,64))
    def forward(self,x):return self.net(x)


class Correction(nn.Module):
    def __init__(self,dim=64):
        super().__init__();self.linear=nn.Linear(dim,1,bias=False);nn.init.zeros_(self.linear.weight)
    def forward(self,h,nonempty):return self.linear(h).squeeze(-1)*nonempty


def rng_state(rng):
    return dict(python=random.getstate(),numpy=np.random.get_state(),generator=rng.bit_generator.state,
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,rng):
    random.setstate(state['python']);np.random.set_state(state['numpy']);rng.bit_generator.state=state['generator']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_checkpoint(path,model,optimizer,rng,epoch,best,stale,history,scheduler=None):
    path=Path(path);temp=path.with_suffix('.writing')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng),epoch=epoch,
        best=best,stale=stale,history=history,scheduler=scheduler.state_dict() if scheduler else None),temp)
    temp.replace(path)


def load_checkpoint(path,model,optimizer,rng,scheduler=None,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'],rng)
    return state
