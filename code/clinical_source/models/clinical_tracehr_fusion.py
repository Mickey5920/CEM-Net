"""Clinical adaptations of pooled-token fusion, MIAM and InfMasking.

These are adaptations, not claims of paper-faithful benchmark reproduction.
MIAM: ICLR 2026, Sec. 3.2; beta corners and score/derivative ratios.
InfMasking: NeurIPS 2025, Sec. 3 and the authors' Gaussian masking loss.
Here augmentations mask latent channels, and SSL is a weighted auxiliary term
alongside clinical BCE. Both branches see exactly the existing retained events.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def symmetric_nce(a,b,temperature=.1):
    a,b=F.normalize(a,dim=-1),F.normalize(b,dim=-1)
    n=a.shape[0];z=torch.cat([a,b]);sim=z@z.T/temperature
    sim=sim.masked_fill(torch.eye(2*n,device=z.device,dtype=torch.bool),-torch.inf)
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(sim,target)


def gaussian_mask_loss(q,views,ratio,temperature=.1):
    """Gaussian mask approximation, including covariance correction and negatives.

    Computes q^T Sigma q via projected centered views, avoiding B x D x D storage.
    Algebraically identical to the quadratic-form expression in the reference.
    """
    q=F.normalize(q,dim=-1);keys=F.normalize(torch.stack(views),dim=-1)
    mean=keys.mean(0);center=keys-mean
    quad=torch.einsum('knd,nd->kn',center,q).square().mean(0)*ratio/temperature
    positive=(q*mean).sum(-1)+.5*quad
    n=q.shape[0];eye=torch.eye(n,device=q.device,dtype=torch.bool)
    cross=torch.einsum('nd,kmd->knm',q,keys).masked_fill(eye[None],-torch.inf)
    negatives=cross.permute(1,0,2).flatten(1)
    within=(q@q.T).masked_fill(eye,-torch.inf)
    logits=torch.cat([positive[:,None],negatives,within],1)/temperature
    return F.cross_entropy(logits,torch.zeros(n,device=q.device,dtype=torch.long))+.5*quad.mean()/temperature


class MIAMController(nn.Module):
    def __init__(self,kappa=10.,strength=1.,priority_mass=.4):
        super().__init__();self.kappa=kappa;self.strength=strength
        self.register_buffer('rho_s',torch.ones(2));self.register_buffer('rho_d',torch.ones(2))
        self.register_buffer('previous',torch.full((2,),float('nan')))
        self.register_buffer('corners',torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.bool))
        self.register_buffer('corner_weights',torch.tensor([priority_mass,.5-priority_mass,.5-priority_mass,priority_mass]))
    @torch.no_grad()
    def update(self,scores):
        scores=torch.as_tensor(scores,device=self.rho_s.device,dtype=torch.float32).clamp_min(1e-5)
        self.rho_s.copy_(scores/scores.log().mean().exp())
        if torch.isfinite(self.previous).all():
            delta=(scores-self.previous).abs().clamp_min(1e-5)
            self.rho_d.copy_(delta/delta.log().mean().exp())
        self.previous.copy_(scores)
    def sample(self):
        # Bounded ratio handles zero/noisy derivatives in the clinical adaptation.
        w=(self.rho_s/self.rho_d).pow(self.strength).clamp(.05,20.)
        c=self.corners[torch.multinomial(self.corner_weights,1).squeeze(0)]
        alpha=torch.where(c,self.kappa*w,torch.ones_like(w))
        beta=torch.where(c,torch.ones_like(w),self.kappa/w)
        return torch.distributions.Beta(alpha,beta).sample()


class PooledFusion(nn.Module):
    def __init__(self,dim,dropout,kind):
        super().__init__();self.kind=kind
        if kind=='concat':
            self.net=nn.Sequential(nn.Linear(2*dim,2*dim),nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(2*dim,dim),nn.LayerNorm(dim))
        else:
            self.cls=nn.Parameter(torch.zeros(1,1,dim))
            self.types=nn.Parameter(torch.randn(1,2,dim)*.02)
            layer=nn.TransformerEncoderLayer(dim,4,dim*2,dropout,activation='gelu',batch_first=True,norm_first=True)
            self.net=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.norm=nn.LayerNorm(dim)
    def forward(self,tokens,absent=None):
        if self.kind=='concat':return self.net(tokens.flatten(1))
        b=tokens.shape[0]
        x=torch.cat([self.cls.expand(b,-1,-1),tokens+self.types],1)
        mask=None
        if absent is not None:
            mask=torch.cat([torch.zeros(b,1,device=tokens.device,dtype=torch.bool),absent],1)
        return self.norm(self.net(x,src_key_padding_mask=mask)[:,0])


class FusionCandidate(nn.Module):
    def __init__(self,dense,temporal,mode,dim=96,dropout=.15,aux_weight=.01,miam_strength=1.):
        super().__init__()
        if mode not in ['concat','simbamm','miam','infmasking','random_mask','inf_no_gaussian','no_temporal']:
            raise ValueError(mode)
        self.dense=dense;self.temporal=temporal;self.mode=mode
        self.fusion=PooledFusion(dim,dropout,'concat' if mode=='concat' else 'transformer')
        self.head=copy.deepcopy(dense.head)
        self.controller=MIAMController(strength=miam_strength)
        self.aux_weight=float(aux_weight);self.aux_loss=None
        self.epoch=1;self.max_epochs=50;self.probe_branch=None
        self.masked_fraction=torch.zeros(2);self.mask_batches=0
    def branch_tokens(self,event_type,event_time,time_gap,source_features):
        s,_=self.dense.encoder(event_type,event_time,time_gap=time_gap,source_features=source_features)
        t,hidden=self.temporal.encoder(event_type,event_time,time_gap=time_gap,source_features=source_features)
        return torch.stack([s,t],1),hidden
    def fuse(self,tokens,force=None):
        absent=None
        if force is not None:
            tokens=tokens.clone();absent=torch.zeros(tokens.shape[:2],device=tokens.device,dtype=torch.bool)
            absent[:,1 if force=='dense' else 0]=True;tokens=tokens.masked_fill(absent[:,:,None],0.)
        return self.fusion(tokens,absent)
    def inf_loss(self,tokens):
        # EHR-specific adaptation: latent dropout rather than image/text augmentation.
        a=F.dropout(tokens,.1,True);b=F.dropout(tokens,.1,True)
        za=self.fuse(a);zb=self.fuse(b)
        terms=[symmetric_nce(za,zb)]
        for m in range(2):
            terms.append(.5*(symmetric_nce(a[:,m],zb)+symmetric_nce(b[:,m],za)))
        if self.mode=='infmasking':
            masked_a=[];masked_b=[]
            for _ in range(6):
                masked_a.append(self.fuse(a*(torch.rand_like(a)>.75)))
                masked_b.append(self.fuse(b*(torch.rand_like(b)>.75)))
            ratio=.25*min(self.epoch/self.max_epochs,1.)
            terms += [gaussian_mask_loss(za,masked_a,ratio),gaussian_mask_loss(zb,masked_b,ratio)]
        # Removing Gaussian terms must not rescale the remaining NCE terms.
        return torch.stack(terms).sum()/5.
    def forward(self,event_type,event_time,time_gap=None,source_features=None):
        tokens,hidden=self.branch_tokens(event_type,event_time,time_gap,source_features)
        self.aux_loss=None
        force=self.probe_branch or ('dense' if self.mode=='no_temporal' else None)
        if self.training and self.mode in ['miam','random_mask'] and force is None:
            prob=self.controller.sample() if self.mode=='miam' else tokens.new_full((2,),.5)
            keep=torch.rand_like(tokens)>prob[None,:,None]
            self.masked_fraction += (1-keep.float().mean((0,2))).detach().cpu();self.mask_batches+=1
            pooled=self.fuse(tokens*keep)
        else:pooled=self.fuse(tokens,force)
        if self.training and self.mode in ['infmasking','inf_no_gaussian'] and force is None:
            self.aux_loss=self.inf_loss(tokens)
        return self.head(pooled),hidden,pooled
