"""Observation-level SeFT, supervised mTAND-Enc, and GRU-D comparators."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_seft_event import SeFTEvent
from models.cem_mtan_reference import enc_mtan_classif

class MeasurementSeFT(SeFTEvent):
    def __init__(self,dim):super().__init__(dim,1,tasks=1)
    def forward(self,types,times,values):
        # The official demographic token receives a constant zero because this
        # shared information contract has no additional demographic covariates.
        continuous=torch.cat([torch.sin(times[:,:,None]*24/self.timescales),torch.cos(times[:,:,None]*24/self.timescales),values[:,:,None]],-1)
        static=self.source_encoder(torch.zeros(len(types),1,device=types.device))
        mask=torch.cat([torch.ones_like(types[:,:1],dtype=torch.bool),types.ne(0)],1)
        phi=self.phi(continuous,types,static);psi=self.psi(continuous,types,static)
        context=self.context((psi*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True))
        keys=torch.cat([self.key.dense(static)[:,None],self.key(continuous,types)],1)+self.context_key(context)[:,None]
        keys=keys.reshape(*keys.shape[:2],self.heads,self.dot_dim)
        scores=(keys*self.query[None,None]).sum(-1)/math.sqrt(self.dot_dim)
        if self.training:scores=scores-(torch.rand_like(scores)<self.attn_dropout).to(scores.dtype)*1e9
        weights=scores.masked_fill(~mask[:,:,None],float('-inf')).softmax(1)
        return self.rho(torch.einsum('blh,bld->bhd',weights,phi).flatten(1))

class SparseMTAND(enc_mtan_classif):
    def __init__(self,dim,small=False):
        super().__init__(dim,torch.linspace(0,1,8 if small else 128),nhidden=8 if small else 256,embed_time=16 if small else 128,num_heads=1,learn_emb=True,device='cpu')
    def forward(self,types,times,values):
        self.device=times.device
        key=self.att.linears[1](self.learn_time_embedding(times))
        query=self.att.linears[0](self.learn_time_embedding(self.query.to(device=times.device,dtype=times.dtype)[None]))
        scores=torch.einsum('bqd,bld->bql',query,key)/math.sqrt(self.embed_time)
        valid=types.ne(0);index=(types-1).clamp_min(0)[:,None].expand(-1,scores.shape[1],-1)
        masked=scores.masked_fill(~valid[:,None],-1e9)
        maxima=torch.full((len(types),scores.shape[1],self.dim),-1e9,device=times.device,dtype=scores.dtype)
        maxima=maxima.scatter_reduce(2,index,masked,reduce='amax',include_self=True)
        shifted=masked-maxima.gather(2,index);w=torch.where(valid[:,None],shifted.exp(),torch.zeros_like(shifted))
        # Empty feature groups produce no measurements or mask contribution.
        w=torch.nan_to_num(w);den=torch.zeros_like(maxima).scatter_add(2,index,w)
        num=torch.zeros_like(maxima).scatter_add(2,index,w*values[:,None])
        pooled=torch.cat([num/den.clamp_min(1e-20),(den>0).to(num.dtype)],-1)
        hidden=self.att.linears[-1](pooled).permute(1,0,2);_,h=self.enc(hidden);z=self.classifier(h.squeeze(0))
        return (z[:,1]-z[:,0])[:,None]

class GRUD(nn.Module):
    """GRU-D decay equations with train-mean imputation and mask-aware gates.

    Paper-equation PyTorch implementation. Glorot decay initialization avoids
    the permanently zero ReLU derivative of the obsolete zero-initialized port.
    This initialization is fixed before any validation result is observed.
    """
    def __init__(self,dim,num_types,width=64):
        super().__init__();self.width=width;self.dropout=.3
        self.register_buffer('mean',torch.cat([torch.ones(num_types),torch.zeros(dim-num_types)]))
        self.decay_x_weight=nn.Parameter(torch.empty(dim));self.decay_x_bias=nn.Parameter(torch.zeros(dim))
        self.decay_h=nn.Linear(dim,width);self.x_gate=nn.Linear(dim,3*width);self.m_gate=nn.Linear(dim,3*width,bias=False)
        self.h_gate=nn.Linear(width,3*width,bias=False);self.head=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
        nn.init.uniform_(self.decay_x_weight,-math.sqrt(3/dim),math.sqrt(3/dim))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.h_gate.weight)
    def forward(self,x,mask,delta):
        h=x.new_zeros(len(x),self.width);last=self.mean[None].expand(len(x),-1)
        def dropout(shape):return F.dropout(x.new_ones(shape),self.dropout,self.training)
        dx=dropout((3,len(x),x.shape[-1]));dm=dropout((3,len(x),x.shape[-1]));dh=dropout((3,len(x),self.width))
        for t in range(x.shape[1]):
            m=mask[:,t];gx=torch.exp(-F.relu(delta[:,t]*self.decay_x_weight+self.decay_x_bias));gh=torch.exp(-F.relu(self.decay_h(delta[:,t])))
            imputed=m*x[:,t]+(1-m)*(gx*last+(1-gx)*self.mean);hd=gh*h
            xx=[F.linear(imputed*dx[k],self.x_gate.weight[k*self.width:(k+1)*self.width],self.x_gate.bias[k*self.width:(k+1)*self.width]) for k in range(3)]
            mm=[F.linear(m*dm[k],self.m_gate.weight[k*self.width:(k+1)*self.width]) for k in range(3)]
            z=torch.sigmoid(xx[0]+mm[0]+F.linear(hd*dh[0],self.h_gate.weight[:self.width]))
            r=torch.sigmoid(xx[1]+mm[1]+F.linear(hd*dh[1],self.h_gate.weight[self.width:2*self.width]))
            candidate=torch.tanh(xx[2]+mm[2]+F.linear(r*hd*dh[2],self.h_gate.weight[2*self.width:]))
            h=(1-z)*hd+z*candidate;last=torch.where(m.bool(),x[:,t],last)
        return self.head(h)

def cpu_checks():
    torch.manual_seed(48)
    # Sparse mTAND and its original dense implementation agree in outputs and
    # parameter gradients on irregular, partially observed measurements.
    m=SparseMTAND(3,small=True).double();m.eval();times=torch.tensor([[.1,.4,.7]],dtype=torch.double)
    values=torch.tensor([[[1.,0.,2.],[0.,3.,0.],[4.,0.,5.]]],dtype=torch.double);mask=values.ne(0).double()
    reference=enc_mtan_classif(3,m.query.double(),nhidden=8,embed_time=16,num_heads=1,learn_emb=True,device='cpu').double()
    reference.load_state_dict(m.state_dict());reference.eval();dense=reference(torch.cat([values,mask],-1),times);dense=dense[:,1]-dense[:,0]
    row,feature=torch.nonzero(mask[0],as_tuple=True);sp=m((feature+1)[None],times[:,row],values[0,row,feature][None]).squeeze(-1)
    torch.testing.assert_close(sp,dense,rtol=1e-10,atol=1e-10);sp.sum().backward();dense.sum().backward()
    for (n,p),(nr,pr) in zip(m.named_parameters(),reference.named_parameters()):
        assert n==nr;torch.testing.assert_close(p.grad,pr.grad,rtol=1e-8,atol=1e-9)
    seft=MeasurementSeFT(4).eval();types=torch.tensor([[1,2,3,0]]);tt=torch.tensor([[.2,.4,.8,0.]]);vv=torch.tensor([[1.,2.,-1.,0.]])
    with torch.no_grad():torch.testing.assert_close(seft(types,tt,vv),seft(types.flip(1),tt.flip(1),vv.flip(1)),atol=1e-6,rtol=1e-6)
    seft(types,tt,vv).sum().backward()
    for n,p in seft.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if n.startswith(('source_encoder.0.','source_encoder.2.weight')):assert not p.grad.any()
    g=GRUD(4,2,width=8).eval();x=torch.randn(2,5,4);ms=torch.randint(2,(2,5,4)).float();dt=torch.ones_like(x)
    z=g(x,ms,dt);z.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    with torch.no_grad():
        changed=x+(1-ms)*100;torch.testing.assert_close(g(x,ms,dt),g(changed,ms,dt),rtol=0,atol=0)
    return ['mTAND_sparse_dense_output_and_gradient_equivalence','SeFT_measurement_set_permutation_invariance','GRUD_missing_values_do_not_leak','GRUD_decay_gradients_connected','native_measurement_times_values_masks']
