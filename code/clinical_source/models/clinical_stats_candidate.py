"""SOAP gradient estimator and light scalar adaptation of numerical PLE.

Sources: NeurIPS 2021 SOAP equations (5)-(7); NeurIPS 2022 numerical embeddings
section 3.2. The PLE coordinates are returned to one scalar, not a reproduction
of the paper's vector embedding. All fitting is restricted to training inputs.
"""
import torch
from torch import nn


class SOAPObjective(nn.Module):
    def __init__(self, n_train, n_tasks, gamma=.1, margin=.6, floor=1e-4):
        super().__init__()
        self.gamma,self.margin,self.floor=gamma,margin,floor
        self.register_buffer('u1',torch.zeros(n_tasks,n_train))
        self.register_buffer('u2',torch.zeros(n_tasks,n_train))
        self.register_buffer('updates',torch.zeros(n_tasks,dtype=torch.long))

    def forward(self, outer_scores, inner_scores, inner_labels, indices, task):
        if len(indices.unique())!=len(indices):raise ValueError('Outer positive IDs must be unique')
        if not len(inner_scores):raise ValueError('No valid natural-distribution inner samples')
        loss=torch.relu(self.margin-(outer_scores[:,None]-inner_scores[None,:])).square()
        with torch.no_grad():
            a=(loss*inner_labels[None,:]).mean(1)
            b=loss.mean(1)
            self.u1[task,indices]=(1-self.gamma)*self.u1[task,indices]+self.gamma*a
            self.u2[task,indices]=((1-self.gamma)*self.u2[task,indices]+self.gamma*b).clamp_min(self.floor)
            numerator=self.u1[task,indices,None]-self.u2[task,indices,None]*inner_labels[None,:]
            coefficient=numerator/self.u2[task,indices,None].square()
            self.updates[task]+=1
        # Its scalar value may cancel. This is an explicit surrogate gradient,
        # not an AP value, and must never be used for early stopping.
        return (coefficient*loss).mean()


class PiecewiseScalar(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()
        self.num_types=num_types
        self.register_buffer('left',torch.zeros(dim,4))
        self.register_buffer('right',torch.ones(dim,4))
        self.register_buffer('valid',torch.zeros(dim,4,dtype=torch.bool))
        self.coefficient=nn.Parameter(torch.zeros(dim,4))
        self.register_buffer('fit_count',torch.tensor(0,dtype=torch.long))

    @torch.no_grad()
    def fit(self, train_features):
        if train_features.ndim!=2 or not torch.isfinite(train_features).all():raise ValueError('Invalid train features')
        n=self.num_types
        for column in range(train_features.shape[1]):
            # Presence remains binary; absent type statistics retain structural zero.
            if n<=column<2*n:continue
            values=train_features[:,column]
            if column<4*n:
                present=train_features[:,n+column%n]>0
                values=values[present]
            if len(values)<2:continue
            knots=torch.quantile(values,torch.linspace(0,1,5,device=values.device)).unique(sorted=True)
            if len(knots)<2:continue
            k=len(knots)-1
            self.left[column,:k]=knots[:-1]
            self.right[column,:k]=knots[1:]
            self.valid[column,:k]=True
        self.fit_count.fill_(len(train_features))

    def forward(self, x):
        if self.fit_count.item()<2:raise ValueError('PLE not fitted to train')
        coordinates=(x[:,:,None]-self.left)/(self.right-self.left).clamp_min(1e-6)
        coordinates=coordinates.clamp(0,1)
        # First and last intervals extrapolate as in the original PLE definition.
        first=self.valid & (self.valid.long().cumsum(1)==1)
        last=self.valid & (self.valid.long().flip(1).cumsum(1).flip(1)==1)
        raw=(x[:,:,None]-self.left)/(self.right-self.left).clamp_min(1e-6)
        coordinates=torch.where(first[None]& (raw<0),raw,coordinates)
        coordinates=torch.where(last[None]& (raw>1),raw,coordinates)
        adjustment=(coordinates*self.valid*self.coefficient).sum(2)
        n=self.num_types
        available=torch.ones_like(x,dtype=torch.bool)
        for group in [0,2,3]:available[:,group*n:(group+1)*n]=x[:,n:2*n]>0
        available[:,n:2*n]=False
        # Zero source counts/proportions stay zero too.
        available[:,4*n:]=x[:,4*n:]!=0
        return x+adjustment*available


class PiecewiseRepresentation(nn.Module):
    def __init__(self, representation):
        super().__init__()
        self.representation=representation
        base=representation.base if hasattr(representation,'base') else representation
        self.calibration=PiecewiseScalar(base.input_dim,base.num_types)

    def features(self, types, times, sources):
        if hasattr(self.representation,'unscaled_features'):
            return self.representation.unscaled_features(types,times,sources)
        return self.representation.build_static_features(types,times,sources)

    def forward(self, event_type, event_time, source_features=None):
        base=self.representation.base if hasattr(self.representation,'base') else self.representation
        f=self.calibration(self.features(event_type,event_time,source_features))
        pooled=base.net(f)
        hidden=pooled[:,None,:].expand(-1,event_type.size(1),-1)*event_type.ne(0)[:,:,None]
        return pooled,hidden
