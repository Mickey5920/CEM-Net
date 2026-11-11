"""SeFT attention adapted to retained event identities, timestamps and sources.

Architecture reference: Horn et al., ICML 2020; official DeepSetAttentionModel.
This is a PyTorch event-input adaptation, not a measurement-value reproduction.
The lookup implementation is exactly a linear projection of categorical one-hot
inputs, avoiding a large [batch, length, vocabulary] intermediate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EventProjection(nn.Module):
    def __init__(self, num_types, continuous_dim, width, bias=True):
        super().__init__()
        self.continuous_dim=continuous_dim
        self.weight=nn.Parameter(torch.empty(width,continuous_dim+num_types))
        self.bias=nn.Parameter(torch.zeros(width)) if bias else None
        nn.init.kaiming_uniform_(self.weight,nonlinearity='relu')

    def forward(self, continuous, event_type):
        z=F.linear(continuous,self.weight[:,:self.continuous_dim],self.bias)
        return z+F.embedding((event_type-1).clamp_min(0),self.weight[:,self.continuous_dim:].t())

    def dense(self, values):return F.linear(values,self.weight,self.bias)


class PointMLP(nn.Module):
    def __init__(self,num_types,continuous_dim,width,layers,latent,dropout):
        super().__init__()
        self.first=EventProjection(num_types,continuous_dim,width)
        blocks=[nn.ReLU(),nn.Dropout(dropout)]
        for _ in range(layers-1):blocks += [nn.Linear(width,width),nn.ReLU(),nn.Dropout(dropout)]
        blocks += [nn.Linear(width,latent),nn.ReLU()]
        self.tail=nn.Sequential(*blocks)
        for m in self.tail:
            if isinstance(m,nn.Linear):nn.init.kaiming_uniform_(m.weight,nonlinearity='relu');nn.init.zeros_(m.bias)

    def forward(self,continuous,event_type,static):
        events=self.first(continuous,event_type)
        return self.tail(torch.cat([self.first.dense(static)[:,None],events],1))


class SeFTEvent(nn.Module):
    def __init__(self,num_types,source_dim,tasks=2,phi_width=32,rho_width=32,dropout=0.,
                 latent=128,heads=4,dot_dim=128,attn_dropout=.1,max_timescale=100.,time_dims=4):
        super().__init__()
        if time_dims%2:raise ValueError('Even sinusoidal dimension required')
        self.register_buffer('timescales',torch.logspace(0,math.log10(max_timescale),time_dims//2))
        self.heads,self.dot_dim,self.attn_dropout=heads,dot_dim,attn_dropout
        continuous_dim=time_dims+1
        # Existing source count/proportion features replace unavailable static
        # demographics. This adaptation is explicit in the experiment contract.
        self.source_encoder=nn.Sequential(nn.Linear(source_dim,phi_width),nn.ReLU(),nn.Linear(phi_width,continuous_dim+num_types))
        self.phi=PointMLP(num_types,continuous_dim,phi_width,3,latent,dropout)
        self.psi=PointMLP(num_types,continuous_dim,64,2,128,0.)
        self.context=nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.key=EventProjection(num_types,continuous_dim,heads*dot_dim,bias=False)
        self.context_key=nn.Linear(128,heads*dot_dim,bias=False)
        self.query=nn.Parameter(torch.zeros(heads,dot_dim))
        layers=[];incoming=heads*latent
        for _ in range(3):layers += [nn.Linear(incoming,rho_width),nn.ReLU(),nn.Dropout(dropout)];incoming=rho_width
        layers.append(nn.Linear(rho_width,tasks));self.rho=nn.Sequential(*layers)
        for group in [self.source_encoder,self.context,self.context_key,self.rho]:
            for m in group.modules():
                if isinstance(m,nn.Linear):
                    nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                    if m.bias is not None:nn.init.zeros_(m.bias)
        for m in self.source_encoder:
            if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.rho[-1].weight)
        key_bound=math.sqrt(6/(continuous_dim+num_types+128))
        nn.init.uniform_(self.key.weight,-key_bound,key_bound)
        nn.init.uniform_(self.context_key.weight,-key_bound,key_bound)

    def forward(self,event_type,event_time,time_gap=None,source_features=None):
        if source_features is None:raise ValueError('Training-schema source features required')
        scaled=event_time[:,:,None]/self.timescales
        # Constant one means that the event occurred, not an imputed lab value.
        continuous=torch.cat([scaled.sin(),scaled.cos(),torch.ones_like(event_time[:,:,None])],-1)
        static=self.source_encoder(source_features)
        mask=torch.cat([torch.ones_like(event_type[:,:1],dtype=torch.bool),event_type.ne(0)],1)
        value=self.phi(continuous,event_type,static)
        psi=self.psi(continuous,event_type,static)
        context=self.context((psi*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True))
        keys=torch.cat([self.key.dense(static)[:,None],self.key(continuous,event_type)],1)
        # Retain the official linear set-context term. With a shared query it
        # is a softmax-common offset; do not claim new contextual interactions.
        keys=keys+self.context_key(context)[:,None]
        keys=keys.reshape(*keys.shape[:2],self.heads,self.dot_dim)
        logits=(keys*self.query[None,None]).sum(-1)/math.sqrt(self.dot_dim)
        if self.training and self.attn_dropout:
            logits=logits-(torch.rand_like(logits)<self.attn_dropout).to(logits.dtype)*1e9
        logits=logits.masked_fill(~mask[:,:,None],float('-inf'))
        weights=logits.softmax(1)
        pooled=torch.einsum('blh,bld->bhd',weights,value).flatten(1)
        return self.rho(pooled),value[:,1:]*event_type.ne(0)[:,:,None],pooled
