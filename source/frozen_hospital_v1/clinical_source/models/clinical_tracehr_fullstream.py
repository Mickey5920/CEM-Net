"""Cached-feature C and capacity-matched zero-initialized residuals."""
import torch
from torch import nn
from models.clinical_static import ResidualMLPBlock


class CachedC(nn.Module):
    def __init__(self, input_dim, width=192, tasks=2):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width),nn.GELU(),nn.Dropout(.15),
            ResidualMLPBlock(width,.15),nn.Linear(width,96),nn.LayerNorm(96))
        self.head=nn.Sequential(nn.Dropout(.15),nn.Linear(96,tasks))

    def forward(self,x,extra=None):return self.head(self.net(x))


def parameter_count(dim,width,tasks=2):
    # LayerNorm input; input Linear; one residual block; projection LN; head.
    return 2*dim + dim*width+width + 4*width*width+5*width + width*96+96 + 192 + 96*tasks+tasks


class CachedResidual(nn.Module):
    def __init__(self,anchor,phi_dim,extra_dim,base_width,extra_mean=None,extra_scale=None):
        super().__init__();self.anchor=anchor.requires_grad_(False).eval()
        target=parameter_count(phi_dim,base_width)
        self.width=min(range(2,193),key=lambda w:abs(parameter_count(phi_dim+extra_dim,w)-target))
        self.residual=CachedC(phi_dim+extra_dim,self.width)
        self.match_error=abs(parameter_count(phi_dim+extra_dim,self.width)/target-1)
        if self.match_error>=.03:raise ValueError('Residual parameter mismatch exceeds 3%')
        self.register_buffer('extra_mean',torch.zeros(extra_dim) if extra_mean is None else torch.as_tensor(extra_mean,dtype=torch.float32))
        self.register_buffer('extra_scale',torch.ones(extra_dim) if extra_scale is None else torch.as_tensor(extra_scale,dtype=torch.float32))
        nn.init.zeros_(self.residual.head[-1].weight);nn.init.zeros_(self.residual.head[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.anchor.eval();return self

    def forward(self,x,extra=None):
        with torch.no_grad():z=self.anchor(x)
        if self.extra_mean.numel():x=torch.cat([x,(extra-self.extra_mean)/self.extra_scale],1)
        return z+self.residual(x)
