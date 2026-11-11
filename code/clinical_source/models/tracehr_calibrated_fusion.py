"""Small independent endpoint gates; all expert outputs remain explicit."""
import numpy as np
import torch
from torch import nn


class ProbabilityGate(nn.Module):
    def __init__(self,dim,kind,rho):
        super().__init__();assert kind in ('G0','G1','G2')
        self.kind=kind
        initial=torch.logit(torch.tensor(np.clip(rho,1e-5,1-1e-5),dtype=torch.float32))
        if kind=='G0':self.bias=nn.Parameter(initial)
        else:
            self.heads=nn.ModuleList([nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1)) for _ in range(2)])
            for j,head in enumerate(self.heads):
                nn.init.zeros_(head[-1].weight);head[-1].bias.data.fill_(initial[j])

    def forward(self,x,base,temporal):
        if self.kind=='G0':g=torch.sigmoid(self.bias)[None].expand(len(x),-1)
        else:g=torch.sigmoid(torch.cat([head(x) for head in self.heads],1))
        return (1-g)*base+g*temporal,g
