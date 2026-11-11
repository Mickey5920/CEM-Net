"""Small readout corrections and a bounded auxiliary-gradient update."""
import torch
from torch import nn


def bounded_gradients(bce, auxiliary, rho):
    """Return detached gradients; the proxy scalar never sets its own weight."""
    if len(bce) != len(auxiliary):
        raise ValueError('Gradient parameter sets differ')
    def norm(values):
        return torch.stack([x.detach().float().square().sum() for x in values]).sum().sqrt()
    nb, na = norm(bce), norm(auxiliary)
    if not torch.isfinite(nb) or not torch.isfinite(na):
        raise ValueError('Nonfinite objective gradient')
    alpha = (float(rho)*nb/(na+1e-12)).clamp(max=1.).detach()
    mixed = [b.detach()+alpha*a.detach() for b,a in zip(bce,auxiliary)]
    cosine = sum((b.detach().float()*a.detach().float()).sum() for b,a in zip(bce,auxiliary))/(nb*na+1e-12)
    return mixed, dict(bce_norm=float(nb),ap_norm=float(na),alpha=float(alpha),
                      cosine=float(cosine),mixed_norm=float(norm(mixed)))


class TaskCorrection(nn.Module):
    def __init__(self, s0, d_model, tasks, specialized, dropout=.15, bound=.5):
        super().__init__()
        if tasks != 2:
            raise ValueError('This bounded comparison requires exactly two tasks')
        self.s0=s0.requires_grad_(False).eval()
        self.specialized=bool(specialized)
        self.bound=float(bound)
        def head(width, outputs):
            net=nn.Sequential(nn.Linear(2*d_model,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,outputs))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            return net
        self.heads=nn.ModuleList([head(24,1) for _ in range(tasks)]) if specialized else nn.ModuleList([head(48,tasks)])

    def train(self, mode=True):
        super().train(mode);self.s0.eval();return self

    def forward(self,event_type,event_time,time_gap=None,source_features=None):
        with torch.no_grad():
            za,_,ha=self.s0.anchor(event_type,event_time,time_gap=time_gap,source_features=source_features)
            zr,hidden,hr=self.s0.residual(event_type,event_time,time_gap=time_gap,source_features=source_features)
            logits=za+zr;representation=torch.cat([ha,hr],-1)
        correction=torch.cat([head(representation) for head in self.heads],-1)
        return logits+self.bound*correction.tanh(),hidden,hr
