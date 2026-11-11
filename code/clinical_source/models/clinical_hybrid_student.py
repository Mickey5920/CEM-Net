"""Complete-feature corrections; borrowed low-rank DCN-V2 cross equation."""
import torch
from torch import nn
from torch.nn import functional as F


class CrossLayer(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        if not 0 < rank < dimension/2: raise ValueError('Invalid low-rank projection')
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension)
        nn.init.xavier_uniform_(self.down.weight); nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x0, x):
        return x + x0*self.up(self.down(x))


class FeatureCorrection(nn.Module):
    def __init__(self, dimension, architecture, rank=32, depth=2, dropout=.15):
        super().__init__()
        self.normalization = nn.LayerNorm(dimension)
        self.architecture = architecture
        if architecture == 'cross':
            self.body = nn.ModuleList([CrossLayer(dimension,rank) for _ in range(depth)])
            self.output = nn.Linear(dimension,2)
        elif architecture == 'mlp':
            target = depth*(2*dimension*rank+dimension)+2*dimension+2
            # Same common LayerNorm, then the closest integer MLP width.
            width = min(range(1,4*depth*rank+32), key=lambda w:abs((dimension*w+3*w+2)-target))
            self.body = nn.Sequential(nn.Linear(dimension,width), nn.GELU(), nn.Dropout(dropout))
            self.output = nn.Linear(width,2)
        else: raise ValueError('Unknown correction architecture')
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, features):
        x0 = self.normalization(features)
        if self.architecture == 'cross':
            x = x0
            for layer in self.body: x = layer(x0,x)
        else: x = self.body(x0)
        return self.output(x)


def distillation_loss(logits, labels, teacher, positive_weight, weight):
    hard = F.binary_cross_entropy_with_logits(logits,labels,pos_weight=positive_weight)
    if not 0 <= weight < 1: raise ValueError('Invalid distillation weight')
    if weight == 0: return hard, hard, logits.new_tensor(0.)
    q = teacher.detach().clamp(1e-6,1-1e-6)
    soft = (q*(q.log()-F.logsigmoid(logits)) + (1-q)*((1-q).log()-F.logsigmoid(-logits))).mean()
    return (1-weight)*hard+weight*soft, hard, soft


class ClinicalFeatureStudent(nn.Module):
    """Deployment graph: frozen S0 plus correction, with no tree invocation."""
    def __init__(self, s0, correction):
        super().__init__()
        self.s0 = s0.requires_grad_(False).eval()
        self.correction = correction

    def train(self, mode=True):
        super().train(mode); self.s0.eval(); return self

    def forward(self, types, times, time_gap=None, source_features=None):
        with torch.no_grad():
            logits, hidden, pooled = self.s0(types,times,time_gap=time_gap,source_features=source_features)
            encoder = self.s0.anchor.encoder.active
            if hasattr(encoder,'unscaled_features'):
                features = encoder.unscaled_features(types,times,source_features)
            else: features = encoder.build_static_features(types,times,source_features)
        return logits+self.correction(features),hidden,pooled
