"""A fixed C logit anchor and a zero-output, independently learned residual."""
import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_tracehr_fusion import symmetric_nce, gaussian_mask_loss


class AnchoredResidual(nn.Module):
    def __init__(self, anchor, residual, aux_weight=0.):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.residual = residual
        self.aux_weight = float(aux_weight)
        self.aux_loss = None
        self.epoch = 1
        self.max_epochs = 50
        self.verify_zero = False
        nn.init.zeros_(residual.head.net[-1].weight)
        nn.init.zeros_(residual.head.net[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        if not mode:
            self.aux_loss = None
        return self

    def invariance_loss(self, s, t):
        # Same fixed additive latent map for weight=0 and weight>0. The anchor is detached.
        tokens = torch.stack([s.detach(), t], 1)
        a, b = F.dropout(tokens, .1, True), F.dropout(tokens, .1, True)
        za, zb = a.sum(1), b.sum(1)
        terms = [symmetric_nce(za, zb)]
        for m in range(2):
            terms.append(.5*(symmetric_nce(a[:, m], zb)+symmetric_nce(b[:, m], za)))
        va = [(a*(torch.rand_like(a)>.75)).sum(1) for _ in range(6)]
        vb = [(b*(torch.rand_like(b)>.75)).sum(1) for _ in range(6)]
        ratio = .25*min(self.epoch/self.max_epochs, 1.)
        return (sum(terms)+gaussian_mask_loss(za, va, ratio)+gaussian_mask_loss(zb, vb, ratio))/5.

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        with torch.no_grad():
            z, _, s = self.anchor(event_type, event_time, time_gap=time_gap, source_features=source_features)
        r, hidden, t = self.residual(event_type, event_time, time_gap=time_gap, source_features=source_features)
        logits = z+r
        if self.verify_zero:
            if torch.count_nonzero(r).item() or not torch.equal(logits, z):
                raise ValueError('Zero residual must exactly preserve the same-forward anchor logits')
        self.aux_loss = self.invariance_loss(s, t) if self.training and self.aux_weight > 0 else None
        return logits, hidden, t
