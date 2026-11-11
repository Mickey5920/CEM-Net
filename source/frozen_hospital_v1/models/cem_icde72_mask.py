"""Zero removed evidence after numerical encoding, with all parameter slots kept."""
import torch
from torch import nn
from models.tracehr_backbone_refine import TypedPeriodic

class MaskedPeriodic(TypedPeriodic):
    def __init__(self,original,removed):
        nn.Module.__init__(self)
        for n,b in original.named_buffers(recurse=False):self.register_buffer(n,b)
        for n,p in original.named_parameters(recurse=False):self.register_parameter(n,p)
        removed=torch.as_tensor(removed,dtype=torch.long)
        q=len(self.numeric_indices)
        erased=torch.where(torch.isin(self.numeric_indices,removed))[0]
        self.register_buffer('removed_raw_columns',removed)
        self.register_buffer('removed_encoded_columns',torch.cat(((erased[:,None]*3+torch.arange(3)).flatten(),3*q+removed)))
    def forward(self,raw):
        return super().forward(raw).index_fill(1,self.removed_encoded_columns,0.)

def mask_tabm(model,removed):
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    removed=torch.as_tensor(removed,dtype=torch.long)
    model.register_buffer('removed_raw_columns',removed)
    def input_mask(m,args):
        return (args[0].index_fill(-1,m.removed_raw_columns,0.),*args[1:])
    model.register_forward_pre_hook(input_mask)
    if model.num_module is not None:
        model.num_module.register_buffer('removed_evidence_columns',removed.clone())
        def encoded_mask(m,args,out):
            assert out.shape[-2]==model._n_num_features
            return out.index_fill(-2,m.removed_evidence_columns,0.)
        model.num_module.register_forward_hook(encoded_mask)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    return model

def invariance(model,raw,removed):
    model.eval();x=raw.detach().clone().requires_grad_(True)
    z=x.detach().clone();z[:,removed]=z[:,removed]*7.13+11.21
    a=model(x);b=model(z)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    grad=torch.autograd.grad(a.square().sum(),x)[0]
    assert not torch.count_nonzero(grad[:,removed])
    keep=torch.ones(x.shape[1],dtype=torch.bool);keep[removed]=False
    assert torch.count_nonzero(grad[:,keep])
    return dict(complete=True,removed_input_probability_invariance_exact=True,removed_input_gradient_exactly_zero=True,
                retained_input_gradient_nonzero=True,removed_columns=len(removed),optimizer_updates=0)
