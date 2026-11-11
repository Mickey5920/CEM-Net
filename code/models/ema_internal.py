"""Three predeclared internal interventions on a freshly initialized EMA network."""
import torch
from torch import nn
from models.tracehr_fte_binding import UntiedNative

VARIANTS = ('EMA_M_raw', 'EMA_M_periodic', 'EMA_no_numeric_direct')


class PartialInputNative(UntiedNative):
    def __init__(self, original, variant):
        nn.Module.__init__(self)
        assert variant in VARIANTS and original.members == 8 and not original.centered
        for n, m in original.named_children(): self.add_module(n, m)
        for n, b in original.named_buffers(recurse=False): self.register_buffer(n, b)
        self.hp, self.info = original.hp, dict(original.info)
        self.members, self.centered = original.members, original.centered
        q = 3 * len(original.numerical.numeric_indices)
        width = original.input_members.shape[1]
        keep = torch.ones(width, dtype=torch.bool, device=original.input_members.device)
        zero = torch.empty(0, dtype=torch.long, device=keep.device)
        if variant == 'EMA_M_raw': keep[:q] = False
        elif variant == 'EMA_M_periodic': keep[q:] = False
        else:
            zero = q + original.numerical.numeric_indices
            keep[zero] = False
        self.register_buffer('modulation_columns', torch.where(keep)[0])
        self.register_buffer('zero_direct_columns', zero)
        self.input_members = nn.Parameter(original.input_members[:, keep].detach().clone())
        self.variant, self.encoded_width = variant, width

    def forward(self, raw):
        z = self.numerical(raw)
        if self.zero_direct_columns.numel():
            z = z.index_fill(1, self.zero_direct_columns, 0.)
        factors = self.input_members.new_ones((self.members, self.encoded_width))
        factors = factors.index_copy(1, self.modulation_columns, self.input_members)
        x = factors[:, None] * z[None]
        tensors = dict(x_cont=x, x_cat=torch.empty((self.members, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)


def intervene(model, variant):
    assert variant in VARIANTS and not model.main_residual_enabled and not model.auxiliary_enabled
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.base = PartialInputNative(model.base, variant)
    cols = model.base.modulation_columns
    for n, p in model.named_parameters():
        ref = before[n][:, cols] if n == 'base.input_members' else before[n]
        assert torch.equal(p, ref), n
    for n, b in buffers.items(): assert torch.equal(b, dict(model.named_buffers())[n]), n
    assert all(p.requires_grad for p in model.parameters())
    return model, dict(variant=variant, retained_initials_exact=True, original_buffers_exact=True,
        input_factor_columns=cols.cpu().tolist(), zero_direct_columns=model.base.zero_direct_columns.cpu().tolist(),
        removed_parameters=sum(p.numel() for p in before.values())-sum(p.numel() for p in model.parameters()),
        encoded_width=model.base.encoded_width, periodic_coordinates=3*len(model.base.numerical.numeric_indices),
        all_retained_parameters_trainable=True, periodic_encoder_retained=True, members_preserved=8,
        raw_evidence_inputs_preserved=True, no_learned_checkpoint_loaded=True)


def cpu_checks():
    import copy
    from types import SimpleNamespace
    class Numeric(nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('numeric_indices',torch.tensor([0,2]))
            self.weight=nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3)/10+1)
        def forward(self,x):return torch.cat(((x[:,[0,2],None]*self.weight).flatten(1),x),1)
    class Net(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.arange(20,dtype=torch.float32).reshape(10,2)/20)
        def forward(self,d):return dict(x_cont=d['x_cont']@self.weight)
    original=nn.Module();original.numerical=Numeric();original.net=Net();original.static=nn.Identity()
    original.members=8;original.centered=False;original.hp=SimpleNamespace();original.info={}
    original.input_members=nn.Parameter(torch.arange(80,dtype=torch.float32).reshape(8,10)/100+1)
    x=torch.tensor([[1.,2.,3.,4.],[2.,1.,4.,3.]])
    for variant in VARIANTS:
        m=PartialInputNative(copy.deepcopy(original),variant);z=original.numerical(x);f=original.input_members.clone()
        if variant=='EMA_M_raw':f=torch.cat((torch.ones_like(f[:,:6]),f[:,6:]),1)
        elif variant=='EMA_M_periodic':f=torch.cat((f[:,:6],torch.ones_like(f[:,6:])),1)
        else:z=z.index_fill(1,torch.tensor([6,8]),0.)
        ref=((f[:,None]*z[None])@original.net.weight).transpose(0,1)
        torch.testing.assert_close(m(x),ref,rtol=0,atol=0)
        grads=torch.autograd.grad(m(x).square().mean(),tuple(m.parameters()))
        assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
        assert m.input_members.shape[1]==({'EMA_M_raw':4,'EMA_M_periodic':6,'EMA_no_numeric_direct':8}[variant])
    return ['exact_raw_only_factor_identity','exact_periodic_only_factor_identity',
        'numeric_direct_zero_slots_only','dead_input_factors_removed','all_retained_tensors_connected_no_updates']
