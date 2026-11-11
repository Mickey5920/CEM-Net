"""Three predeclared controls of the fixed EMA clinical evidence network."""
import math
import torch
from torch import nn
from models.tracehr_fte_binding import UntiedNative
from models.tracehr_c5_ablation import IdentityInputNative
from models.mimiciii_pm_mini import WithoutPeriodic

VARIANTS = ('EMA_no_duration', 'EMA_NRE_MLP', 'EMA_no_NRE_CMM')


def sliced_parameter(p, rows):
    value = p.detach().index_select(0, rows).clone()
    if hasattr(p, 'context'):
        with p.set_context():
            return type(p)(value, trainable=p.trainable, requires_grad=p.requires_grad, hyper_factors=p.hyper_factors)
    return nn.Parameter(value, requires_grad=p.requires_grad)


class ResponseControl(nn.Module):
    def __init__(self, original, variant):
        super().__init__(); self.variant = variant
        for n,b in original.named_buffers(recurse=False): self.register_buffer(n,b)
        q = len(original.numeric_indices); d = len(original.center)
        # The frozen preparation concatenates the event profile, 352 ordinary
        # laboratory summaries, and 32 x 15 duration summaries, in this order.
        keep = torch.arange(q)
        zero = torch.empty(0, dtype=torch.long)
        if variant == 'EMA_no_duration':
            keep = torch.where(original.numeric_indices < d-480)[0]
            erased = torch.where(original.numeric_indices >= d-480)[0]
            zero = torch.cat(((erased[:,None]*3+torch.arange(3)).flatten(), 3*q+torch.arange(d-480,d)))
        self.register_buffer('retained_response_rows', keep)
        self.register_buffer('zero_coordinates', zero)
        for n,p in original.named_parameters(recurse=False):
            self.register_parameter(n, sliced_parameter(p,keep) if variant=='EMA_no_duration' else p)

    def forward(self, raw):
        x=(raw-self.center)*self.factor
        x=torch.where(self.passthrough,raw,x/torch.sqrt(1+(x/3).square()))
        indices=self.numeric_indices.index_select(0,self.retained_response_rows)
        num=x.index_select(1,indices)
        affine=2*math.pi*num.unsqueeze(-1)*self.frequency+self.phase
        h=torch.relu(affine) if self.variant=='EMA_NRE_MLP' else torch.cos(affine)
        z=torch.bmm(h.transpose(0,1),self.weight).transpose(0,1)+self.bias
        if self.variant=='EMA_no_duration':
            full=z.new_zeros((len(raw),len(self.numeric_indices),3))
            z=full.index_copy(1,self.retained_response_rows,z)
        joined=torch.cat((z.flatten(1),x),1)
        return joined.index_fill(1,self.zero_coordinates,0.) if self.zero_coordinates.numel() else joined


class DurationNative(UntiedNative):
    def __init__(self, original):
        nn.Module.__init__(self)
        for n,m in original.named_children(): self.add_module(n,m)
        for n,b in original.named_buffers(recurse=False): self.register_buffer(n,b)
        self.hp,self.info=original.hp,dict(original.info)
        self.members,self.centered=original.members,original.centered
        self.numerical=ResponseControl(original.numerical,'EMA_no_duration')
        keep=torch.ones(original.input_members.shape[1],dtype=torch.bool)
        keep[self.numerical.zero_coordinates]=False
        self.register_buffer('modulation_columns',torch.where(keep)[0])
        self.encoded_width=keep.numel()
        self.input_members=nn.Parameter(original.input_members[:,keep].detach().clone())

    def forward(self, raw):
        z=self.numerical(raw)
        factors=self.input_members.new_ones((self.members,self.encoded_width))
        factors=factors.index_copy(1,self.modulation_columns,self.input_members)
        tensors=dict(x_cont=factors[:,None]*z[None],x_cat=torch.empty((self.members,len(raw),0),dtype=torch.long,device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0,1)


def intervene(model, variant, phi_dim):
    assert variant in VARIANTS and model.members==8
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    assert len(model.base.numerical.center)==phi_dim+832
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    if variant=='EMA_no_duration': model.base=DurationNative(model.base)
    elif variant=='EMA_NRE_MLP': model.base.numerical=ResponseControl(model.base.numerical,variant)
    else: model.base=IdentityInputNative(WithoutPeriodic(model.base))
    mappings={}
    for n,p in model.named_parameters():
        ref=before[n]
        if variant=='EMA_no_duration':
            if n=='base.input_members':ref=ref[:,model.base.modulation_columns]
            elif n.startswith('base.numerical.'):
                ref=ref.index_select(0,model.base.numerical.retained_response_rows)
        assert torch.equal(p,ref),n
        mappings[n]=dict(shape=list(p.shape),retained_initial_exact=True)
    for n,b in buffers.items():assert torch.equal(b,dict(model.named_buffers())[n]),n
    assert all(p.requires_grad for p in model.parameters())
    return model,dict(variant=variant,retained_initials_exact=True,original_buffers_exact=True,
        removed_parameters=sum(p.numel() for p in before.values())-sum(p.numel() for p in model.parameters()),
        mappings=mappings,trainable_parameters_all=True,members=8,EMA_retained=True,
        duration_dimensions_removed=480 if variant=='EMA_no_duration' else 0,
        MLP_parameters_per_numeric_feature=83 if variant=='EMA_NRE_MLP' else None)


def cpu_checks():
    from models.tracehr_backbone_refine import TypedPeriodic
    import numpy as np
    prep=dict(center=np.zeros(835,np.float32),factor=np.ones(835,np.float32),
              binary=np.zeros(835,bool),numeric_indices=np.arange(835))
    original=TypedPeriodic(prep,48);x=torch.randn(5,835)
    masked=ResponseControl(original,'EMA_no_duration');response=masked(x)
    assert torch.count_nonzero(response[:,masked.zero_coordinates])==0
    changed=x.clone();changed[:,-480:]+=500
    torch.testing.assert_close(masked(changed),response,rtol=0,atol=0)
    cols=torch.ones(response.shape[1],dtype=torch.bool);cols[masked.zero_coordinates]=False
    torch.testing.assert_close(response[:,cols],original(x)[:,cols],rtol=0,atol=0)
    mlp=ResponseControl(original,'EMA_NRE_MLP')
    assert sum(p.numel() for p in mlp.parameters())==835*83
    transformed=x/torch.sqrt(1+(x/3).square())
    ref=torch.relu(2*math.pi*transformed[...,None]*original.frequency+original.phase)
    ref=torch.einsum('bqk,qko->bqo',ref,original.weight)+original.bias
    torch.testing.assert_close(mlp(x)[:,:835*3],ref.flatten(1))
    assert not torch.equal(mlp(x),original(x))
    return ['duration_raw_and_encoded_slots_zero','duration_values_cannot_affect_forward',
            'other_coordinates_exact','per_feature_MLP_1_16_3_83_parameters','MLP_changes_forward']
