"""Three single-component C5 ablations, applied only to freshly initialized C5."""
import torch
from torch import nn
from models.tracehr_auxiliary_isolation import AuxiliaryIsolationNetwork
from models.tracehr_centered_native import CenteredNative
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('C5_noT', 'C5_noP', 'C5_noM')


class SummaryOnlyFusion(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Variable identity is shared by the summary and is not trajectory data.
        self.trajectory = nn.Module()
        self.trajectory.variable = original.trajectory.variable
        for name in ('summary', 'refine', 'gate', 'combine', 'output', 'auxiliary_head'):
            setattr(self, name, getattr(original, name))
        self.residual_scale = original.residual_scale
        first = self.combine[0]
        width = first.out_features
        first.weight = nn.Parameter(first.weight[:, :width].detach().clone())
        first.in_features = width

    def encode(self, batch):
        return None, None


class NoPeriodicEncoding(nn.Module):
    """Delete the learned periodic encoder; raw values and downstream layout stay fixed.

    Zero slots preserve the original input-width-dependent native parameterization
    and all retained tensors. This is a functional ablation, not a compressed model.
    """
    def __init__(self, original):
        super().__init__()
        for name, value in original.named_buffers(recurse=False):
            self.register_buffer(name, value)
        self.zero_coordinates = len(self.numeric_indices)*3

    def forward(self, raw):
        x = (raw-self.center)*self.factor
        x = torch.where(self.passthrough, raw, x/torch.sqrt(1+(x/3).square()))
        return torch.cat((x.new_zeros((len(x), self.zero_coordinates)), x), 1)


class IdentityInputNative(CenteredNative):
    """Retain shared native parameters and 8 members, remove learned input modulation."""
    def __init__(self, original):
        nn.Module.__init__(self)
        assert not original.centered and original.members == 8
        for name, child in original.named_children():
            self.add_module(name, child)
        for name, value in original.named_buffers(recurse=False):
            self.register_buffer(name, value)
        self.register_parameter('input_members', None)
        self.hp, self.info = original.hp, dict(original.info)
        self.members, self.centered = original.members, original.centered
        self.info['input_modulation'] = 'removed; identity map, 8 shared internal members retained'

    def forward(self, raw):
        x = self.numerical(raw)[None].expand(self.members, -1, -1)
        tensors = dict(x_cont=x, x_cat=torch.empty((self.members,len(raw),0),dtype=torch.long,device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0,1)


class C5AblationNetwork(AuxiliaryIsolationNetwork):
    def __init__(self, fresh_c1, variant, seed=42):
        assert variant in VARIANTS
        super().__init__(fresh_c1, seed)
        reference_hash = parameter_hash(self)
        original = {n:p.detach().clone() for n,p in self.named_parameters()}
        buffers = {n:p.detach().clone() for n,p in self.named_buffers()}
        count = sum(p.numel() for p in self.parameters())
        self.variant = variant
        if variant == 'C5_noT':
            self.fusion = SummaryOnlyFusion(self.fusion)
        elif variant == 'C5_noP':
            self.base.numerical = NoPeriodicEncoding(self.base.numerical)
        else:
            self.base = IdentityInputNative(self.base)
        mappings = {}
        for name, p in self.named_parameters():
            expected = original[name]
            if variant == 'C5_noT' and name == 'fusion.combine.0.weight':
                expected = expected[:, :expected.shape[0]]
                mappings[name] = dict(source=name, slice='[:, :64]')
            else:
                mappings[name] = dict(source=name, slice=None)
            assert torch.equal(p,expected), name
        for name,b in self.named_buffers():
            assert torch.equal(b,buffers[name]), name
        removed = sorted(set(original)-set(dict(self.named_parameters())))
        remaining_count = sum(p.numel() for p in self.parameters())
        assert count>remaining_count and all(p.requires_grad for p in self.parameters())
        self.info.update(structure=variant, reference_structure='C5',
            reference_C5_initial_weights_sha256=reference_hash,
            reference_C5_parameter_count=count, removed_from_C5=count-remaining_count,
            removed_parameter_names=removed, retained_parameter_mapping=mappings,
            retained_initialization_exact=True, retained_buffers_exact=True,
            trajectory_loaded_by_network=variant!='C5_noT',
            learned_numeric_periodic_encoding=variant!='C5_noP',
            learned_input_member_modulation=variant!='C5_noM',
            periodic_ablation_padding='zero, unchanged native width and initializer' if variant=='C5_noP' else None,
            original_summary_features_preserved=True, members_preserved=8,
            frozen_learnable_parameters=0, learned_checkpoint_loaded=False,
            freeze=False, stop_gradient=False, all_retained_parameters_jointly_trainable=True)

    def _before_head(self, module, args):
        if self.variant != 'C5_noT':
            return super()._before_head(module,args)
        if not self.fusion_enabled:
            return None
        summary, temporal, mask, dual = self._context
        assert temporal is None and mask is None
        f=self.fusion
        s=f.summary(summary)+f.trajectory.variable.weight[None]
        refined=s+.1*f.gate(s).sigmoid()*f.refine(s)
        delta=f.residual_scale*f.output(f.combine(refined).flatten(1))
        if dual:
            self._aux_logits=f.auxiliary_head(delta)[:,None,:]
        tensors=dict(args[0]);tensors['x_cont']=tensors['x_cont']+delta[None]
        return (tensors,)
