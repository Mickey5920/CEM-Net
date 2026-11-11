"""Single deletions of the fixed, untied, branch-free MIMIC-III model."""
import torch
from torch import nn
from models.tracehr_fte_binding import UntiedNative
from models.tracehr_c5_ablation import NoPeriodicEncoding, IdentityInputNative


class WithoutPeriodic(UntiedNative):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, module in original.named_children():
            self.add_module(name, module)
        for name, value in original.named_buffers(recurse=False):
            self.register_buffer(name, value)
        self.hp, self.info = original.hp, dict(original.info)
        self.members, self.centered = original.members, original.centered
        self.numerical = NoPeriodicEncoding(original.numerical)
        self.zero_coordinates = self.numerical.zero_coordinates
        # The discarded factors multiply fixed zeros and have no loss path.
        # Retain the original raw-coordinate factors and native input width.
        self.input_members = nn.Parameter(original.input_members[:, self.zero_coordinates:].detach().clone())

    def forward(self, raw):
        z = self.numerical(raw)
        x = torch.cat((z[None, :, :self.zero_coordinates].expand(self.members, -1, -1),
                       self.input_members[:, None] * z[None, :, self.zero_coordinates:]), dim=-1)
        tensors = dict(x_cont=x, x_cat=torch.empty((self.members, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)


def delete_component(model, component):
    assert component in ('P', 'M')
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    assert not list(model.fusion.parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.base = WithoutPeriodic(model.base) if component == 'P' else IdentityInputNative(model.base)
    mapping = {}
    for n, p in model.named_parameters():
        ref = before[n]
        start = model.base.zero_coordinates if component == 'P' and n == 'base.input_members' else None
        if start is not None:
            ref = ref[:, start:]
        assert torch.equal(p, ref), n
        mapping[n] = dict(source=n, start_column=start)
    for n, b in model.named_buffers():
        assert torch.equal(b, buffers[n]), n
    removed = sorted(set(before) - set(dict(model.named_parameters())))
    assert removed and all(p.requires_grad for p in model.parameters())
    assert all(n.startswith('base.numerical.') if component == 'P' else n == 'base.input_members' for n in removed)
    model.variant = 'noEvidence_no' + component
    return model, dict(component=component, retained_initials_exact=True, retained_buffers_exact=True,
        mapping=mapping, removed_names=removed, removed_parameters=sum(v.numel() for v in before.values())-sum(p.numel() for p in model.parameters()),
        zero_slots_preserve_native_width=component == 'P', members_preserved=8, evidence_branch_present=False,
        auxiliary_present=False, learned_checkpoint_loaded=False, main_loss_coefficient=.5)
