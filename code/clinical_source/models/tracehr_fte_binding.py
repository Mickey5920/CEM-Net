"""Untie only the fresh PM input factors; no checkpoint or learned state is used."""
import torch
from torch import nn
from models.tracehr_unified_next import parameter_hash
from models.tracehr_backbone_refine import TypedNative

PM = 'C5_noT_noA_noR'
VARIANT = PM + '_untied'
VARIANTS = (VARIANT,)


class UntiedNative(TypedNative):
    def __init__(self, original):
        nn.Module.__init__(self)
        assert original.members == 8 and not original.centered
        for name, module in original.named_children(): self.add_module(name, module)
        for name, value in original.named_buffers(recurse=False): self.register_buffer(name, value)
        self.hp, self.info = original.hp, dict(original.info)
        self.members, self.centered = original.members, original.centered
        # Copies of fresh random values become separate trainable coordinates.
        # detach is used for construction only; forward has no detached path.
        self.input_members = nn.Parameter(original.input_members.index_select(1, original.feature_map).detach().clone())

    def forward(self, raw):
        z = self.numerical(raw)[None]
        x = self.input_members[:, None] * z
        tensors = dict(x_cont=x, x_cat=torch.empty((self.members, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)


def untie_fresh_pm(model):
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    assert not list(model.fusion.parameters())
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    reference = parameter_hash(model)
    count = sum(p.numel() for p in model.parameters())
    old = model.base
    model.base = UntiedNative(old)
    for n, p in model.named_parameters():
        if n != 'base.input_members': assert torch.equal(p, initial[n]), n
    for n, b in model.named_buffers(): assert torch.equal(b, buffers[n]), n
    assert torch.equal(model.base.input_members, initial['base.input_members'].index_select(1, old.feature_map))
    assert set(initial) == set(dict(model.named_parameters()))
    model.variant = model.functional_variant = VARIANT
    model.info.update(structure=VARIANT, reference_PM_initial_weights_sha256=reference,
        reference_PM_parameter_count=count, retained_PM_tensors_except_expanded_factors_exact=True,
        retained_PM_buffers_exact=True, expanded_factors_initial_values_exact=True,
        factor_parameters_tied=initial['base.input_members'].numel(),
        factor_parameters_untied=model.base.input_members.numel(),
        additional_parameters=model.base.input_members.numel()-initial['base.input_members'].numel(),
        R_absent=True, learned_checkpoint_loaded=False, all_parameters_trainable=True,
        coordinate_factors_independent=True)
    return model


@torch.no_grad()
def coordinate_divergence(model):
    base = model.base
    q3 = 3 * len(base.numerical.numeric_indices)
    reference = base.input_members[:, q3:].index_select(1, base.feature_map[:q3])
    difference = base.input_members[:, :q3] - reference
    return dict(max_abs=float(difference.abs().max()), mean_abs=float(difference.abs().mean()),
                unequal_coordinates=int(torch.count_nonzero(difference)))
