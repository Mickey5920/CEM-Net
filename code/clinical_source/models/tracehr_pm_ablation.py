"""Delete P or M inside fresh PM; reuse the verified noR construction exactly."""
import torch
from models.tracehr_noa_validation import NoAAblation
from models.tracehr_c5_ablation import NoPeriodicEncoding, IdentityInputNative
from models.tracehr_unified_next import parameter_hash

PM = 'C5_noT_noA_noR'
VARIANTS = (PM + '_noP', PM + '_noM')


def remove_pm_component(model, variant):
    assert variant in VARIANTS
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    assert model.members == 8 and not list(model.fusion.parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    reference_hash = parameter_hash(model)
    reference_count = sum(p.numel() for p in model.parameters())
    if variant.endswith('noP'):
        model.base.numerical = NoPeriodicEncoding(model.base.numerical)
    else:
        model.base = IdentityInputNative(model.base)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
    names = sorted(set(before) - set(dict(model.named_parameters())))
    prefix = 'base.numerical.' if variant.endswith('noP') else 'base.input_members'
    assert names and all(n.startswith(prefix) for n in names)
    assert not any(n.startswith('fusion.') for n, _ in model.named_parameters())
    assert all(p.requires_grad for p in model.parameters())
    count = sum(p.numel() for p in model.parameters())
    model.info.update(reference_PM_initial_weights_sha256=reference_hash,
        reference_PM_parameter_count=reference_count, removed_from_PM=reference_count-count,
        removed_from_PM_names=names, retained_PM_tensors_exact=True, retained_PM_buffers_exact=True,
        removed_from_noA=model.info['reference_noA_parameter_count']-count,
        removed_from_noA_names=sorted(model.info['removed_from_noA_names']+names),
        structure=variant, noR_disconnected_branch_pruned=True,
        learned_numeric_periodic_encoding=not variant.endswith('noP'),
        learned_input_member_modulation=not variant.endswith('noM'))
    model.variant = model.functional_variant = variant


class PMAblation(NoAAblation):
    def __init__(self, fresh_c1, variant, seed):
        assert variant in VARIANTS and seed in (48, 49, 50)
        super().__init__(fresh_c1, PM, seed)
        remove_pm_component(self, variant)
