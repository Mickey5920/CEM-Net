"""Single deletions inside freshly initialized noA; no trained weights are imported."""
import torch
from torch import nn
from models.tracehr_not_stability_functional import NoTrajectoryFunctional
from models.tracehr_c5_ablation import NoPeriodicEncoding, IdentityInputNative
from models.tracehr_unified_next import parameter_hash

VARIANTS = tuple('C5_noT_noA_no'+c for c in 'PMR')

class NoAAblation(NoTrajectoryFunctional):
    def __init__(self, fresh_c1, variant, seed):
        assert variant in VARIANTS and seed in (43,44,45)
        super().__init__(fresh_c1, 'C5_noT_noA', seed)
        original = {n:p.detach().clone() for n,p in self.named_parameters()}
        buffers = {n:b.detach().clone() for n,b in self.named_buffers()}
        reference_hash = parameter_hash(self)
        count = sum(p.numel() for p in self.parameters())
        self.variant = self.functional_variant = variant
        if variant.endswith('noP'):
            self.base.numerical = NoPeriodicEncoding(self.base.numerical)
        elif variant.endswith('noM'):
            self.base = IdentityInputNative(self.base)
        else:
            self._hook.remove()
            self.fusion = nn.Module()
            self.main_residual_enabled = False
        for n,p in self.named_parameters(): assert torch.equal(p, original[n]), n
        for n,b in self.named_buffers(): assert torch.equal(b, buffers[n]), n
        removed = sorted(set(original)-set(dict(self.named_parameters())))
        assert removed and all(p.requires_grad for p in self.parameters())
        prefixes = {'P':'base.numerical.', 'M':'base.input_members', 'R':'fusion.'}
        assert all(n.startswith(prefixes[variant[-1]]) for n in removed)
        self.info.update(structure=variant, reference_noA_initial_weights_sha256=reference_hash,
            reference_noA_parameter_count=count, removed_from_noA=count-sum(p.numel() for p in self.parameters()),
            removed_from_noA_names=removed, retained_noA_tensors_exact=True, retained_noA_buffers_exact=True,
            main_residual_enabled=self.main_residual_enabled, auxiliary_enabled=False,
            auxiliary_loss_coefficient=0., main_loss_coefficient=.5, members_preserved=8,
            noR_disconnected_branch_pruned=variant.endswith('noR'), learned_checkpoint_loaded=False,
            all_retained_parameters_jointly_trainable=True, freeze=False, stop_gradient=False)

    def _forward(self, batch, dual):
        if self.main_residual_enabled:
            return super()._forward(batch, dual)
        logits = self.base(batch['raw'])
        return (logits, None) if dual else logits
