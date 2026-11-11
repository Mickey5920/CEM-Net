"""Controlled deletions within fresh C5_noT, with its execution path unchanged."""
import torch
from models.tracehr_c5_ablation import C5AblationNetwork, NoPeriodicEncoding, IdentityInputNative
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('C5_noT', 'C5_noT_noP', 'C5_noT_noM')


class NoTrajectoryConfirmation(C5AblationNetwork):
    def __init__(self, fresh_c1, variant, seed=42):
        assert variant in VARIANTS
        super().__init__(fresh_c1, 'C5_noT', seed)
        original = {n: p.detach().clone() for n, p in self.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in self.named_buffers()}
        reference_hash = parameter_hash(self)
        reference_count = sum(p.numel() for p in self.parameters())
        if variant == 'C5_noT_noP':
            self.base.numerical = NoPeriodicEncoding(self.base.numerical)
        elif variant == 'C5_noT_noM':
            self.base = IdentityInputNative(self.base)
        # self.variant remains C5_noT: the inherited pre-hook must use summary-only
        # execution for every deletion, never the trajectory path of full C5.
        self.confirmation_variant = variant
        for n, p in self.named_parameters():
            assert torch.equal(p, original[n]), n
        for n, b in self.named_buffers():
            assert torch.equal(b, buffers[n]), n
        removed = sorted(set(original) - set(dict(self.named_parameters())))
        self.info.update(structure=variant, seed=seed,
            reference_noT_initial_weights_sha256=reference_hash,
            reference_noT_parameter_count=reference_count,
            removed_from_noT=reference_count-sum(p.numel() for p in self.parameters()),
            removed_from_noT_names=removed, retained_noT_tensors_exact=True,
            retained_noT_buffers_exact=True, trajectory_loaded_by_network=False,
            learned_numeric_periodic_encoding=variant!='C5_noT_noP',
            learned_input_member_modulation=variant!='C5_noT_noM')
        assert self.variant=='C5_noT' and all(p.requires_grad for p in self.parameters())
