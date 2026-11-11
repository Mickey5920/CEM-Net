"""X1: a learned zero-start coefficient on U1's conditional gate logit."""
import torch
from torch import nn
from torch.nn import functional as F
from models.cem_refine import RefineStructure
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('X1',)
DESCRIPTIONS = {'X1': 'U1 conditional gate with a jointly learned zero-initialized coefficient'}


def expected_new_names(variant):
    assert variant == 'X1'
    return {'gate.weight', 'gate.bias', 'gate.alpha'}


def expected_removed_names(variant, original_names):
    assert variant == 'X1'
    return set()


class StartGate(nn.Module):
    def __init__(self, fresh_u1_gate):
        super().__init__()
        assert fresh_u1_gate.weight.shape == (1, 192)
        # Reuse fresh random Parameters, never learned checkpoint weights.
        self.weight = fresh_u1_gate.weight
        self.bias = fresh_u1_gate.bias
        self.alpha = nn.Parameter(self.bias.new_zeros(()))

    def forward(self, inputs):
        return self.bias + self.alpha * F.linear(inputs, self.weight, None)


class StartGateStructure(RefineStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        assert variant == 'X1'
        super().__init__(fresh_cem, phi_dim, num_types, 'U1', seed)
        reference_hash = parameter_hash(self)
        original = {n:p.detach().clone() for n,p in self.fusion.named_parameters()}
        rng = torch.get_rng_state().clone()
        self.fusion.gate = StartGate(self.fusion.gate)
        assert torch.equal(rng, torch.get_rng_state())
        actual = dict(self.fusion.named_parameters())
        assert set(actual)-set(original) == {'gate.alpha'}
        assert set(original) <= set(actual)
        for n in original:
            assert torch.equal(original[n], actual[n]), n
        assert self.fusion.gate.alpha.requires_grad and self.fusion.gate.alpha.item() == 0.
        self.variant = variant
        self.info.update(structure=variant, description=DESCRIPTIONS[variant],
            reference_U1_initial_weights_sha256=reference_hash, retained_U1_tensors_exact=True,
            removed_U1_parameters=0, removed_U1_parameter_names=[],
            new_U1_parameters=1, new_U1_parameter_names=['fusion.gate.alpha'],
            removed_S3_parameters=0, removed_S3_parameter_names=[],
            new_S3_parameters=194, new_S3_parameter_names=sorted('fusion.'+n for n in expected_new_names(variant)),
            added_parameters=sum(p.numel() for p in self.fusion.parameters()),
            gate_condition='LN(event), LN(interaction)', gate_alpha_initial=0., gate_alpha_trainable=True,
            conditional_weight_initialization='unchanged fresh same-seed U1',
            both_evidence_paths_retained=True)
