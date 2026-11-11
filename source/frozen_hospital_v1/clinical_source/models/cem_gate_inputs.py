"""Select one of U1's two gate inputs without changing either evidence path."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from models.cem_gated_event import GatedEventChannels
from models.cem_refine import RefineStructure
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('W1', 'W2')
DESCRIPTIONS = {
    'W1': 'event-only condition on the event direct residual gate',
    'W2': 'interaction-only condition on the event direct residual gate',
}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def expected_new_names(variant):
    assert variant in VARIANTS
    return {'gate.weight', 'gate.bias'}


def expected_removed_names(variant, original_names):
    assert variant in VARIANTS
    return set()


class InputGate(nn.Module):
    def __init__(self, fresh_u1_gate, variant):
        super().__init__()
        assert variant in VARIANTS and fresh_u1_gate.weight.shape == (1, 192)
        self.start = 0 if variant == 'W1' else 96
        # Select initial parameters, never weights from a fitted checkpoint.
        self.weight = nn.Parameter(fresh_u1_gate.weight[:, self.start:self.start+96].detach().clone())
        self.bias = nn.Parameter(fresh_u1_gate.bias.detach().clone())

    def forward(self, inputs):
        assert inputs.shape[-1] == 192
        return F.linear(inputs[..., self.start:self.start+96], self.weight, self.bias)


class InputGateChannels(GatedEventChannels):
    def __init__(self, num_types, global_dim, variant, seed):
        super().__init__(num_types, global_dim, seed)
        self.gate = InputGate(self.gate, variant)
        self.gate_variant = variant
        # The inherited forward, normalization, evidence paths, and dropout order
        # stay unchanged; InputGate selects the specified half of its input.


class InputGateStructure(RefineStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        assert variant in VARIANTS
        super().__init__(fresh_cem, phi_dim, num_types, 'U1', seed)
        reference_hash = parameter_hash(self)
        original = {n:p.detach().clone() for n,p in self.fusion.named_parameters()}
        rng = torch.get_rng_state().clone()
        self.fusion = InputGateChannels(num_types, phi_dim-4*num_types, variant, seed)
        assert torch.equal(rng, torch.get_rng_state())
        actual = dict(self.fusion.named_parameters())
        assert set(actual) == set(original)
        start = 0 if variant == 'W1' else 96
        for name in actual:
            target = original[name][:, start:start+96] if name == 'gate.weight' else original[name]
            assert torch.equal(actual[name], target), name
        mapping = dict(parameter='fusion.gate.weight', source_shape=[1,192], target_shape=[1,96],
            source_columns=[start,start+96], source_tensor_sha256=tensor_sha(original['gate.weight']),
            source_slice_sha256=tensor_sha(original['gate.weight'][:,start:start+96]),
            target_tensor_sha256=tensor_sha(actual['gate.weight']), exact=True)
        self.variant = variant
        self.info.update(structure=variant, description=DESCRIPTIONS[variant],
            reference_U1_initial_weights_sha256=reference_hash,
            retained_U1_unsliced_tensors_exact=True, retained_U1_slice_exact=True,
            gate_initialization_mapping=mapping,
            removed_U1_parameters=96, removed_U1_parameter_names=[],
            new_U1_parameters=0, new_U1_parameter_names=[],
            removed_S3_parameters=0, removed_S3_parameter_names=[],
            new_S3_parameters=97, new_S3_parameter_names=['fusion.gate.bias','fusion.gate.weight'],
            added_parameters=sum(p.numel() for p in self.fusion.parameters()),
            gate_condition='LN(event)' if variant == 'W1' else 'LN(interaction)',
            both_evidence_paths_retained=True)
