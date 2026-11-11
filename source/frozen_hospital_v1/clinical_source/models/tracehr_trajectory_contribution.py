"""C2 removes only the new trajectory path from the randomly initialized C1."""
import copy
import torch
from torch import nn


class SummaryResidual(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.variable = original.trajectory.variable
        self.summary = original.summary
        self.refine = original.refine
        self.gate = original.gate
        self.combine = copy.deepcopy(original.combine)
        first = self.combine[0]
        width = original.variable_width if hasattr(original, 'variable_width') else first.out_features
        first.weight = nn.Parameter(first.weight[:, :width].detach().clone())
        first.in_features = width
        self.output = original.output
        self.residual_scale = original.residual_scale

    def encode(self, batch):
        return None, None

    def fuse(self, hidden, summary, temporal, mask, variant):
        assert variant == 'C2' and temporal is None and mask is None
        s = self.summary(summary) + self.variable.weight[None]
        s = s + .1 * self.gate(s).sigmoid() * self.refine(s)
        delta = self.residual_scale * self.output(self.combine(s).flatten(1))
        return hidden + delta[None]


def remove_trajectory(random_c1):
    assert random_c1.variant == 'C1'
    old = {n: p.clone() for n, p in random_c1.state_dict().items()}
    old_count = sum(p.numel() for p in random_c1.parameters())
    random_c1.fusion = SummaryResidual(random_c1.fusion)
    random_c1.variant = 'C2'
    mappings = {}
    for name, value in random_c1.state_dict().items():
        origin = 'fusion.trajectory.variable.weight' if name == 'fusion.variable.weight' else name
        expected = old[origin]
        if name == 'fusion.combine.0.weight':
            expected = expected[:, :expected.shape[0]]
        assert torch.equal(value, expected), name
        mappings[name] = origin
    removed = old_count - sum(p.numel() for p in random_c1.parameters())
    assert removed > 0 and not hasattr(random_c1.fusion, 'trajectory')
    assert all(p.requires_grad for p in random_c1.parameters())
    random_c1.info.update(structure='C2', reference_structure='C1',
        change='remove added trajectory encoder, attention reading, and trajectory combine columns',
        retained_initialization_exact=True, removed_from_C1=removed,
        retained_parameter_mapping=mappings, original_summary_features_preserved=True,
        trajectory_loaded_by_network=False, freeze=False, stop_gradient=False,
        learned_checkpoint_loaded=False, all_retained_parameters_jointly_trainable=True)
    return random_c1
