"""C1: G2 with only the hidden-mean context projection removed."""
import torch
from models.tracehr_structure import StructureFusion, masked_attention
from models.tracehr_unified import independent_init


class FusionWithoutHiddenContext(StructureFusion):
    def __init__(self, hidden=256, width=64, bins=8):
        # Consume exactly the G2 initialization stream before removing the edge.
        super().__init__(hidden, width, bins)
        del self.context

    def fuse(self, hidden, summary, temporal, mask, variant):
        assert variant == 'C1'
        s = self.summary(summary)
        s = s + self.trajectory.variable.weight[None]
        t = masked_attention(self.query(s), self.key(temporal), self.value(temporal), mask)
        feedback = .1 * self.gate(s).sigmoid() * self.refine(s)
        s = s + feedback
        joined = self.combine(torch.cat((s, t), -1)).flatten(1)
        delta = self.residual_scale * self.output(joined)
        return hidden + delta[None]


def remove_hidden_context(randomly_initialized_g2, seed=42):
    """No learned checkpoint is accepted or loaded; match retained random tensors."""
    model = randomly_initialized_g2
    assert model.variant == 'G2'
    original = model.fusion.state_dict()
    with independent_init(seed, 'G/shared_structure'):
        replacement = FusionWithoutHiddenContext(256, 64, 8)
    kept = replacement.state_dict()
    removed = sorted(set(original)-set(kept))
    assert removed == ['context.bias', 'context.weight']
    assert all(torch.equal(value, original[name]) for name, value in kept.items())
    removed_parameters = sum(original[name].numel() for name in removed)
    assert removed_parameters == 16448
    model.fusion = replacement
    model.variant = 'C1'
    model.info = dict(model.info, structure='C1', reference_structure='G2',
                      change='remove hidden-mean context projection only',
                      retained_fusion_random_initialization_exact=True,
                      removed_parameter_names=['fusion.'+name for name in removed],
                      removed_parameter_count=removed_parameters,
                      learned_checkpoint_loaded=False, stop_gradient=False,
                      freeze=False, all_retained_parameters_jointly_trainable=True)
    assert all(p.requires_grad for p in model.parameters())
    return model
