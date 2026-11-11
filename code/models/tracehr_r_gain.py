"""Shared unmodulated summary conditions the native member features by gain."""
from models.tracehr_not_stability_functional import NoTrajectoryFunctional
from models.tracehr_unified_next import parameter_hash

VARIANT = 'C5_noT_noA_RG'


def apply_gain(hidden, delta):
    assert hidden.ndim == 3 and delta.shape == hidden.shape[1:]
    return hidden * (1 + delta[None])


class GainResidual(NoTrajectoryFunctional):
    def __init__(self, fresh_c1, seed):
        super().__init__(fresh_c1, 'C5_noT_noA', seed)
        initial = parameter_hash(self)
        count = sum(p.numel() for p in self.parameters())
        self.info.update(structure=VARIANT, reference_noA_initial_weights_sha256=initial,
            reference_noA_parameter_count=count, removed_from_noA=0, removed_from_noA_names=[],
            retained_noA_tensors_exact=True, retained_noA_buffers_exact=True,
            residual_input='original unmodulated 32x26 summary; M only in native backbone',
            residual_shared_across_members=True, residual_outputs_member_conditioned=False,
            main_connection='hidden * (1 + summary_delta)', independent_additive_summary=False,
            added_parameters=0, learned_checkpoint_loaded=False, stop_gradient=False)
        assert parameter_hash(self) == initial and not self.auxiliary_enabled

    def summary_delta(self, summary):
        assert summary.ndim == 3 and summary.shape[1:] == (32, 26)
        f = self.fusion
        s = f.summary(summary) + f.trajectory.variable.weight[None]
        refined = s + .1 * f.gate(s).sigmoid() * f.refine(s)
        return f.residual_scale * f.output(f.combine(refined).flatten(1))

    def _before_head(self, module, args):
        if not self.fusion_enabled:
            return None
        summary, temporal, mask, dual = self._context
        assert temporal is None and mask is None and not self.auxiliary_enabled
        delta = self.summary_delta(summary)
        tensors = dict(args[0])
        tensors['x_cont'] = apply_gain(tensors['x_cont'], delta)
        return (tensors,)
