"""Existing M modulates the corresponding summary coordinates of shared R."""
import torch
from models.tracehr_not_stability_functional import NoTrajectoryFunctional
from models.tracehr_unified_next import parameter_hash

VARIANT='C5_noT_noA_RM'

def member_summary(summary, members, phi_dim):
    assert summary.ndim==3 and summary.shape[1:]==(32,26)
    assert members.shape[0]==8 and members.shape[1]==phi_dim+832
    simple=members[:,phi_dim:phi_dim+352].reshape(8,32,11)
    state=members[:,phi_dim+352:].reshape(8,32,15)
    modulation=torch.cat((simple,state),-1)
    return summary[None]*modulation[:,None]

class MemberResidual(NoTrajectoryFunctional):
    def __init__(self,fresh_c1,seed):
        super().__init__(fresh_c1,'C5_noT_noA',seed)
        initial=parameter_hash(self);count=sum(p.numel() for p in self.parameters())
        self.info.update(structure=VARIANT,reference_noA_initial_weights_sha256=initial,
            reference_noA_parameter_count=count,removed_from_noA=0,removed_from_noA_names=[],
            retained_noA_tensors_exact=True,retained_noA_buffers_exact=True,
            residual_input='existing input_members on matching 32x11 and 32x15 raw summary coordinates',
            residual_shared_across_members=True,residual_outputs_member_conditioned=True,
            added_parameters=0,learned_checkpoint_loaded=False,stop_gradient=False)
        assert parameter_hash(self)==initial and not self.auxiliary_enabled

    def _before_head(self,module,args):
        if not self.fusion_enabled:return None
        summary,temporal,mask,dual=self._context
        assert temporal is None and mask is None and not self.auxiliary_enabled
        batch=len(summary)
        conditioned=member_summary(summary,self.base.input_members,self.phi_dim).reshape(8*batch,32,26)
        f=self.fusion
        s=f.summary(conditioned)+f.trajectory.variable.weight[None]
        refined=s+.1*f.gate(s).sigmoid()*f.refine(s)
        delta=f.residual_scale*f.output(f.combine(refined).flatten(1)).reshape(8,batch,256)
        tensors=dict(args[0]);assert tensors['x_cont'].shape==delta.shape
        tensors['x_cont']=tensors['x_cont']+delta
        return (tensors,)
