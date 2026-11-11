"""Original models on training-hospital-only feature maps and normalization."""
import gc
from pathlib import Path
import numpy as np
import torch
from scripts.cem_icde72_common import RawData,expected,read,sha,atomic
from scripts.cem_evidence_common import parameter_hash,tensor_hashes

class HospitalData(RawData):
    def __init__(self,project,preparation=None):
        super().__init__(project)
        prep=Path(project)/expected()['input_preparation']
        self.record=read(prep/'completion.json');assert self.record['complete'] and self.record['fit_split']=='train'
        assert self.record['train_only_vocabulary'] and self.record['train_only_lab_selection'] and self.record['train_only_scaling']
        assert sha(prep/'preprocessing.npz')==self.record['preprocessing_sha256']
        self.prep=dict(np.load(prep/'preprocessing.npz'))
        for s in ('train','dev'):
            for name in (s+'_raw.npy',s+'_targets.npz'):assert sha(prep/name)==self.record['artifacts'][name]['sha256'],name
        assert not np.intersect1d(self.targets['train']['hospital_ids'],self.targets['dev']['hospital_ids']).size
    def batch(self,split,indices,device='cuda'):
        return {'raw':torch.from_numpy(np.array(self.raw[split][indices],copy=True)).to(device)},torch.from_numpy(self.targets[split]['y'][indices]).to(device)

def fresh_cem(data,preparation,seed,endpoint,out,variant):
    assert variant=='CEM_hospital'
    from scripts.tracehr_fte_binding_common import model_for,VARIANT
    from scripts.mimiciii_ema_runtime import digest
    model=model_for(data,preparation,VARIANT,endpoint,seed)
    initial=parameter_hash(model);hashes=tensor_hashes(model);buffers=digest(dict(model.named_buffers()))
    reference=model_for(data,preparation,VARIANT,endpoint,seed)
    assert initial==parameter_hash(reference) and hashes==tensor_hashes(reference)
    assert buffers==digest(dict(reference.named_buffers()))
    with torch.no_grad():
        model.eval();reference.eval();x,_=data.batch('train',np.arange(16),'cpu')
        torch.testing.assert_close(model(x),reference(x),atol=0,rtol=0)
    del reference;gc.collect();model.train()
    assert all(p.requires_grad for p in model.parameters()) and model.members==8
    assert not model.main_residual_enabled and not model.auxiliary_enabled and not list(model.fusion.parameters())
    observation=dict(complete=True,optimizer_updates=0,initial_weights_sha256=initial,retained_tensor_hashes=hashes,
        full_initial_tensors_exact=True,original_static_buffers_exact=True,
        reference='second fresh invocation of original CEM factory on this training-hospital-only preparation',
        comparison_to_old_random_split_initialization=False,reference_seed=seed,
        deletion=dict(retained_initials_exact=True,intervention='none; original full model on the new split'),
        input_preparation_sha256=sha(preparation/'completion.json'))
    atomic(out/'initialization_observation.json',observation)
    atomic(out/'connection_audit.json',dict(observation,learned_checkpoint_loaded=False,all_retained_parameters_trainable=True,
        P_retained=True,M_retained=True,evidence_branch_present=False,auxiliary_present=False,members=8,
        main_loss_coefficient=.5,test_loaded=False))
    return model
