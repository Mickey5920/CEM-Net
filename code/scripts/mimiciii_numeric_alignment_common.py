"""One fixed preprocessing contrast; retain the complete branch-free P/M network."""
from pathlib import Path
import numpy as np
import torch
from scripts.mimiciii_baselines_common import numeric_protocol
from scripts.mimiciii_fixed_common import Data, model_for, parameter_hash
from scripts.train_mimiciii_numeric_baseline import NoEvidence
from scripts.tracehr_multiview_common import read, sha, atomic

ROOT=Path(__file__).resolve().parents[1]
RUN='mimiciii_numeric_alignment_20261024'
VARIANT='noEvidence_Q'
SEEDS=(48,49,50)


def expected():return read(ROOT/'work'/RUN/'expected.json')
def matrix():return [dict(variant=VARIANT,seed=s,endpoint=e) for s in SEEDS for e in (0,1)]
def protocol(variant,seed):
    assert variant==VARIANT and seed in SEEDS
    return dict(numeric_protocol('noEvidence',seed),variant=variant,
        numeric_preprocessing='archived train-fit TabM quantile transform replaces native summary scaling; no second scaling',
        initialization_policy='same-seed original random parameters and static buffers; change only summary preprocessing buffers')


class AlignedData(Data):
    def __init__(self,preparation,project):
        import joblib
        super().__init__(preparation)
        self.phi=int(self.record['phi_dim'])
        assert self.phi==12840 and self.input_dim-self.phi==832
        path=Path(project)/expected()['transform_path']
        assert sha(path)==expected()['transform_sha256']
        transform=joblib.load(path)
        assert transform['phi_dim']==self.phi and transform['fit_split']=='train'
        self.quantile=transform['numeric_quantile']
        assert self.quantile.n_features_in_==832
        self.summary={s:self.transform_summary(raw[:,self.phi:]) for s,raw in self.raw.items()}
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in self.summary.values())

    def transform_summary(self,summary):
        # This is the frozen transformer used by TabM/TabM-mini; never fit here.
        values=np.empty(summary.shape,dtype=np.float32)
        for start in range(0,len(values),1024):
            values[start:start+1024]=self.quantile.transform(np.array(summary[start:start+1024],copy=True))
        assert np.isfinite(values).all()
        return values

    def batch(self,split,indices,device='cuda'):
        batch,y=super().batch(split,indices,device)
        batch['raw'][:,self.phi:]=torch.as_tensor(self.summary[split][indices],device=device)
        return batch,y


def tensor_hashes(model):
    import hashlib
    return {n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in model.named_parameters()}


def fresh_model(data,preparation,seed,endpoint,out):
    # Original raw matrices are retained for the archived data-dependent RANDOM
    # initializer. Only batch inputs and three fixed scaling buffers differ.
    full=model_for(data,preparation,endpoint,seed);model=NoEvidence(full);del full
    ref=expected()['references'][f'noEvidence_s{seed}_t{endpoint}']
    path=Path(ref['output'])/'initialization_observation.json'
    assert sha(path)==ref['initialization_sha256']
    observation=read(path);before=tensor_hashes(model)
    assert before==observation['retained_tensor_hashes']
    assert parameter_hash(model)==ref['initial_weights_sha256']
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    num=model.base.numerical;phi=data.phi
    with torch.no_grad():
        # torch.as_tensor can alias the preparation arrays, and passthrough
        # may alias binary. Isolate the three intentionally changed buffers.
        num.center=num.center.clone();num.factor=num.factor.clone();num.passthrough=num.passthrough.clone()
        num.center[phi:]=0.;num.factor[phi:]=1.;num.passthrough[phi:]=True
    assert tensor_hashes(model)==before
    allowed={'base.numerical.center','base.numerical.factor','base.numerical.passthrough'}
    changed=[]
    for n,b in model.named_buffers():
        if not torch.equal(b,buffers[n]):
            assert n in allowed and torch.equal(b[:phi],buffers[n][:phi]),n
            changed.append(n)
    assert set(changed)==allowed
    model.variant=VARIANT
    assert not model.main_residual_enabled and not model.auxiliary_enabled and model.members==8
    assert sum(p.numel() for p in model.parameters())==12902328
    assert all(p.requires_grad for p in model.parameters())
    # Independently check the two input paths and the exact no-double-scaling identity.
    ix=np.arange(16);batch,_=data.batch('train',ix,device='cpu')
    raw=torch.as_tensor(np.array(data.raw['train'][ix],copy=True))
    assert torch.equal(batch['raw'][:,:phi],raw[:,:phi])
    q=data.quantile.transform(np.array(data.raw['train'][ix,phi:],copy=True))
    np.testing.assert_array_equal(batch['raw'][:,phi:].numpy(),q)
    with torch.no_grad():
        z=num(batch['raw']);retained=z[:,-data.input_dim:]
        original=(raw-buffers['base.numerical.center'])*buffers['base.numerical.factor']
        original=torch.where(buffers['base.numerical.passthrough'],raw,original/torch.sqrt(1+(original/3).square()))
        assert torch.equal(retained[:,:phi],original[:,:phi])
        assert torch.equal(retained[:,phi:],batch['raw'][:,phi:])
    audit=dict(complete=True,reference_seed=seed,retained_CEM_tensors_exact=True,
        learned_checkpoint_loaded=False,all_retained_parameters_trainable=True,
        reference_initial_weights_sha256=ref['initial_weights_sha256'],initial_weights_sha256=parameter_hash(model),
        retained_tensor_hashes=before,changed_buffers=changed,changed_coordinates=[phi,data.input_dim],
        event_coordinates_unchanged=True,quantile_transform_exact=True,double_scaling=False,
        P_retained=True,M_retained=True,members=8,main_loss_coefficient=.5,
        evidence_branch_present=False,auxiliary_present=False,transform_sha256=expected()['transform_sha256'],
        transform_fit_split='train',transform_refitted=False,test_loaded=False)
    atomic(out/'initialization_observation.json',audit);atomic(out/'connection_audit.json',audit)
    return model
