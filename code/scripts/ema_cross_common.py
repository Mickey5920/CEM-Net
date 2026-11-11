"""Fixed cross-database EMA confirmation; no subsequent deletion experiments."""
import hashlib
import os
from pathlib import Path
import torch
from scripts.tracehr_fte_binding_common import Data, model_for, protocol as original_protocol, VARIANT as ORIGINAL
from scripts.tracehr_multiview_common import atomic, read, sha
from models.tracehr_unified_next import parameter_hash
from scripts.mimiciii_ema_common import EMA_RULE, EARLY

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tracehr_ema_cross_20261024'
VARIANT = 'fullEMA'
SEEDS = (48, 49, 50)

def database(): return os.environ['TRACEHR_DATABASE']
def expected(): return read(ROOT/'work'/RUN/(database()+'_expected.json'))
def matrix():
    return [dict(variant=v,seed=s,endpoint=e) for v in (VARIANT,'TabM-mini') for s in SEEDS for e in (0,1)]
def budget(seed, endpoint):
    assert seed in SEEDS and endpoint in (0,1)
    return 52 if database()=='mimic' else 256
def protocol(variant, seed):
    if variant==VARIANT: return original_protocol(ORIGINAL,database(),seed)
    assert variant=='TabM-mini'
    return dict(variant=variant,seed=seed,epochs=256,batch=256,k=32,d_block=256,n_blocks=3,dropout=.2,
                optimizer='AdamW',lr=.0005,weight_decay=.001,patience=16,selection='endpoint dev AP',
                precision='float32',loss='mean member BCE',tail='retained',arch_type='tabm-mini')

def fresh_model(data, preparation, seed, endpoint, out):
    model=model_for(data,preparation,ORIGINAL,endpoint,seed)
    ref=expected()['references'][f'noEvidence_s{seed}_t{endpoint}']
    hashes={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    observation=dict(optimizer_updates=0,initial_weights_sha256=parameter_hash(model),retained_tensor_hashes=hashes,
                     reference_initial_weights_sha256=ref['initial_weights_sha256'],reference_seed=seed)
    atomic(out/'initialization_observation.json',observation)
    assert parameter_hash(model)==ref['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters())==ref['parameters']
    assert all(p.requires_grad for p in model.parameters()) and model.members==8
    assert not model.main_residual_enabled and not model.auxiliary_enabled and not list(model.fusion.parameters())
    old=torch.load(Path(ref['output'])/'epoch_001_model.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(b.detach().cpu(),old['model'][n]) for n,b in model.named_buffers())
    del old
    audit=dict(complete=True,**observation,retained_CEM_tensors_exact=True,original_static_buffers_exact=True,
               learned_checkpoint_loaded=False,reference_checkpoint_read_for_buffer_comparison_only=True,
               P_retained=True,M_retained=True,members=8,main_loss_coefficient=.5,evidence_branch_present=False,
               auxiliary_present=False,preprocessing_unchanged=True,all_retained_parameters_trainable=True,test_loaded=False)
    atomic(out/'connection_audit.json',audit)
    return model

def bind_ema_runtime():
    from scripts import mimiciii_ema_runtime as runner
    from scripts.tracehr_fte_binding_runtime import update,predict
    runner.expected=expected;runner.budget=budget;runner.update=update;runner.predict=predict
    return runner
