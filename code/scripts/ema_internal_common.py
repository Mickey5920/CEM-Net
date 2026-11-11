"""Routing for the authorized eICU M-path and MIMIC-III numeric-direct controls."""
import os
from pathlib import Path
from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.mimiciii_ema_common import EARLY, EMA_RULE, tensor_hashes
from models.tracehr_unified_next import parameter_hash
from models.ema_internal import intervene

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tracehr_ema_internal_20261025'
SEEDS = (48, 49, 50)


def database(): return os.environ['TRACEHR_DATABASE']
def expected(): return read(ROOT/'work'/RUN/(database()+'_expected.json'))
def variants(): return ('EMA_M_raw','EMA_M_periodic') if database() == 'eicu' else ('EMA_no_numeric_direct',)
def matrix(): return [dict(variant=v,seed=s,endpoint=e) for s in SEEDS for v in variants() for e in (0,1)]
def pilot_name(v,e): return f'probe_{v}_t{e}_s48'
def budget(s,e): return 256 if database()=='eicu' else (120 if e==0 or s==49 else 52)
def protocol(v,s):
    assert v in variants() and s in SEEDS
    if database()=='eicu':
        from scripts.ema_cross_common import protocol as parent
        return parent('fullEMA',s)
    from scripts.mimiciii_ema_common import protocol as parent
    return parent('noEvidence_EMA',s)


def Data(project, preparation):
    if database()=='eicu':
        from scripts.ema_cross_common import Data as Parent
        return Parent(project,'eicu',preparation)
    from scripts.mimiciii_fixed_common import Data as Parent
    return Parent(preparation)


def full_model(data, preparation, seed, endpoint, out):
    if database()=='eicu':
        from scripts.ema_cross_common import fresh_model
    else:
        from scripts.mimiciii_ema_common import fresh_model
    return fresh_model(data,preparation,seed,endpoint,out)


def fresh_model(data, preparation, seed, endpoint, out, variant):
    model = full_model(data,preparation,seed,endpoint,out)
    full_hash = parameter_hash(model)
    ref = expected()['full_ema'][f'{seed}:{endpoint}']
    assert full_hash == ref['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters()) == ref['parameters']
    original = read(out/'connection_audit.json')
    model, mapping = intervene(model, variant)
    observation = dict(complete=True,optimizer_updates=0,reference_seed=seed,
        full_initial_weights_sha256=full_hash,initial_weights_sha256=parameter_hash(model),
        retained_tensor_hashes=tensor_hashes(model),deletion=mapping,full_initial_tensors_exact=True,
        original_static_buffers_exact=True)
    atomic(out/'initialization_observation.json',observation)
    original.update(observation,P_retained=True,M_retained=True,evidence_branch_present=False,auxiliary_present=False,
        learned_checkpoint_loaded=False,all_retained_parameters_trainable=True)
    atomic(out/'connection_audit.json',original)
    return model


def bind_runtime():
    from scripts import ema_internal_runtime as runtime
    from scripts import mimiciii_ema_runtime as ema
    if database()=='eicu':
        from scripts.tracehr_fte_binding_runtime import update,predict
    else:
        from scripts.cem_v1_group_confirm_runtime import update,predict
    runtime.budget=budget;runtime.update=update;runtime.predict=predict
    ema.update=update;ema.predict=predict
    def probe(a,out,data,model,identity,spec):
        ema.probe(a,out,data,model,identity,spec)
        atomic(out/'path_audit.json',read(out/'ema_path_audit.json'))
    runtime.probe=probe
    return runtime
