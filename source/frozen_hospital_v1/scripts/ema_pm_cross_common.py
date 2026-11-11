"""Single P/M deletions of the fixed cross-database EMA model."""
import os
from pathlib import Path
from scripts import ema_cross_common as parent
from scripts.ema_cross_common import Data, EARLY, EMA_RULE, parameter_hash
from scripts.mimiciii_ema_common import tensor_hashes
from scripts.tracehr_multiview_common import atomic, read, sha
from models.mimiciii_pm_mini import delete_component

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tracehr_ema_pm_cross_20261025'
VARIANTS = ('EMA_noP', 'EMA_noM')
SEEDS = (48, 49, 50)

def database(): return os.environ['TRACEHR_DATABASE']
def expected(): return read(ROOT/'work'/RUN/(database()+'_expected.json'))
def budget(seed, endpoint): return parent.budget(seed, endpoint)
def protocol(variant, seed):
    assert variant in VARIANTS
    return parent.protocol('fullEMA', seed)
def matrix():
    return [dict(variant=v, seed=s, endpoint=e) for s in SEEDS for v in VARIANTS for e in (0, 1)]
def pilot_name(variant, endpoint): return f'probe_{variant}_t{endpoint}_s48'

def fresh_model(data, preparation, seed, endpoint, out, variant):
    assert variant in VARIANTS
    model = parent.fresh_model(data, preparation, seed, endpoint, out)
    full_hash = parameter_hash(model)
    ref = expected()['full_ema'][f'fullEMA_t{endpoint}_s{seed}']
    assert full_hash == ref['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters()) == ref['parameters']
    original_audit = read(out/'connection_audit.json')
    model, mapping = delete_component(model, variant[-1])
    observation = dict(complete=True, optimizer_updates=0, reference_seed=seed,
        full_initial_weights_sha256=full_hash, initial_weights_sha256=parameter_hash(model),
        retained_tensor_hashes=tensor_hashes(model), deletion=mapping,
        full_initial_tensors_exact=True, original_static_buffers_exact=True)
    atomic(out/'initialization_observation.json', observation)
    audit = dict(original_audit)
    audit.update(complete=True, P_retained=variant.endswith('M'), M_retained=variant.endswith('P'),
        deletion=mapping, full_initial_weights_sha256=full_hash,
        initial_weights_sha256=parameter_hash(model), retained_CEM_tensors_exact=True,
        preprocessing_unchanged=True, all_retained_parameters_trainable=True)
    atomic(out/'connection_audit.json', audit)
    assert model.members == 8 and all(p.requires_grad for p in model.parameters())
    return model

def bind_probe():
    runtime = parent.bind_ema_runtime()
    runtime.expected = expected
    return runtime.probe
