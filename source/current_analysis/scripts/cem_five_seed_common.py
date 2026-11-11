"""Five-seed extension of the existing Table VI cells; no new model recipe."""
import os
from pathlib import Path
import numpy as np
import torch
from scripts.tracehr_multiview_common import read, sha, atomic
from models.tracehr_unified_next import parameter_hash
from scripts.mimiciii_ema_common import EMA_RULE, EARLY, tensor_hashes

RUN = 'cem_five_seed_20261102'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (51, 52)

def database(): return os.environ['TRACEHR_DATABASE']
def expected(): return read(ROOT/'work'/RUN/'config.json')
def matrix(): return expected()['matrix']
def budget(seed, endpoint):
    assert seed in SEEDS and endpoint in (0, 1)
    return 256

def budget_stop(history, spec):
    # Exactly the archived non-EMA decision, evaluated on the raw trajectory
    # already present in the EMA fit. EMA predictions never determine its length.
    if database() == 'eicu':
        return dict(stop=len(history) == 256, reason='original eICU fixed 256 epochs')
    from scripts.tracehr_centered_native_common import stopping
    raw = [dict(epoch=r['epoch'], dev_ap=r['raw_metrics']['mean_ap']) for r in history]
    return stopping(raw, spec)

def protocol(variant, seed):
    assert variant == 'CEM' and seed in (48, 49, 50, 51, 52)
    return dict(expected()['full_ema']['48:0']['protocol'], seed=seed)

def Data(project, preparation):
    from scripts.cem_evidence_common import Data as original
    return original(project, preparation)

def construct(data, preparation, seed, endpoint):
    # The archived constructors restrict seed IDs independently of their math.
    # New immutable copies extend only that whitelist, retaining every operation.
    from models import tracehr_noa_validation as constructor_module
    from models.five_seed_noa_validation import NoAAblation
    from scripts import tracehr_noa_validation_common as constructor_route
    constructor_module.NoAAblation = constructor_route.NoAAblation = NoAAblation
    from scripts import five_seed_constructor_route
    if database() == 'mimiciii':
        from scripts import mimiciii_fixed_common as old
        from scripts.train_mimiciii_numeric_baseline import NoEvidence
        old.SEEDS = (48, 49, 50, 51, 52)
        model = NoEvidence(old.model_for(data, preparation, endpoint, seed))
    else:
        from scripts import tracehr_fte_binding_common as old
        old.SEEDS = (48, 49, 50, 51, 52)
        model = old.model_for(data, preparation, old.VARIANT, endpoint, seed)
    assert sum(p.numel() for p in model.parameters()) == expected()['full_ema'][f'48:{endpoint}']['parameters']
    assert all(p.requires_grad for p in model.parameters()) and model.members == 8
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    return model

def fresh_model(data, preparation, seed, endpoint, out, variant):
    assert variant == 'CEM'
    model = construct(data, preparation, seed, endpoint)
    v = read(Path(os.environ['CEM_FIVE_VERIFICATION']))
    reference = v['initials'][f'CEM:{seed}:{endpoint}']
    assert parameter_hash(model) == reference['initial_weights_sha256']
    assert tensor_hashes(model) == reference['tensor_hashes']
    buffers = {n: sha_tensor(b) for n, b in model.named_buffers()}
    assert buffers == reference['buffer_hashes']
    observation = dict(complete=True, optimizer_updates=0, initial_weights_sha256=parameter_hash(model),
        retained_tensor_hashes=tensor_hashes(model), original_static_buffers_exact=True,
        full_initial_tensors_exact=True, reference_seed=seed, learned_checkpoint_loaded=False,
        all_retained_parameters_trainable=True, auxiliary_present=False, members=8,
        CPU_reference_sha256=sha(Path(os.environ['CEM_FIVE_VERIFICATION'])))
    atomic(out/'initialization_observation.json', observation)
    atomic(out/'connection_audit.json', observation)
    return model

def sha_tensor(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def configure():
    from scripts import cem_additional_common as tab
    from scripts import cem_icde72_common as tree
    tab.expected = expected
    tree.expected = expected
    return tab

def make_tabular(variant, seed, data):
    from scripts.train_tracehr_unified import seed_all
    seed_all(seed)
    if variant == 'TabM_mini':
        from tabm import TabM
        model = TabM.make(n_num_features=data.input_dim, d_out=1, k=32,
                         d_block=256, n_blocks=3, dropout=.2, arch_type='tabm-mini')
        # Original no-PLE TabM-mini starts dropout RNG after construction.
        model._initial_torch_rng = torch.random.get_rng_state()
        return model
    from models.cem_additional_baselines import make_model
    return make_model(variant, seed, data)

def bind_runtime():
    from scripts import five_seed_cem_runtime as runtime
    from scripts import mimiciii_ema_runtime as ema
    if database() == 'mimiciii':
        from scripts.cem_v1_group_confirm_runtime import update, predict
    else:
        from scripts.tracehr_fte_binding_runtime import update, predict
    runtime.update = ema.update = update
    runtime.predict = ema.predict = predict
    def probe(a, out, data, model, identity, spec):
        ema.probe(a, out, data, model, identity, spec)
        atomic(out/'path_audit.json', read(out/'ema_path_audit.json'))
    runtime.probe = probe
    return runtime

def verify(project):
    import gc
    from scripts.train_tracehr_unified import seed_all
    from scripts.cem_evidence_host_queue import policy_checks
    c = expected(); torch.set_num_threads(4); configure()
    data = Data(project, project/c['input_preparation'])
    initials = {}; archived = {}
    for endpoint in (0, 1):
        seed_all(48); model = construct(data, project/c['input_preparation'], 48, endpoint)
        ref = c['full_ema'][f'48:{endpoint}']
        assert parameter_hash(model) == ref['initial_weights_sha256']
        old = torch.load(Path(ref['output'])/'best.pt', map_location='cpu', weights_only=False)
        assert all(torch.equal(b, old['model'][n]) for n, b in model.named_buffers())
        buffers = {n: sha_tensor(b) for n, b in model.named_buffers()}
        archived[str(endpoint)] = dict(initial_exact=True, static_buffers_exact=True)
        del model, old; gc.collect()
        for seed in SEEDS:
            seed_all(seed); model = construct(data, project/c['input_preparation'], seed, endpoint)
            assert parameter_hash(model) != ref['initial_weights_sha256']
            seed_buffers = {n:sha_tensor(b) for n,b in model.named_buffers()}
            seed_hash = parameter_hash(model)
            initials[f'CEM:{seed}:{endpoint}'] = dict(initial_weights_sha256=parameter_hash(model),
                parameters=sum(p.numel() for p in model.parameters()), tensor_hashes=tensor_hashes(model),
                buffer_hashes=seed_buffers)
            del model; gc.collect()
            seed_all(seed); repeat = construct(data, project/c['input_preparation'], seed, endpoint)
            assert parameter_hash(repeat) == seed_hash
            assert {n:sha_tensor(b) for n,b in repeat.named_buffers()} == seed_buffers
            del repeat; gc.collect()
    # Reuse the existing fitted transforms and quantile edges. Never refit them.
    prep = project/c['baseline_preparation']; record = read(prep/'completion.json')
    for name, h in record['artifacts'].items(): assert sha(prep/name) == h
    assert sha(Path(c['tabm_preprocessing'])) == c['tabm_preprocessing_sha256']
    from scripts.mimiciii_ema_runtime import cpu_checks
    checks = policy_checks() + cpu_checks()[:3]
    return dict(complete=True, formal_fits=len(matrix()), test_evaluations=len(matrix()),
        initials=initials, archived_seed48=archived, test_loaded=False,
        train_rows=len(data.targets['train']['y']), dev_rows=len(data.targets['dev']['y']),
        input_dim=data.input_dim, cpu_checks=checks, original_preprocessing_preserved=True)
