"""Seed-only extension of the archived core experiments; no recipe search."""
import gc
import os
from pathlib import Path
import numpy as np
import torch
from scripts.tracehr_multiview_common import read, sha, atomic
from models.tracehr_unified_next import parameter_hash
from scripts.mimiciii_ema_common import EARLY, EMA_RULE, tensor_hashes

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_core_five_seed_20261107'
SEEDS = (51, 52)

def expected(): return read(ROOT/'work'/RUN/'config.json')
def database(): return os.environ['TRACEHR_DATABASE']
def matrix(): return expected()['matrix']
def variants(): return expected()['variants']
def budget(seed, endpoint):
    return expected()['budgets'][f'{seed}:{endpoint}']

def protocol(variant, seed):
    return dict(expected()['control_protocols'][variant], seed=seed)

def Data(project, preparation):
    if expected()['stage'] == 'hospital':
        from scripts.cem_icde72_hospital import HospitalData
        return HospitalData(project, preparation)
    from scripts.cem_evidence_common import Data as original
    return original(project, preparation)

def full_model(data, preparation, seed, endpoint):
    from scripts import cem_five_seed_common as five
    five.expected = expected
    if expected()['stage'] != 'hospital':
        return five.construct(data, preparation, seed, endpoint)
    # The hospital split has its own train-only preprocessing and input width.
    from models import tracehr_noa_validation as ctor
    from models.five_seed_noa_validation import NoAAblation
    from scripts import tracehr_noa_validation_common as route
    ctor.NoAAblation = route.NoAAblation = NoAAblation
    from scripts import five_seed_constructor_route
    from scripts import tracehr_fte_binding_common as old
    old.SEEDS = (48, 49, 50, 51, 52)
    return old.model_for(data, preparation, old.VARIANT, endpoint, seed)

def intervene(model, data, variant):
    if variant in ('EMA_noP', 'EMA_noM'):
        from models.mimiciii_pm_mini import delete_component
        return delete_component(model, variant[-1])
    if variant in ('EMA_no_duration', 'EMA_no_NRE_CMM'):
        from models.cem_evidence_controls import intervene as operation
        return operation(model, variant, data.record['phi_dim'])
    if variant in ('EMA_M_raw', 'EMA_M_periodic', 'EMA_no_numeric_direct'):
        from models.ema_internal import intervene as operation
        return operation(model, variant)
    if variant == 'CEM_basic':
        from scripts.cem_icde72_neural import removed_columns
        from models.cem_icde72_mask import MaskedPeriodic
        before = tensor_hashes(model)
        model.base.numerical = MaskedPeriodic(model.base.numerical, removed_columns(data))
        assert tensor_hashes(model) == before
    else:
        assert variant in ('CEM_hospital','CEM_noEMA')
    return model, dict(retained_initials_exact=True, original_buffers_exact=True,
                       intervention=variant, parameter_slots_unchanged=True)

def fresh_model(data, preparation, seed, endpoint, out, variant):
    model = full_model(data, preparation, seed, endpoint)
    full_hash = parameter_hash(model)
    if expected()['stage'] != 'hospital':
        ref = expected()['full_ema'][f'{seed}:{endpoint}']
        assert full_hash == ref['initial_weights_sha256']
    model, mapping = intervene(model, data, variant)
    ref = read(Path(os.environ['CEM_CORE_VERIFICATION']))['initials'][f'{variant}:{seed}:{endpoint}']
    assert parameter_hash(model) == ref['initial_weights_sha256']
    assert tensor_hashes(model) == ref['tensor_hashes']
    from scripts.cem_five_seed_common import sha_tensor
    assert {n:sha_tensor(b) for n,b in model.named_buffers()} == ref['buffer_hashes']
    record = dict(complete=True, optimizer_updates=0, reference_seed=seed,
                  full_initial_weights_sha256=full_hash, initial_weights_sha256=parameter_hash(model),
                  full_initial_tensors_exact=True, original_static_buffers_exact=True,
                  retained_tensor_hashes=tensor_hashes(model), deletion=mapping,
                  learned_checkpoint_loaded=False, all_retained_parameters_trainable=True)
    atomic(out/'initialization_observation.json', record)
    atomic(out/'connection_audit.json', record)
    return model

def bind_runtime():
    from scripts import core_seed_ema_runtime as runtime
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

def configure():
    from scripts import cem_additional_common as tab
    from scripts import cem_icde72_common as tree
    tab.expected = tree.expected = expected
    if expected()['stage'] in ('basic', 'hospital'):
        from scripts import cem_icde72_neural as neural
        neural.expected = expected
        if expected()['stage'] == 'hospital':
            from scripts import cem_icde72_hospital as hospital
            hospital.expected = expected
            tab.Data = hospital.HospitalData
    if expected()['stage'] == 'replacement':
        from scripts import cem_replacement_common as replacement
        replacement.expected = expected
        replacement.RUN = RUN
    return tab

def tabular_model(variant, seed, data):
    from scripts.cem_five_seed_common import make_tabular
    if variant=='TabM':
        from scripts.train_tracehr_unified import seed_all
        from tabm import TabM
        seed_all(seed)
        model=TabM.make(n_num_features=data.input_dim,d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2)
        model._initial_torch_rng=torch.random.get_rng_state()
        return model
    if variant.endswith('_basic') or variant.endswith('_hospital'):
        from models.cem_icde72_mask import mask_tabm
        from scripts.cem_icde72_neural import removed_columns
        base = variant.removesuffix('_basic').removesuffix('_hospital')
        model = make_tabular(base, seed, data)
        if variant.endswith('_basic'):
            before = parameter_hash(model)
            model = mask_tabm(model, removed_columns(data))
            assert parameter_hash(model) == before
        return model
    return make_tabular(variant, seed, data)

def verify(project):
    from scripts.train_tracehr_unified import seed_all
    from scripts.cem_five_seed_common import sha_tensor
    from scripts.cem_evidence_host_queue import policy_checks
    configure(); c = expected(); torch.set_num_threads(4)
    initials = {}; archived = {}
    if c['stage'] in ('controls', 'basic', 'hospital', 'legacy'):
        data = Data(project, project/c['input_preparation'])
        for variant in c['variants']:
            if not (variant.startswith('EMA_') or variant.startswith('CEM_')): continue
            for endpoint in sorted({j['endpoint'] for j in matrix() if j['variant'] == variant}):
                for seed in (48, 51, 52):
                    seed_all(seed)
                    model, mapping = intervene(full_model(data, project/c['input_preparation'], seed, endpoint), data, variant)
                    value = dict(initial_weights_sha256=parameter_hash(model),
                                 parameters=sum(p.numel() for p in model.parameters()),
                                 tensor_hashes=tensor_hashes(model),
                                 buffer_hashes={n:sha_tensor(b) for n,b in model.named_buffers()})
                    if seed == 48:
                        old = read(Path(c['archived_controls'][f'{variant}:{endpoint}'])/'run_manifest.json')
                        assert value['initial_weights_sha256'] == old['initial_weights_sha256'], variant
                        assert value['parameters'] == old['parameters'], variant
                        assert {k:v for k,v in protocol(variant,seed).items() if k!='seed'} == {k:v for k,v in old['protocol'].items() if k!='seed'}
                        archived[f'{variant}:{endpoint}'] = dict(initial_exact=True, protocol_exact=True)
                    else:
                        initials[f'{variant}:{seed}:{endpoint}'] = value
                    del model; gc.collect()
        del data; gc.collect()
    if c['stage'] in ('tabular','sequence','basic','hospital','legacy'):
        from scripts.cem_additional_common import make_data
        data = make_data(project)
        for variant in c['variants']:
            if variant.startswith(('EMA_','CEM_','LightGBM')) or variant=='RealMLP': continue
            for endpoint in sorted({j['endpoint'] for j in matrix() if j['variant']==variant}):
                seed_all(48); model = tabular_model(variant,48,data)
                old = read(Path(c['archived_controls'][f'{variant}:{endpoint}'])/'run_manifest.json')
                if 'initial_weights_sha256' in old:
                    assert parameter_hash(model)==old['initial_weights_sha256'],(variant,endpoint)
                    assert sum(p.numel() for p in model.parameters())==old['parameters']
                else:
                    assert variant=='TabM' and sha(ROOT/'train_tracehr_tabular.py')==old['source_sha256']
                archived[f'{variant}:{endpoint}']=dict(initial_exact=True,parameters_exact=True)
                del model;gc.collect()
        del data;gc.collect()
    # Existing preprocessing artifacts are reused without fitting new transforms.
    if c.get('baseline_preparation'):
        prep = project/c['baseline_preparation']; record = read(prep/'completion.json')
        for name, value in record.get('artifacts', {}).items():
            assert sha(prep/name) == (value['sha256'] if isinstance(value,dict) else value), name
    return dict(complete=True, formal_fits=len(matrix()), test_evaluations=len(matrix()),
                initials=initials, archived_seed48=archived, test_loaded=False,
                cpu_checks=policy_checks(), original_preprocessing_preserved=True)
