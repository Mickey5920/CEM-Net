"""Exactly 24 PM-internal deletions across the two existing database splits."""
from pathlib import Path
from scripts.tracehr_noa_validation_common import (
    Data, stopping, protocol as prior_protocol, PROTOCOL_FIELDS,
    Optimizer, diagnostics, parameter_hash, read, e_model, StructureNetwork, remove_hidden_context)
from models.tracehr_pm_ablation import PMAblation, VARIANTS, PM

RUN = 'tracehr_pm_ablation_20261021'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48, 49, 50)
BASELINES = ('B0', 'tabm', 'xgboost')

def expected():
    return read(ROOT / 'work' / RUN / 'expected.json')

def matrix(database):
    assert database in ('mimic', 'eicu')
    return tuple((v, e, s) for s in SEEDS for v in VARIANTS for e in (0, 1))

def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = prior_protocol(PM if database == 'mimic' else 'C5_noT_noA', database, seed)
    spec.update(variant=variant, structure=variant, ablated_component=variant[-1],
        description='single P/M deletion inside PM; R/T/A absent; fixed z0; 0.5 main CE',
        ablation_implementation={
            'P': 'PM without learned periodic encoding; zero slots preserve native width and retained initial tensors',
            'M': 'PM without input modulation; eight internal members and normalization preserved'}[variant[-1]])
    return spec

def model_for(data, preparation, variant, endpoint, seed):
    assert variant in VARIANTS and endpoint in (0, 1) and seed in SEEDS
    base = e_model(data, preparation, 'E2', endpoint, seed)
    base_hash = parameter_hash(base)
    c1 = remove_hidden_context(StructureNetwork(base, data.record['phi_dim'], 'G2', seed), seed)
    model = PMAblation(c1, variant, seed)
    model.info['seed_route'] = dict(seed=seed, native_initialization_seed=seed,
        periodic_namespace_seed=seed, input_members_namespace_seed=seed, fusion_namespace_seed=seed,
        auxiliary_head_namespace_seed=seed, training_order_seed=seed, training_torch_seed=seed+10000,
        base_initial_weights_sha256=base_hash)
    return model
