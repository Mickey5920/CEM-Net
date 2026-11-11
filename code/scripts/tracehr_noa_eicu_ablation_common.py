"""Authorized eICU P/M/R removals, with the previous scientific implementation."""
from pathlib import Path
from scripts.tracehr_noa_validation_common import (
    Data, stopping, model_for, protocol as parent_protocol, PROTOCOL_FIELDS,
    Optimizer, diagnostics, parameter_hash, read, VARIANTS)

RUN = 'tracehr_noa_eicu_ablation_20261020'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48, 49, 50)
BASELINES = ('B0', 'tabm', 'xgboost')

def expected():
    return read(ROOT / 'work' / RUN / 'expected.json')

def matrix(database):
    assert database == 'eicu'
    # Prioritize the six noR fits, then complete all predeclared P/M fits.
    return tuple((v, e, s) for v in (VARIANTS[2], VARIANTS[0], VARIANTS[1])
                 for s in SEEDS for e in (0, 1))

def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = parent_protocol('C5_noT_noA', database, seed)
    spec.update(variant=variant, structure=variant, ablated_component=variant[-1],
        description='single deletion inside noA; fixed z0; 0.5 main CE; no auxiliary loss',
        ablation_implementation={
            'P': 'zero slots preserve native width and retained initial tensors',
            'M': 'identity input modulation; eight internal members and normalization preserved',
            'R': 'delete summary-to-main residual; prune the now disconnected summary branch'}[variant[-1]])
    return spec
