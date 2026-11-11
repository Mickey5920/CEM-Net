"""Frozen scope and recipes for 12 deletions plus six TabM-mini fits."""
from pathlib import Path
from scripts.mimiciii_baselines_common import numeric_protocol as previous_protocol
from scripts.tracehr_multiview_common import read, sha, atomic

ROOT = Path(__file__).resolve().parents[1]
RUN = 'mimiciii_pm_mini_20261024'
VARIANTS = ('noEvidence_noP', 'noEvidence_noM', 'TabM-mini')
SEEDS = (48, 49, 50)


def expected():
    return read(ROOT / 'work' / RUN / 'expected.json')


def protocol(variant, seed):
    assert variant in VARIANTS and seed in SEEDS
    p = previous_protocol('TabM' if variant == 'TabM-mini' else 'noEvidence', seed)
    p.update(variant=variant)
    if variant == 'TabM-mini':
        p.update(arch_type='tabm-mini', modification='only official arch_type differs from archived TabM')
    else:
        p.update(structure=variant, ablated_component=variant[-1],
                 description='single deletion in fixed branch-free untied main model')
    return p


def matrix():
    return [dict(variant=v, seed=s, endpoint=e) for s in SEEDS for v in VARIANTS for e in (0, 1)]


def reference(variant, seed, endpoint):
    m = 'TabM' if variant == 'TabM-mini' else 'noEvidence'
    return expected()['references'][f'{m}_s{seed}_t{endpoint}']
