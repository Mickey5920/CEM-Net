"""One authorized V1 encoder/readout rate split; unchanged model and training loop."""
import hashlib, os
from pathlib import Path
from scripts.cem_gate_common import (Data, CEM, stopping, atomic, read, sha,
    parameter_hash, diagnostics, np, torch, PROTOCOL_FIELDS as GATE_FIELDS,
    protocol as v1_protocol, model_for as original_model_for)
from scripts.tracehr_structure_runtime import Optimizer as OriginalOptimizer
from scripts.tracehr_backbone_refine_runtime import Optimizer as BaseOptimizer

RUN = 'tracehr_cem_v1_group_rate_20261023'
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('V1_group_rate',)
SEEDS = (48,)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
READOUT_NAMES = frozenset(('output.weight', 'output.bias', 'output_norm.weight',
    'output_norm.bias', 'extra_output.weight', 'extra_output.bias', 'residual_scale', 'extra_scale'))
RATE_FIELDS = {'variant', 'fusion_readout_lr', 'fusion_readout_parameter_names', 'tuning'}
PROTOCOL_FIELDS = GATE_FIELDS | RATE_FIELDS


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return tuple((v, e, 48) for v in VARIANTS for e in (0, 1))


def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = v1_protocol('V1', database, seed)
    assert spec['fusion_lr'] == .00030
    spec.update(variant=variant, fusion_readout_lr=.00015,
                fusion_readout_parameter_names=sorted(READOUT_NAMES), tuning=True)
    return spec


def split_fusion(fusion):
    named = dict(fusion.named_parameters())
    assert READOUT_NAMES <= named.keys(), READOUT_NAMES-named.keys()
    assert 'gate.bias' in named
    groups = {role: [(n, p) for n, p in named.items() if (n in READOUT_NAMES) == reduced]
              for role, reduced in (('encoder', False), ('readout', True))}
    assert set(n for n, _ in groups['readout']) == READOUT_NAMES
    assert all(p.requires_grad for p in named.values())
    return groups


def model_for(data, preparation, variant, endpoint, seed):
    assert (variant, endpoint, seed) in matrix(data.database if hasattr(data, 'database') else 'mimic')
    model = original_model_for(data, preparation, 'V1', endpoint, seed)
    ref = expected()['controls'][f'V1_t{endpoint}_s{seed}']
    observed = read(Path(ref['source'])/'initialization_observation.json')['retained_tensor_hashes']
    actual = {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for n,p in model.named_parameters()}
    evidence = dict(optimizer_updates=0, reference_variant='V1', reference_seed=seed,
        expected_initial_weights_sha256=ref['initial_weights_sha256'],
        actual_initial_weights_sha256=parameter_hash(model), parameters=sum(p.numel() for p in model.parameters()),
        tensor_mismatches=[n for n in actual.keys()&observed.keys() if actual[n]!=observed[n]],
        removed_names=sorted(set(observed)-set(actual)), new_names=sorted(set(actual)-set(observed)),
        learned_checkpoint_loaded=False, structural_variant=model.variant)
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']).parent/'V1_initialization_observation.json', evidence)
    assert evidence['actual_initial_weights_sha256']==evidence['expected_initial_weights_sha256']
    assert evidence['parameters']==ref['parameters']
    assert not evidence['tensor_mismatches'] and not evidence['removed_names'] and not evidence['new_names']
    assert model.variant=='V1'
    model.info['archived_V1_all_initial_parameters_verified']=True
    return model


class Optimizer(OriginalOptimizer):
    """Split AdamW groups; inherit the original update and recovery operations."""
    def __init__(self, model, spec, lr_multiplier=1.):
        assert lr_multiplier==1. and spec==protocol(spec['variant'], 'mimic', 48)
        assert model.variant=='V1'
        self.model, self.spec = model, spec
        self.base = BaseOptimizer(model.base, spec, 1.)
        groups = split_fusion(model.fusion)
        self.fusion = torch.optim.AdamW([
            dict(params=[p for n,p in members], parameter_names=[n for n,p in members], role=role,
                 lr=spec['fusion_lr'] if role=='encoder' else spec['fusion_readout_lr'])
            for role,members in groups.items()], betas=(.9,spec['fusion_beta2']),
            weight_decay=spec['fusion_weight_decay'])
        actual = [id(p) for g in self.base.inner.param_groups for p in g['params']]
        if self.base.adapter:
            actual += [id(p) for g in self.base.adapter.param_groups for p in g['params']]
        actual += [id(p) for g in self.fusion.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        self.base.schedule(epoch_float)
        factor = float(self.base.wrapped.get_hyper_values('lr',self.base.clock_group)/self.base.clock_base)
        for group in self.fusion.param_groups:
            group['lr'] = factor*self.spec['fusion_lr' if group['role']=='encoder' else 'fusion_readout_lr']

    def values(self):
        value = super().values()
        value.update(protocol_search=True, prespecified_group_rate_only=True, fixed_structural_variant='V1',
            fusion_groups={g['role']:dict(lr=g['lr'],betas=g['betas'],weight_decay=g['weight_decay'],
                parameter_names=g['parameter_names'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params']))
                for g in self.fusion.param_groups})
        return value
