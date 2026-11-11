"""One authorized V1 encoder/readout rate split; unchanged model and training loop."""
import hashlib, os
from scripts import cem_v1_group_confirm_seed_route
from pathlib import Path
from scripts.cem_gate_common import (Data, CEM, stopping, atomic, read, sha,
    parameter_hash, diagnostics, np, torch, PROTOCOL_FIELDS as GATE_FIELDS,
    protocol as archived_v1_protocol)
from scripts.tracehr_structure_runtime import Optimizer as OriginalOptimizer
from scripts.tracehr_backbone_refine_runtime import Optimizer as BaseOptimizer

RUN = 'tracehr_cem_v1_group_confirm_20261023'
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('V1_group_rate',)
SEEDS = (49, 50)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
READOUT_NAMES = frozenset(('output.weight', 'output.bias', 'output_norm.weight',
    'output_norm.bias', 'extra_output.weight', 'extra_output.bias', 'residual_scale', 'extra_scale'))
RATE_FIELDS = {'variant', 'fusion_readout_lr', 'fusion_readout_parameter_names', 'tuning'}
PROTOCOL_FIELDS = GATE_FIELDS | RATE_FIELDS


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return tuple((v, e, s) for s in SEEDS for v in VARIANTS for e in (0, 1))


def v1_protocol(variant, database, seed):
    assert variant == 'V1' and database == 'mimic' and seed in SEEDS
    spec = archived_v1_protocol('V1', database, 48)
    spec['seed'] = seed
    return spec


def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = v1_protocol('V1', database, seed)
    spec.update(variant=variant, fusion_readout_lr=.00015,
                fusion_readout_parameter_names=sorted(READOUT_NAMES), tuning=False)
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
    assert (variant, endpoint, seed) in matrix('mimic')
    from scripts.cem_structure_common import cem_model
    from models.cem_gate_location import GateStructure
    fresh = cem_model(data, preparation, CEM, endpoint, seed)
    fresh_hash = parameter_hash(fresh)
    model = GateStructure(fresh, data.record['phi_dim'], data.num_types, 'V1', seed)
    ref = expected()['controls'][f'{CEM}_t{endpoint}_s{seed}']
    evidence = dict(optimizer_updates=0, reference_variant=CEM, reference_seed=seed,
        reference_kind='archived same-seed CEM plus unchanged fresh GateStructure V1',
        expected_CEM_initial_weights_sha256=ref['initial_weights_sha256'],
        actual_CEM_initial_weights_sha256=fresh_hash,
        actual_initial_weights_sha256=parameter_hash(model),
        parameters=sum(p.numel() for p in model.parameters()),
        learned_checkpoint_loaded=False, structural_variant=model.variant,
        retained_CEM_tensors_exact=model.info['retained_CEM_tensors_exact'],
        retained_CEM_buffers_exact=model.info['retained_CEM_buffers_exact'],
        retained_tensor_hashes={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                               for n,p in model.named_parameters()})
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']).parent/'V1_initialization_observation.json', evidence)
    assert fresh_hash == ref['initial_weights_sha256'] == model.info['reference_CEM_initial_weights_sha256']
    assert evidence['parameters'] == 12431071 and model.variant == 'V1'
    assert evidence['retained_CEM_tensors_exact'] and evidence['retained_CEM_buffers_exact']
    assert model.info['retained_S3_tensors_exact'] and model.info['retained_U1_tensors_exact']
    return model


class Optimizer(OriginalOptimizer):
    """Split AdamW groups; inherit the original update and recovery operations."""
    def __init__(self, model, spec, lr_multiplier=1.):
        assert lr_multiplier==1. and spec==protocol(spec['variant'], 'mimic', spec['seed'])
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
        value.update(protocol_search=False, prespecified_group_rate_only=True, fixed_structural_variant='V1',
            fusion_groups={g['role']:dict(lr=g['lr'],betas=g['betas'],weight_decay=g['weight_decay'],
                parameter_names=g['parameter_names'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params']))
                for g in self.fusion.param_groups})
        return value
