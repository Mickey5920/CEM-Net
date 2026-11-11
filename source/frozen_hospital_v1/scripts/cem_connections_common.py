"""Fixed seed48, three connection deletions, two MIMIC endpoints."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.cem_structure_common import (Data, cem_model, CEM, Optimizer, stopping,
    atomic, read, sha, parameter_hash, protocol as s3_protocol, PROTOCOL_FIELDS as S3_FIELDS)
from models.cem_connection_ablation import ConnectionStructure, VARIANTS

RUN = 'tracehr_cem_connections_20261023'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48,)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
PROTOCOL_FIELDS = S3_FIELDS | {'event_direct_enabled', 'interaction_event_skip_enabled'}


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return tuple((v, e, 48) for v in VARIANTS for e in (0, 1))


def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = s3_protocol('S3', database, seed)
    spec.update(variant=variant, structure=variant, ablated_component=variant,
        description='S3 connection deletion: '+variant,
        ablation_implementation='delete specified event residual connections; preserve all other S3 computation',
        event_direct_enabled=variant == 'S3_noSkip',
        interaction_event_skip_enabled=variant == 'S3_noDirect')
    return spec


def model_for(data, preparation, variant, endpoint, seed):
    fresh = cem_model(data, preparation, CEM, endpoint, seed)
    model = ConnectionStructure(fresh, data.record['phi_dim'], data.num_types, variant, seed)
    ref = expected()['controls'][f'S3_t{endpoint}_s{seed}']
    observed = read(Path(ref['source'])/'initialization_observation.json')['retained_tensor_hashes']
    actual = {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for n, p in model.named_parameters()}
    audit = dict(optimizer_updates=0, reference_S3_initial_weights_sha256=ref['initial_weights_sha256'],
        reconstructed_S3_initial_weights_sha256=model.info['reference_S3_initial_weights_sha256'],
        tensor_mismatches=[n for n, h in actual.items() if observed.get(n) != h],
        removed_names=sorted(set(observed)-set(actual)), new_names=sorted(set(actual)-set(observed)))
    # Save evidence before asserting, without changing any RNG or parameter.
    import os
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']), audit)
    assert audit['reference_S3_initial_weights_sha256'] == audit['reconstructed_S3_initial_weights_sha256']
    assert not audit['tensor_mismatches'] and not audit['new_names'], audit
    assert audit['removed_names'] == model.info['removed_S3_parameter_names']
    assert sum(p.numel() for p in model.parameters()) == ref['parameters']-model.info['removed_S3_parameters']
    model.info['archived_S3_retained_initialization_verified'] = True
    return model


@torch.no_grad()
def diagnostics(model, data, initial, seed=48):
    return dict(fusion_residual_scale=float(model.fusion.residual_scale) if model.fusion.event_direct_enabled else None,
        interaction_residual_scale=float(model.fusion.extra_scale),
        event_direct_enabled=model.fusion.event_direct_enabled,
        interaction_event_skip_enabled=model.fusion.interaction_event_skip_enabled,
        parameter_relative_L2_displacement={n: float((p-initial[n]).norm()/(initial[n].norm()+1e-12))
                                            for n, p in model.named_parameters()},
        variant=model.variant, all_parameters_jointly_trainable=all(p.requires_grad for p in model.parameters()))
