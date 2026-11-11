"""Fixed five-structure, seed48 MIMIC comparison; all optimization settings reused."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.cem_structure_common import (Data, cem_model, CEM, Optimizer, stopping,
    atomic, read, sha, parameter_hash, protocol as s3_protocol, PROTOCOL_FIELDS as S3_FIELDS)
from models.cem_refine import RefineStructure, VARIANTS, DESCRIPTIONS

RUN = 'tracehr_cem_refine_20261023'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48,)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
PROTOCOL_FIELDS = S3_FIELDS | {'event_direct_enabled', 'interaction_event_skip_enabled',
    'event_direct_gate', 'lab_observation_bias', 'interaction_direction',
    'event_slot_self_block', 'member_query_conditioning'}


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return tuple((v, e, 48) for v in VARIANTS for e in (0, 1))


def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = s3_protocol('S3', database, seed)
    spec.update(variant=variant, structure=variant, ablated_component=None,
        description=DESCRIPTIONS[variant], ablation_implementation='one prespecified structural change',
        event_direct_enabled=variant == 'U1', interaction_event_skip_enabled=variant != 'U3',
        event_direct_gate='sigmoid_linear_192_to_1' if variant == 'U1' else None,
        lab_observation_bias='scaled_simple_fields_1_to_4_linear4_to1_no_bias' if variant == 'U2' else None,
        interaction_direction='labs_query_events' if variant == 'U3' else 'events_query_labs',
        event_slot_self_block=variant != 'U4',
        member_query_conditioning='LN_current_backbone_hidden_linear256_to96_no_bias' if variant == 'U5' else None)
    return spec


def model_for(data, preparation, variant, endpoint, seed):
    fresh = cem_model(data, preparation, CEM, endpoint, seed)
    model = RefineStructure(fresh, data.record['phi_dim'], data.num_types, variant, seed)
    ref = expected()['controls'][f'S3_t{endpoint}_s{seed}']
    observed = read(Path(ref['source'])/'initialization_observation.json')['retained_tensor_hashes']
    actual = {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for n, p in model.named_parameters()}
    audit = dict(optimizer_updates=0, reference_S3_initial_weights_sha256=ref['initial_weights_sha256'],
        reconstructed_S3_initial_weights_sha256=model.info['reference_S3_initial_weights_sha256'],
        tensor_mismatches=[n for n in actual.keys()&observed.keys() if observed[n] != actual[n]],
        removed_names=sorted(set(observed)-set(actual)), new_names=sorted(set(actual)-set(observed)))
    import os
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']), audit)
    assert audit['reference_S3_initial_weights_sha256'] == audit['reconstructed_S3_initial_weights_sha256']
    assert not audit['tensor_mismatches'], audit
    assert audit['removed_names'] == model.info['removed_S3_parameter_names']
    assert audit['new_names'] == model.info['new_S3_parameter_names']
    assert sum(p.numel() for p in model.parameters()) == ref['parameters']-model.info['removed_S3_parameters']+model.info['new_S3_parameters']
    model.info['archived_S3_retained_initialization_verified'] = True
    return model


@torch.no_grad()
def diagnostics(model, data, initial, seed=48):
    return dict(fusion_residual_scale=float(model.fusion.residual_scale) if model.fusion.event_direct_enabled else None,
        interaction_residual_scale=float(model.fusion.extra_scale),
        gate_summary=getattr(model.fusion, 'last_gate_summary', None),
        parameter_relative_L2_displacement={n: float((p-initial[n]).norm()/(initial[n].norm()+1e-12))
                                            for n, p in model.named_parameters()},
        variant=model.variant, all_parameters_jointly_trainable=all(p.requires_grad for p in model.parameters()))
