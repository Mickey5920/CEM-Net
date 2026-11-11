"""Exactly V1/V2, seed48 and two MIMIC endpoints; U1 optimization unchanged."""
import hashlib, os
from pathlib import Path
from scripts.cem_refine_common import (Data, cem_model, CEM, Optimizer, stopping, atomic, read, sha,
    parameter_hash, diagnostics, protocol as u1_protocol, PROTOCOL_FIELDS as U1_FIELDS, np, torch)
from models.cem_gate_location import GateStructure, VARIANTS, DESCRIPTIONS

RUN = 'tracehr_cem_gate_20261023'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48,)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
PROTOCOL_FIELDS = U1_FIELDS | {'event_gate_backbone_condition'}


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return tuple((v,e,48) for v in VARIANTS for e in (0,1))


def protocol(variant, database, seed):
    assert (variant,0,seed) in matrix(database)
    spec = u1_protocol('U1',database,seed)
    spec.update(variant=variant,structure=variant,description=DESCRIPTIONS[variant],
        event_direct_gate='sigmoid_scalar_fresh_U1_bias' if variant=='V1' else 'sigmoid_linear_192_to_1_plus_LN_h_linear256_to1_no_bias',
        event_gate_backbone_condition='current_member_pre_residual_hidden' if variant=='V2' else None)
    return spec


def model_for(data, preparation, variant, endpoint, seed):
    fresh=cem_model(data,preparation,CEM,endpoint,seed)
    model=GateStructure(fresh,data.record['phi_dim'],data.num_types,variant,seed)
    actual={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters()}
    audits={}
    for parent in ('S3','U1'):
        ref=expected()['controls'][f'{parent}_t{endpoint}_s{seed}']
        observed=read(Path(ref['source'])/'initialization_observation.json')['retained_tensor_hashes']
        audit=dict(optimizer_updates=0,
            **{f'reference_{parent}_initial_weights_sha256':ref['initial_weights_sha256'],
               f'reconstructed_{parent}_initial_weights_sha256':model.info[f'reference_{parent}_initial_weights_sha256']},
            tensor_mismatches=[n for n in actual.keys()&observed.keys() if actual[n]!=observed[n]],
            removed_names=sorted(set(observed)-set(actual)),new_names=sorted(set(actual)-set(observed)))
        audits[parent]=audit
    # Save evidence before any assertion or optimizer update, including failures.
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']),dict(audits['S3'],U1=audits['U1']))
    for parent,audit in audits.items():
        assert audit[f'reference_{parent}_initial_weights_sha256']==audit[f'reconstructed_{parent}_initial_weights_sha256']
        assert not audit['tensor_mismatches'],audit
        assert audit['removed_names']==model.info[f'removed_{parent}_parameter_names']
        assert audit['new_names']==model.info[f'new_{parent}_parameter_names']
        ref=expected()['controls'][f'{parent}_t{endpoint}_s{seed}']
        assert sum(p.numel() for p in model.parameters())==ref['parameters']-model.info[f'removed_{parent}_parameters']+model.info[f'new_{parent}_parameters']
        model.info[f'archived_{parent}_retained_initialization_verified']=True
    return model
