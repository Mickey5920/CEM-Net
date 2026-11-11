"""Exactly two X1 fits; preserve the U1 optimization recipe and original inputs."""
import hashlib, os
from pathlib import Path
from scripts.cem_refine_common import (Data, cem_model, CEM, Optimizer, stopping, atomic, read, sha,
    parameter_hash, diagnostics as parent_diagnostics, protocol as u1_protocol,
    PROTOCOL_FIELDS as U1_FIELDS, np, torch)
from models.cem_gate_start import StartGateStructure, VARIANTS, DESCRIPTIONS

RUN = 'tracehr_cem_gate_start_20261023'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48,)
PREPARATION = 'runs/tracehr_cem_structure_20261023/mimic/structure_v1/preparation'
PROTOCOL_FIELDS = U1_FIELDS | {'event_gate_alpha_initial', 'event_gate_alpha_trainable'}


def expected():
    return read(ROOT/'work'/RUN/'expected.json')


def matrix(database):
    assert database == 'mimic'
    return (('X1', 0, 48), ('X1', 1, 48))


def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = u1_protocol('U1', database, seed)
    spec.update(variant=variant, structure=variant, description=DESCRIPTIONS[variant],
        event_direct_gate='sigmoid_bias_plus_learned_alpha_times_linear192_no_bias',
        event_gate_alpha_initial=0., event_gate_alpha_trainable=True)
    return spec


def model_for(data, preparation, variant, endpoint, seed):
    fresh = cem_model(data, preparation, CEM, endpoint, seed)
    model = StartGateStructure(fresh, data.record['phi_dim'], data.num_types, variant, seed)
    actual = {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for n,p in model.named_parameters()}
    audits = {}
    for parent in ('S3', 'U1', 'V1'):
        ref = expected()['controls'][f'{parent}_t{endpoint}_s{seed}']
        observed = read(Path(ref['source'])/'initialization_observation.json')['retained_tensor_hashes']
        audit = dict(optimizer_updates=0, reference_initial_weights_sha256=ref['initial_weights_sha256'],
            tensor_mismatches=[n for n in actual.keys()&observed.keys() if actual[n]!=observed[n]],
            removed_names=sorted(set(observed)-set(actual)), new_names=sorted(set(actual)-set(observed)))
        if parent in ('S3','U1'):
            audit.update({f'reference_{parent}_initial_weights_sha256':ref['initial_weights_sha256'],
                f'reconstructed_{parent}_initial_weights_sha256':model.info[f'reference_{parent}_initial_weights_sha256']})
        audits[parent] = audit
    if os.environ.get('CEM_INITIALIZATION_OBSERVER'):
        atomic(Path(os.environ['CEM_INITIALIZATION_OBSERVER']), dict(audits['S3'], U1=audits['U1'], V1=audits['V1']))
    for parent,audit in audits.items():
        assert not audit['tensor_mismatches'] and not audit['removed_names'], audit
        ref = expected()['controls'][f'{parent}_t{endpoint}_s{seed}']
        if parent in ('S3','U1'):
            assert audit[f'reference_{parent}_initial_weights_sha256']==audit[f'reconstructed_{parent}_initial_weights_sha256']
            assert audit['new_names']==model.info[f'new_{parent}_parameter_names']
            assert sum(p.numel() for p in model.parameters())==ref['parameters']+model.info[f'new_{parent}_parameters']
        else:
            assert audit['new_names']==['fusion.gate.alpha', 'fusion.gate.weight']
            assert sum(p.numel() for p in model.parameters())==ref['parameters']+193
        model.info[f'archived_{parent}_retained_initialization_verified'] = True
    return model


@torch.no_grad()
def diagnostics(model, data, initial, seed=48):
    result = parent_diagnostics(model, data, initial, seed)
    gate = model.fusion.gate
    result['gate_alpha'] = float(gate.alpha)
    result['gate_alpha_trainable'] = gate.alpha.requires_grad
    result['gate_conditional_weight_norm'] = float(gate.weight.norm())
    result['gate_effective_weight_norm'] = float((gate.alpha*gate.weight).norm())
    return result
