"""Exactly one untied control, six fits per database, existing seeds and recipe."""
from pathlib import Path
import numpy as np
import torch
from scripts.tracehr_noa_validation_common import (
    Data, stopping, protocol as prior_protocol, PROTOCOL_FIELDS,
    Optimizer, diagnostics as previous_diagnostics, parameter_hash, read,
    model_for as tied_model)
from models.tracehr_fte_binding import PM, VARIANT, VARIANTS, untie_fresh_pm, coordinate_divergence

RUN = 'tracehr_fte_binding_20261022'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48, 49, 50)
BASELINES = ('B0', 'tabm', 'xgboost')  # archived controls only; no baseline fitting

def expected(): return read(ROOT/'work'/RUN/'expected.json')

def matrix(database):
    assert database in ('mimic', 'eicu')
    return tuple((VARIANT, e, s) for s in SEEDS for e in (0, 1))

def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = prior_protocol(PM if database == 'mimic' else 'C5_noT_noA', database, seed)
    spec.update(variant=variant, structure=variant, ablated_component='feature_tying',
        description='independent expanded-coordinate factors; equal fresh initial function; fixed PM recipe',
        ablation_implementation='expand fresh K*d factors through feature_map to K*(d+3q) independent parameters')
    return spec

def model_for(data, preparation, variant, endpoint, seed):
    assert variant == VARIANT and endpoint in (0, 1) and seed in SEEDS
    model = tied_model(data, preparation, PM, endpoint, seed)
    was_training = model.training
    # A read-only train-batch check; no optimizer or saved learned weights.
    with torch.random.fork_rng(devices=[]):
        model.eval()
        batch, labels = data.batch('train', np.arange(16), device='cpu')
        before = model(batch)
        tied_grad = torch.autograd.grad(model.loss((before, None), labels[:, endpoint]), model.base.input_members)[0]
        before = before.detach()
        model = untie_fresh_pm(model)
        after = model(batch)
        untied_grad = torch.autograd.grad(model.loss((after, None), labels[:, endpoint]), model.base.input_members)[0]
        folded = torch.zeros_like(tied_grad)
        folded.scatter_add_(1, model.base.feature_map[None].expand(8, -1), untied_grad)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(tied_grad, folded, rtol=1e-5, atol=1e-7)
        model.info['binding_initial_equivalence'] = dict(complete=True,optimizer_updates=0,fit_split='train',
            max_logit_drift=float((before-after).abs().max()),
            gradient_sum_max_abs_drift=float((tied_grad-folded).abs().max()),
            initial_coordinate_divergence=coordinate_divergence(model))
        model.train(was_training)
    return model

@torch.no_grad()
def diagnostics(model, data, initial, seed=48):
    value = previous_diagnostics(model, data, initial, seed)
    value['untied_coordinate_divergence'] = coordinate_divergence(model)
    return value
