"""P/M deletions inside the fixed EMA method; original training recipe preserved."""
import hashlib
from pathlib import Path
import torch
from scripts.mimiciii_ema_common import EARLY, EMA_RULE, budget, tensor_hashes
from scripts.mimiciii_pm_mini_common import protocol as deletion_protocol
from scripts.mimiciii_fixed_common import Data, model_for, parameter_hash
from scripts.train_mimiciii_numeric_baseline import NoEvidence
from scripts.tracehr_multiview_common import read, sha, atomic
from models.mimiciii_pm_mini import delete_component

ROOT = Path(__file__).resolve().parents[1]
RUN = 'mimiciii_ema_pm_20261024'
VARIANTS = ('noEvidence_EMA_noP', 'noEvidence_EMA_noM')
SEEDS = (48, 49, 50)


def expected(): return read(ROOT/'work'/RUN/'expected.json')
def raw_variant(variant):
    assert variant in VARIANTS
    return variant.replace('_EMA', '')
def protocol(variant, seed): return deletion_protocol(raw_variant(variant), seed)
def matrix():
    return [dict(variant=v, seed=s, endpoint=e) for s in SEEDS for v in VARIANTS for e in (0, 1)]
def reference(variant, seed, endpoint):
    return expected()['raw_deletions'][f'{raw_variant(variant)}_t{endpoint}_s{seed}']


def fresh_model(data, preparation, seed, endpoint, out, variant):
    full = model_for(data, preparation, endpoint, seed)
    model = NoEvidence(full)
    del full
    ref = expected()['references'][f'noEvidence_s{seed}_t{endpoint}']
    obs_path = Path(ref['output'])/'initialization_observation.json'
    full_hashes = tensor_hashes(model)
    initial = parameter_hash(model)
    atomic(out/'initialization_observation.json', dict(stage='before_full_assertion', optimizer_updates=0,
           initial_weights_sha256=initial, full_retained_tensor_hashes=full_hashes))
    assert sha(obs_path) == ref['initialization_sha256']
    assert full_hashes == read(obs_path)['retained_tensor_hashes']
    assert initial == ref['initial_weights_sha256']
    old = torch.load(Path(ref['output'])/ref.get('checkpoint_file', 'best.pt'), map_location='cpu', weights_only=False)
    assert all(torch.equal(b.detach().cpu(), old['model'][n]) for n, b in model.named_buffers())
    del old
    model, mapping = delete_component(model, variant[-1])
    retained = tensor_hashes(model)
    r = reference(variant, seed, endpoint)
    observation = dict(complete=True, optimizer_updates=0, reference_main_initial_sha256=initial,
        initial_weights_sha256=parameter_hash(model), retained_tensor_hashes=retained, deletion=mapping,
        reference_seed=seed, full_initial_tensors_exact=True, original_static_buffers_exact=True,
        archived_deletion_initial_sha256=r['initial_weights_sha256'])
    atomic(out/'initialization_observation.json', observation)
    prior_obs = Path(r['output'])/'initialization_observation.json'
    assert sha(prior_obs) == r['initialization_sha256']
    assert retained == read(prior_obs)['retained_tensor_hashes']
    assert parameter_hash(model) == r['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters()) == r['parameters']
    assert all(p.requires_grad for p in model.parameters()) and model.members == 8
    assert not model.main_residual_enabled and not model.auxiliary_enabled
    audit = dict(complete=True, reference_seed=seed, retained_CEM_tensors_exact=True,
        archived_deletion_initials_exact=True, learned_checkpoint_loaded=False,
        reference_checkpoint_read_for_buffer_comparison_only=True, original_static_buffers_exact=True,
        P_retained=variant.endswith('M'), M_retained=variant.endswith('P'), deletion=mapping,
        members=8, main_loss_coefficient=.5, evidence_branch_present=False, auxiliary_present=False,
        preprocessing_unchanged=True, all_retained_parameters_trainable=True, test_loaded=False)
    atomic(out/'connection_audit.json', audit)
    return model
