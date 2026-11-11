"""Frozen original preprocessing/P/M recipe with one parameter-averaging rule."""
import hashlib
from pathlib import Path
import torch
from scripts.mimiciii_baselines_common import numeric_protocol
from scripts.mimiciii_fixed_common import Data, model_for, parameter_hash
from scripts.train_mimiciii_numeric_baseline import NoEvidence
from scripts.tracehr_multiview_common import read, sha, atomic

ROOT = Path(__file__).resolve().parents[1]
RUN = 'mimiciii_ema_20261024'
VARIANT = 'noEvidence_EMA'
SEEDS = (48, 49, 50)
EARLY = (1, 5, 8, 10, 12, 15, 18, 20)
EMA_RULE = dict(decay=.999, warmup='min(0.999, (t+1)/(t+10))',
                t='zero-based optimizer update index', initialization='theta_0',
                update_frequency='after each effective optimizer step',
                buffers='unchanged original static buffers', selection='EMA z0 development AP')


def expected(): return read(ROOT/'work'/RUN/'expected.json')
def matrix(): return [dict(variant=VARIANT, seed=s, endpoint=e) for s in SEEDS for e in (0, 1)]
def protocol(variant, seed):
    assert variant == VARIANT and seed in SEEDS
    # Exact original optimizer input: EMA is deliberately outside this recipe.
    return numeric_protocol('noEvidence', seed)


def budget(seed, endpoint):
    return 120 if endpoint == 0 or seed == 49 else 52


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters()}


def fresh_model(data, preparation, seed, endpoint, out):
    full = model_for(data, preparation, endpoint, seed)
    model = NoEvidence(full)
    del full
    ref = expected()['references'][f'noEvidence_s{seed}_t{endpoint}']
    path = Path(ref['output'])/'initialization_observation.json'
    assert sha(path) == ref['initialization_sha256']
    hashes = tensor_hashes(model)
    assert hashes == read(path)['retained_tensor_hashes']
    assert parameter_hash(model) == ref['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters()) == 12902328
    assert all(p.requires_grad for p in model.parameters())
    assert not model.main_residual_enabled and not model.auxiliary_enabled and model.members == 8
    # The original buffers are static preprocessing; validate rather than alter them.
    old = torch.load(Path(ref['output'])/ref.get('checkpoint_file', 'best.pt'), map_location='cpu', weights_only=False)
    assert all(torch.equal(b.detach().cpu(), old['model'][n]) for n, b in model.named_buffers())
    del old
    audit = dict(complete=True, optimizer_updates=0, retained_tensor_hashes=hashes,
                 retained_CEM_tensors_exact=True, original_static_buffers_exact=True,
                 reference_seed=seed, initial_weights_sha256=parameter_hash(model),
                 learned_checkpoint_loaded=False, reference_checkpoint_read_for_buffer_comparison_only=True,
                 P_retained=True, M_retained=True, members=8, main_loss_coefficient=.5,
                 evidence_branch_present=False, auxiliary_present=False,
                 preprocessing_unchanged=True, all_retained_parameters_trainable=True,
                 test_loaded=False)
    atomic(out/'initialization_observation.json', audit)
    atomic(out/'connection_audit.json', audit)
    return model
