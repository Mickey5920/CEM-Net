"""Train the previously accepted model on III, without changes to its loop."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path.insert(0, str(deps))
for wheel in deps.glob('*.whl'):
    sys.path.insert(0, str(wheel))
import torch
from scripts.mimiciii_fixed_common import *
from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
from scripts.train_group_v1_validation import longest_evidence_stress
from scripts import cem_v1_group_confirm_runtime as runner

runner.Optimizer = Optimizer
runner.diagnostics = diagnostics
runner.stopping = stopping


def rates(model, spec):
    opt = Optimizer(model, spec)
    rows = []
    for epoch in (0., .5, 1., 10., 52., 120., 200., 255.):
        opt.schedule(epoch)
        groups = opt.values()['fusion_groups']
        # The unchanged native warmup has zero LR at epoch 0. Multiplication
        # verifies the same exact ratio without dividing by that valid zero.
        assert groups['readout']['lr'] == .5 * groups['encoder']['lr']
        assert set(groups['readout']['parameter_names']) == set(spec['fusion_readout_parameter_names'])
        rows.append(dict(epoch=epoch, groups=groups))
    assert not opt.fusion.state
    return dict(complete=True, optimizer_updates=0, schedules=rows,
                optimizer_class='unchanged group_v1_validation_common.Optimizer',
                encoder_lr=spec['fusion_lr'], readout_lr=spec['fusion_readout_lr'])


def main():
    p = argparse.ArgumentParser()
    for name in ('project', 'preparation', 'output', 'pilot-record'):
        p.add_argument('--' + name, type=Path, required=name != 'pilot-record')
    p.add_argument('--action', choices=['probe', 'train'], required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--endpoint', type=int, choices=[0, 1], required=True)
    p.add_argument('--expected-gib', type=float, required=True)
    a = p.parse_args()
    a.project = a.project.resolve(strict=True)
    a.resume = False
    a.lr_multiplier = 1.
    a.start_barrier = None
    out = a.output.resolve()
    assert out.is_relative_to(a.project) and out != a.project and not out.exists()
    assert a.preparation.resolve().is_relative_to(a.project)
    assert shutil.disk_usage(a.project).free > (a.expected_gib + 10) * 2**30
    out.mkdir(parents=True)
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        value = Path(os.environ[key]).resolve()
        assert value.is_relative_to(a.project)
        value.mkdir(parents=True, exist_ok=True)
    atomic(out / 'storage.json', dict(project=str(a.project), code=str(ROOT), output=str(out),
        preparation=str(a.preparation), cache=os.environ['XDG_CACHE_HOME'], temp=os.environ['TMPDIR'], expected_gib=a.expected_gib))
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    atomic(out / 'progress.json', dict(status='loading_verified_inputs'))
    data = Data(a.preparation)
    spec = protocol(VARIANT, 'mimiciii', a.seed)
    atomic(out / 'progress.json', dict(status='randomly_initializing_native'))
    os.environ['CEM_INITIALIZATION_OBSERVER'] = str(out / 'S3_initialization_observation.json')
    model = model_for(data, a.preparation, a.endpoint, a.seed)
    info = model.info
    identity = dict(database='mimiciii', variant=VARIANT, endpoint=a.endpoint, seed=a.seed,
        lr_multiplier=1., protocol=spec, source_manifest_sha256=sha(ROOT / 'source_manifest.json'),
        input_manifest_sha256=sha(a.preparation / 'completion.json'), selected_structure=VARIANT,
        initial_weights_sha256=parameter_hash(model), common_weights_sha256=parameter_hash(model, True),
        parameters=sum(p.numel() for p in model.parameters()), input_dim=data.input_dim, members=model.members)
    atomic(out / 'initialization_observation.json', dict(optimizer_updates=0, reference_seed=a.seed,
        reference_kind='fresh same-seed backbone on MIMIC-III; no archived trained model',
        actual_reference_CEM_hash=info['reference_CEM_initial_weights_sha256'],
        initial_weights_sha256=identity['initial_weights_sha256'],
        retained_tensor_hashes={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}))
    assert model.members == 8 and model.main_residual_enabled and not model.auxiliary_enabled
    assert all(p.requires_grad for p in model.parameters())
    edge = dict(complete=True, reference_seed=a.seed,
        reference_CEM_random_initial_hash=info['reference_CEM_initial_weights_sha256'],
        retained_CEM_tensors_exact=info['retained_CEM_tensors_exact'], learned_checkpoint_loaded=False,
        stop_gradient=False, all_parameters_trainable=True, members=8, main_view='z0',
        checkpoint_selection_view='z0', stopping_view='z0', auxiliary_enabled=False,
        main_loss_coefficient=.5, auxiliary_loss_coefficient=0., main_residual_enabled=True,
        fusion_position=info['fusion_position'], fresh_initialization_reference=True)
    atomic(out / 'connection_audit.json', edge)
    atomic(out / 'run_manifest.json', identity)
    atomic(out / 'native_initialization.json', info)
    model = model.cuda()
    gc.collect()
    atomic(out / 'G0_equivalence.json', initial_equivalence(model, data))
    atomic(out / 'execution_audit.json', dict(from_scratch=True, test_loaded=False,
        training_sampler='unchanged NumPy default_rng permutation; retain tail',
        effective_batch=256, microbatch=128, precision='FP32', hyperparameter_search=False,
        optimizer_clock_reference_epochs=256, stopping_rule='unchanged MIMIC 52/120 boundary rule; cap 256',
        input_adapter_only=True, frozen_parent_manifest=expected()['parent_manifest_sha256']))
    if a.action == 'probe':
        rate = rates(model, spec)
        atomic(out / 'rate_check.json', rate)
        stress = longest_evidence_stress(model, data, spec, a.endpoint)
        assert parameter_hash(model) == identity['initial_weights_sha256']
        runner.probe(a, out, data, model, identity, spec)
        r = read(out / 'completion.json')
        r.update(longest_evidence_stress=stress, rate_check=rate,
                 peak_reserved_mib=max(r['peak_reserved_mib'], stress['peak_reserved_mib']))
        atomic(out / 'completion.json', r)
    else:
        assert a.pilot_record
        runner.train(a, out, data, model, identity, spec)


if __name__ == '__main__':
    main()
