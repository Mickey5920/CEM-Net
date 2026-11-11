"""Fresh original-network training with EMA, followed by locked zero-update tests."""
import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path[:0] = [str(deps), *[str(p) for p in deps.glob('*.whl')]]
import numpy as np
import torch
from scripts.mimiciii_ema_pm_common import *
from scripts import mimiciii_ema_pm_runtime as runtime
from scripts.train_tracehr_unified import seed_all


def evaluate(a, out, data):
    from scripts.tracehr_oof_common import all_scores
    job = a.fit_output.resolve(strict=True)
    c = read(job/'completion.json'); identity = read(job/'run_manifest.json')
    locked = read(job.parent.parent/'fixed-checkpoint-lock.json')
    assert locked['complete'] and len(locked['selections']) == 12 and not locked['test_used_for_selection']
    assert c['complete'] and not c['pilot'] and c['parameter_source'] == 'EMA'
    assert c['source_manifest_sha256'] == sha(ROOT/'source_manifest.json')
    ckpath = job/'best.pt'
    assert sha(ckpath) == c['checkpoint_sha256'] == locked['selections'][job.name]['checkpoint_sha256']
    model = fresh_model(data, a.preparation, a.seed, a.endpoint, out, a.variant).cuda().eval()
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    assert ck['role'] == 'EMA_selected_inference' and ck['identity'] == identity
    model.load_state_dict(ck['model'], strict=True); del ck; gc.collect()
    before = runtime.digest(model.state_dict()); torch.set_grad_enabled(False)
    def forbidden(*args, **kwargs): raise AssertionError('Test evaluation forbids weight updates')
    torch.Tensor.backward = forbidden; torch.autograd.backward = forbidden
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD): cls.step = forbidden
    saved = dict(np.load(job/'predictions_dev.npz'))
    q = runtime.predict(model, data, 256)
    drift = float(np.max(np.abs(q-saved['probs']))); assert drift <= 1e-5
    np.testing.assert_array_equal(saved['ids'], data.targets['dev']['ids'])
    np.testing.assert_array_equal(saved['labels'][:,0], data.targets['dev']['y'][:,a.endpoint])
    inp = a.project/expected()['test_inputs']
    for n in ('test_raw.npy', 'test_targets.npz'):
        assert sha(inp/n) == expected()['test_manifest'][n]['sha256']
    raw = np.load(inp/'test_raw.npy', mmap_mode='r'); target = dict(np.load(inp/'test_targets.npz'))
    for split in ('train', 'dev'):
        assert not set(target['ids'][:,0]) & set(data.targets[split]['ids'][:,0])
    values = []
    for start in range(0, len(raw), 256):
        x = torch.as_tensor(np.array(raw[start:start+256], copy=True), device='cuda')
        values.append(model.base(x).softmax(-1)[...,1].mean(1).cpu().numpy())
    q = np.concatenate(values)[:,None]; labels = target['y'][:,a.endpoint:a.endpoint+1]
    metrics = all_scores(labels, q)
    np.savez_compressed(out/'predictions_test.npz', probs=q, labels=labels, ids=target['ids'])
    assert runtime.digest(model.state_dict()) == before and sha(ckpath) == c['checkpoint_sha256']
    atomic(out/'completion.json', dict(complete=True, **identity, split='test', optimizer_updates=0,
           parameters_updated=False, model_sha256=c['checkpoint_sha256'], parameter_source='EMA',
           predictions_sha256=sha(out/'predictions_test.npz'), metrics=metrics, dev_replay_max_drift=drift,
           best_epoch=c['best_epoch'], selected_using='EMA z0 development AP only'))
    atomic(out/'progress.json', dict(status='complete', test_ap=metrics['mean_ap']))


def main():
    p = argparse.ArgumentParser()
    for n in ('project', 'output', 'pilot-record', 'fit-output'):
        p.add_argument('--'+n, type=Path, required=n in ('project', 'output'))
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--endpoint', type=int, choices=[0, 1], required=True)
    p.add_argument('--action', choices=['probe', 'train', 'evaluate'], required=True)
    p.add_argument('--expected-gib', type=float, required=True)
    a = p.parse_args(); a.project = a.project.resolve(strict=True)
    a.preparation = a.project/expected()['input_preparation']
    out = a.output.resolve(); assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free > (a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for k in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        value = Path(os.environ[k]).resolve(); assert value.is_relative_to(a.project)
        value.mkdir(parents=True, exist_ok=True)
    atomic(out/'storage.json', dict(project=str(a.project), output=str(out), preparation=str(a.preparation),
           expected_gib=a.expected_gib, code=str(ROOT), cache=os.environ['XDG_CACHE_HOME'], temp=os.environ['TMPDIR']))
    torch.set_num_threads(4); assert torch.cuda.is_available(); seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json', dict(status='loading_verified_inputs')); data = Data(a.preparation)
    if a.action == 'evaluate': evaluate(a, out, data); return
    atomic(out/'progress.json', dict(status='randomly_initializing_native'))
    model = fresh_model(data, a.preparation, a.seed, a.endpoint, out, a.variant)
    identity = dict(database='mimiciii', variant=a.variant, endpoint=a.endpoint, seed=a.seed, input_dim=data.input_dim,
                    source_manifest_sha256=sha(ROOT/'source_manifest.json'), input_manifest_sha256=sha(a.preparation/'completion.json'),
                    protocol=protocol(a.variant, a.seed), ema_rule=EMA_RULE, epoch_budget=budget(a.seed, a.endpoint),
                    initial_weights_sha256=parameter_hash(model), parameters=sum(p.numel() for p in model.parameters()),
                    members=8, lr_multiplier=1., selected_structure=raw_variant(a.variant))
    atomic(out/'run_manifest.json', identity); model.cuda(); gc.collect()
    from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
    atomic(out/'G0_equivalence.json', initial_equivalence(model, data))
    (runtime.probe if a.action == 'probe' else runtime.train)(a, out, data, model, identity, identity['protocol'])


if __name__ == '__main__': main()
