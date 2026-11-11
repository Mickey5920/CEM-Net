"""Locked CEM/TabM control inference on one idle GPU; no optimizer or fitting."""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--database', required=True); p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args(); project = a.project.resolve(strict=True); source = a.source.resolve(strict=True); out = a.output.resolve()
    assert source.is_relative_to(project) and out.is_relative_to(project) and not out.exists()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from scripts.cem_matched_common import configuration, Data, construct
    from scripts.tracehr_multiview_common import read, sha, atomic
    from scripts.train_tracehr_unified import seed_all
    from scripts.cem_evidence_host_queue import inventory
    from scripts.train_cem_matched_controls import predict
    import fcntl

    lock_path = project/'work/tracehr_fullstream_20261003/locks'/f'gpu{a.gpu}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observation = next(x for x in inventory(project) if x['index'] == a.gpu)
    assert not observation['processes'] and observation['free_mib'] >= 8192 and observation['utilization'] == 0
    out.mkdir(parents=True); c = configuration(); assert c['database'] == a.database
    host = project/'runs'/c['run']/('host_eicu' if a.database == 'eicu' else 'host_mimic')
    state = read(host/'state.json'); assert state['status'] == 'complete'
    assert read(host/'scheduler_v4_exit.json')['returncode'] == 0
    group = next(g for g in state['groups'].values() if g['database'] == a.database)
    assert group['source_manifest_sha256'] == sha(source/'source_manifest.json')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # Import only the already-used pure timing functions, without its old entry point.
    reference = Path(__file__).with_name('reference_inference.py')
    parsed = ast.parse(reference.read_text()); names = ('apply_transform', 'predict', 'benchmark')
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    api = dict(np=np, torch=torch, time=time)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), api)
    data = Data(project)
    raw = np.load(project/c['parent']['test_inputs']/'test_raw.npy', mmap_mode='r')
    five = project/'runs/cem_five_seed_20261102'/('host_eicu' if a.database == 'eicu' else 'host_mimic')
    old_group = next(g for g in read(five/'state.json')['groups'].values() if g['database'] == a.database)
    rows, protected = [], {}
    for endpoint in (0, 1):
        for seed in (48, 49, 50, 51, 52):
            for variant in ('reference', 'matched_tabm'):
                if variant == 'reference':
                    fit = Path(c['parent']['full_ema'][f'{seed}:{endpoint}']['output']) if seed <= 50 else Path(old_group['jobs'][f'CEM_t{endpoint}_s{seed}']['output'])
                else:
                    fit = Path(group['jobs'][f'matched_tabm_t{endpoint}_s{seed}']['output'])
                receipt = read(fit/'completion.json'); checkpoint = fit/receipt['checkpoint_file']
                assert receipt['complete'] and sha(checkpoint) == receipt['checkpoint_sha256']
                protected[str(checkpoint)] = receipt['checkpoint_sha256']
                seed_all(seed); model = construct(data, seed, endpoint, variant)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model']); del saved
                model.cuda().eval(); expected = dict(np.load(fit/'predictions_dev.npz'))
                observed = predict(model, data, indices=np.arange(32))
                np.testing.assert_array_equal(expected['ids'], data.targets['dev']['ids'])
                original = expected['probs'][:32]
                if original.ndim == 1: original = original[:, None]
                if original.shape[1] == 2: original = original[:, endpoint:endpoint+1]
                drift = float(np.max(np.abs(observed-original))); assert drift <= 1e-5
                before = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
                gpu_before = next(x for x in inventory(project) if x['index'] == a.gpu)
                assert all(p['pid'] == os.getpid() for p in gpu_before['processes']), 'Concurrent GPU process'
                # Both predictors expose the same logits interface; use the exact old timing path.
                timing = api['benchmark'](model, raw, 'CEM-Net', None, 'cuda')
                gpu_after = next(x for x in inventory(project) if x['index'] == a.gpu)
                assert all(p['pid'] == os.getpid() for p in gpu_after['processes']), 'Concurrent GPU process'
                assert all(torch.equal(before[n], value.detach().cpu()) for n, value in model.state_dict().items())
                rows.append(dict(variant=variant, endpoint=endpoint, seed=seed, checkpoint_sha256=receipt['checkpoint_sha256'],
                                 replay_max_drift=drift, timing=timing))
                atomic(out/'progress.json', dict(completed=len(rows), total=20, variant=variant, endpoint=endpoint, seed=seed))
                del model, before, expected; gc.collect(); torch.cuda.empty_cache()
    for path, expected in protected.items(): assert sha(path) == expected
    result = dict(complete=True, database=a.database, rows=rows, optimizer_updates=0,
                  original_checkpoints_unchanged=True, initial_gpu_observation=observation,
                  source_manifest_sha256=sha(source/'source_manifest.json'), timing_source_sha256=sha(reference),
                  benchmark_source_sha256=sha(Path(__file__)), checkpoints=protected,
                  protocol='FP32; four CPU threads; batches 1/8/32; 10 warmups and 30 repeats; same GPU and rows for both methods')
    atomic(out/'verified-results.json', result)
    atomic(out/'completion.json', dict(complete=True, files={'verified-results.json': sha(out/'verified-results.json')}))
    print(json.dumps(dict(complete=True, database=a.database, checkpoints=20)))


if __name__ == '__main__':
    main()
