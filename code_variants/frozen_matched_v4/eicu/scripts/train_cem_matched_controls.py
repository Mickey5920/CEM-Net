"""Fixed-budget matched controls; reuse original optimizer, EMA and checkpoint format."""
import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.cem_matched_common import Data, Optimizer, configuration, construct, budget, protocol, VARIANTS, SEEDS, RUN
from scripts.tracehr_multiview_common import read, atomic, sha
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng
from scripts.mimiciii_ema_runtime import ParameterEMA, digest, save_checkpoint, EMA_RULE
from scripts.tracehr_oof_common import all_scores
from models.tracehr_unified_next import parameter_hash


def update(model, opt, data, indices, endpoint, check=False):
    model.train(); opt.zero_grad(); loss_sum = torch.zeros((), device='cuda')
    for start in range(0, len(indices), 128):
        part = indices[start:start+128]; x, y = data.batch('train', part)
        loss = model.loss(model(x), y[:, endpoint])
        assert torch.isfinite(loss)
        (loss*(len(part)/len(indices))).backward()
        loss_sum += loss.detach()*(len(part)/len(indices))
    before = {n: p.detach().clone() for n, p in model.named_parameters()} if check else None
    if check:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    report = None
    if check:
        report = dict(relative_update={n: float((p-before[n]).norm()/(before[n].norm()+1e-12)) for n, p in model.named_parameters()},
                      gradient_norm={n: float(p.grad.norm()) for n, p in model.named_parameters()})
        assert all(torch.isfinite(p).all() for p in model.parameters())
    return loss_sum, report


@torch.no_grad()
def predict(model, data, split='dev', indices=None, micro=256):
    model.eval(); indices = np.arange(len(data.targets[split]['ids'])) if indices is None else indices
    result = []
    for start in range(0, len(indices), micro):
        x, _ = data.batch(split, indices[start:start+micro])
        result.append(model(x).softmax(-1)[..., 1].mean(1, keepdim=True).cpu().numpy())
    p = np.concatenate(result)
    assert p.shape == (len(indices), 1) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    return p


def state_digest(model, opt, ema, rng):
    return digest(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.state_dict(), rng=rng_state(rng)))


def probe(a, out, data, model, identity, spec):
    began = time.time(); seed_all(a.seed+10000); rng = np.random.default_rng(a.seed)
    opt = Optimizer(model, spec); ema = ParameterEMA(model)
    order = rng.permutation(len(data.raw['train'])); steps = math.ceil(len(order)/256)
    timings = []; expected = None
    for i in range(20):
        opt.schedule(i/steps); torch.cuda.synchronize(); tick = time.perf_counter()
        loss, report = update(model, opt, data, order[i*256:(i+1)*256], a.endpoint, check=i in (0, 19))
        ema.step(model); torch.cuda.synchronize(); timings.append(time.perf_counter()-tick)
        if report is not None:
            atomic(out/('first_step_gradient_check.json' if i == 0 else 'gradient_check.json'), report)
        if i == 9:
            save_checkpoint(out/'resume_probe.pt', model, opt, ema, rng, dict(updates=10), identity)
    assert report and all(v > 0 for v in report['relative_update'].values())
    expected = state_digest(model, opt, ema, rng)
    before = digest(rng_state(rng)); before_model = digest(model.state_dict())
    with ema.apply(model):
        p1 = predict(model, data, indices=np.arange(min(513, len(data.raw['dev']))), micro=256)
        p2 = predict(model, data, indices=np.arange(len(p1)), micro=128)
    assert digest(rng_state(rng)) == before and digest(model.state_dict()) == before_model
    assert np.max(np.abs(p1-p2)) <= 1e-5
    ck = torch.load(out/'resume_probe.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); ema.load_state_dict(ck['ema']); restore_rng(ck['rng'], rng)
    del ck
    for i in range(10, 20):
        opt.schedule(i/steps); update(model, opt, data, order[i*256:(i+1)*256], a.endpoint); ema.step(model)
    assert state_digest(model, opt, ema, rng) == expected
    audit = dict(complete=True, continuous_updates=20, resume_raw_optimizer_ema_rng_exact=True,
                 evaluation_preserves_raw_and_rng=True, all_parameters_connected=True,
                 all_parameters_updated=True, fixed_train_microbatch=128, effective_batch=256,
                 evaluation_batch_max_drift=float(np.max(np.abs(p1-p2))), test_loaded=False)
    atomic(out/'path_audit.json', audit)
    result = dict(complete=True, pilot=True, from_scratch=True, test_loaded=False, identity=identity,
                  resume_exact=True, raw_path_exact=True, ema_resume_exact=True,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  median_step_seconds=float(np.median(timings[2:])), epoch_train_seconds_estimate=float(np.median(timings[2:])*steps),
                  checkpoint_bytes=(out/'resume_probe.pt').stat().st_size,
                  model_details=model.base.info, seconds=time.time()-began)
    atomic(out/'completion.json', result)
    atomic(out/'progress.json', dict(status='complete'))


def train(a, out, data, model, identity, spec):
    pilot = read(a.pilot_record/'completion.json'); pi = pilot['identity']
    assert pilot['complete'] and pilot['pilot'] and pilot['ema_resume_exact']
    for key in ('database', 'variant', 'endpoint', 'input_dim', 'parameters', 'source_manifest_sha256', 'window_sha256'):
        assert pi[key] == identity[key], key
    assert {k: v for k, v in pi['protocol'].items() if k != 'seed'} == {k: v for k, v in spec.items() if k != 'seed'}
    seed_all(a.seed+10000); rng = np.random.default_rng(a.seed)
    opt = Optimizer(model, spec); ema = ParameterEMA(model)
    atomic(out/'optimizer_initial.json', opt.values())
    n = len(data.raw['train']); steps = math.ceil(n/256); limit = identity['epoch_budget']
    updates = 0; history = []; best_ap = -1.; best_epoch = 0; began = time.time()
    labels = data.targets['dev']['y'][:, a.endpoint:a.endpoint+1]
    for epoch in range(1, limit+1):
        tick = time.time(); order = rng.permutation(n); total = torch.zeros((), device='cuda')
        for step, start in enumerate(range(0, n, 256), 1):
            ix = order[start:start+256]; opt.schedule(updates/steps)
            loss, report = update(model, opt, data, ix, a.endpoint, check=step == 1)
            ema.step(model); updates += 1; total += loss*len(ix)
            if report is not None:
                atomic(out/'gradient_latest.json', dict(epoch=epoch, groups=report, optimizer=opt.values()))
            if step == 1 or step % 30 == 0 or step == steps:
                atomic(out/'progress.json', dict(status='training', epoch=epoch, step=step, steps=steps,
                       optimizer_updates=updates, epoch_budget=limit, best_epoch=best_epoch, elapsed_seconds=time.time()-began))
        training_seconds = time.time()-tick
        with ema.apply(model):
            pred = predict(model, data)
        metrics = all_scores(labels, pred); ap = metrics['mean_ap']
        row = dict(epoch=epoch, dev_ap=ap, metrics=metrics, train_loss=float(total)/n,
                   optimizer_updates=updates, train_seconds=training_seconds)
        assert math.isfinite(ap) and math.isfinite(row['train_loss'])
        history.append(row)
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            save_checkpoint(out/'best.pt', model, opt, ema, rng,
                            dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap), identity, selected=True)
            np.savez_compressed(out/'predictions_dev.npz', ids=data.targets['dev']['ids'], labels=labels, probs=pred,
                                endpoint=a.endpoint, epoch=epoch, parameter_source='EMA')
            atomic(out/'metrics.json', metrics)
        save_checkpoint(out/'latest.pt', model, opt, ema, rng,
                        dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap, history=history), identity)
        row['epoch_seconds'] = time.time()-tick
        atomic(out/'history.json', history)
        atomic(out/'progress.json', dict(status='epoch_complete', epoch=epoch, epoch_budget=limit,
               best_ap=best_ap, best_epoch=best_epoch, dev_ap=ap, elapsed_seconds=time.time()-began))
    del opt, ema; gc.collect()
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    assert ck['role'] == 'EMA_selected_inference' and ck['identity'] == identity
    model.load_state_dict(ck['model']); del ck
    saved = dict(np.load(out/'predictions_dev.npz')); replay = predict(model, data)
    drift = float(np.max(np.abs(replay-saved['probs']))); assert drift <= 1e-5
    assert best_epoch == 1+int(np.argmax([r['dev_ap'] for r in history]))
    result = dict(identity, complete=True, pilot=False, from_scratch=True, test_loaded=False,
                  epochs=limit, best_epoch=best_epoch, dev_ap=best_ap, optimizer_updates=updates, ema_updates=updates,
                  metrics=all_scores(saved['labels'], saved['probs']), checkpoint_file='best.pt',
                  checkpoint_sha256=sha(out/'best.pt'), predictions_sha256=sha(out/'predictions_dev.npz'),
                  checkpoint_replay=dict(complete=True, max_probability_drift=drift),
                  termination='fixed_matched_CEM_budget', peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  seconds=time.time()-began)
    atomic(out/'completion.json', result); atomic(out/'progress.json', dict(status='complete', epochs=limit))


def evaluate(a, out, data, model, identity):
    fit = read(a.fit_output/'completion.json'); checkpoint = a.fit_output/'best.pt'
    assert fit['complete'] and sha(checkpoint) == fit['checkpoint_sha256']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model']); del ck
    before = digest(model.state_dict()); pred_dev = predict(model, data)
    saved = dict(np.load(a.fit_output/'predictions_dev.npz'))
    drift = float(np.max(np.abs(saved['probs']-pred_dev))); assert drift <= 1e-5
    pred = predict(model, data, 'test'); assert digest(model.state_dict()) == before
    labels = data.targets['test']['y'][:, a.endpoint:a.endpoint+1]
    np.savez_compressed(out/'predictions_test.npz', probs=pred, labels=labels, ids=data.targets['test']['ids'],
                        seed=a.seed, endpoint=a.endpoint, variant=a.variant)
    atomic(out/'completion.json', dict(identity, complete=True, optimizer_updates=0, parameters_updated=False,
           dev_replay_max_drift=drift, model_sha256=fit['checkpoint_sha256'], metrics=all_scores(labels, pred),
           predictions_sha256=sha(out/'predictions_test.npz')))


def main():
    p = argparse.ArgumentParser()
    for name in ('project', 'output', 'pilot-record', 'fit-output'):
        p.add_argument('--'+name, type=Path, required=name in ('project', 'output'))
    p.add_argument('--database', required=True); p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--endpoint', type=int, choices=(0, 1), required=True)
    p.add_argument('--action', choices=('probe', 'train', 'evaluate'), required=True)
    p.add_argument('--expected-gib', type=float, required=True); a = p.parse_args()
    project, out = a.project.resolve(strict=True), a.output.resolve()
    assert configuration()['database'] == a.database and out.is_relative_to(project) and not out.exists()
    assert shutil.disk_usage(project).free > (a.expected_gib+10)*2**30
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        path = Path(os.environ[name]).resolve(); assert path.is_relative_to(project); path.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True); torch.set_num_threads(4); seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); assert torch.cuda.is_available(); torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json', dict(status='loading_verified_inputs'))
    data = Data(project, window=a.variant == 'window_summary', test=a.action == 'evaluate')
    atomic(out/'progress.json', dict(status='randomly_initializing_native'))
    model = construct(data, a.seed, a.endpoint, a.variant)
    c = configuration(); spec = protocol(a.seed, a.endpoint)
    identity = dict(database=a.database, variant=a.variant, endpoint=a.endpoint, seed=a.seed,
                    input_dim=data.input_dim, parameters=sum(p.numel() for p in model.parameters()), members=8,
                    protocol=spec, epoch_budget=budget(a.database, a.seed, a.endpoint), ema_rule=EMA_RULE,
                    source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                    input_manifest_sha256=sha(data.preparation/'completion.json'),
                    window_sha256=sha(project/c['window_preparation']/'completion.json') if a.variant == 'window_summary' else None,
                    initial_weights_sha256=parameter_hash(model), initialization='fresh; no learned weights loaded')
    if a.variant == 'window_summary':
        assert identity['parameters'] == c['parent']['full_ema'][f'48:{a.endpoint}']['parameters']
    atomic(out/'run_manifest.json', identity); atomic(out/'model_details.json', model.base.info)
    model.cuda(); gc.collect()
    if a.action == 'probe': probe(a, out, data, model, identity, spec)
    elif a.action == 'train': train(a, out, data, model, identity, spec)
    else: evaluate(a, out, data, model, identity)


if __name__ == '__main__':
    main()
