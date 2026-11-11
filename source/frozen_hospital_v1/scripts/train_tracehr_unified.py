"""Measured pilots and joint random-initialization training for N0--N3."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.tracehr_unified import UnifiedNetwork, member_loss, parameter_hash
from scripts.tracehr_unified_common import RUN, PROTOCOL, Data, prepare, stopping
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output
from scripts.tracehr_oof_common import all_scores


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, python=random.getstate(), numpy_global=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state, rng):
    rng.bit_generator.state = state['numpy']
    random.setstate(state['python'])
    np.random.set_state(state['numpy_global'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_checkpoint(path, model, optimizer, rng, state, identity):
    temp = path.with_name(path.name+'.writing')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(rng),
                    state=state, identity=identity, lr_schedule=PROTOCOL), temp)
    os.replace(temp, path)


def gradients(model):
    result = {}
    for name, p in model.named_parameters():
        assert p.requires_grad and p.grad is not None, ('Frozen or disconnected parameter', name)
        assert torch.isfinite(p.grad).all(), name
        key = name.split('.')[0]
        row = result.setdefault(key, dict(tensors=0, norm_squared=0.))
        row['tensors'] += 1
        row['norm_squared'] += float(p.grad.detach().square().sum())
    assert all(row['norm_squared'] > 0 for row in result.values())
    return result


def update(model, optimizer, data, ix, microbatch, check=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device='cuda')
    for start in range(0, len(ix), microbatch):
        part = ix[start:start+microbatch]
        x, y = data.batch('train', part)
        loss = member_loss(model(x), y)
        fraction = len(part)/len(ix)
        (loss*fraction).backward()
        loss_sum += loss.detach()*fraction
    report = gradients(model) if check else None
    # This also rejects nonfinite gradients rather than hiding an unstable update.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
    optimizer.step()
    return loss_sum.detach(), report


@torch.no_grad()
def predict(model, data, microbatch, indices=None):
    model.eval()
    if indices is None:
        indices = np.arange(len(data.targets['dev']['y']))
    parts = []
    for start in range(0, len(indices), microbatch):
        x, _ = data.batch('dev', indices[start:start+microbatch])
        parts.append(model.probabilities(x).cpu().numpy())
    value = np.concatenate(parts)
    assert np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
    return value


def optimizer_for(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=PROTOCOL['weight_decay'])


def rate_for_update(base, update_index, steps):
    warmup = PROTOCOL['warmup_epochs']*steps
    end = PROTOCOL['cosine_end_epoch']*steps
    if update_index < warmup:
        return base*(update_index+1)/warmup
    fraction = min(1., (update_index-warmup)/max(1, end-warmup))
    return base*(PROTOCOL['min_lr_ratio']+(1-PROTOCOL['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction)))


def probe(a, out, data, model, identity):
    started = time.time()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(a.seed)
    orders = [rng.permutation(len(data.targets['train']['y']))[:512] for _ in range(4)]
    benchmarks = []
    gradient_checks = {}
    for micro in (128, 256):
        model.load_state_dict(initial)
        seed_all(a.seed+10000)
        optimizer = optimizer_for(model, a.lr)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i, ix in enumerate(orders):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss, report = update(model, optimizer, data, ix, micro, check=i == 0)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-tick)
            assert torch.isfinite(loss)
            if report is not None:
                gradient_checks[str(micro)] = report
        benchmarks.append(dict(microbatch=micro, effective_batch=512, step_seconds=times,
                               median_seconds=float(np.median(times[1:])),
                               peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                               peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        atomic(out/'progress.json', dict(status='training_probe', benchmarks=benchmarks))
        del optimizer
    fastest = min(row['median_seconds'] for row in benchmarks)
    selected = next(row['microbatch'] for row in benchmarks if row['median_seconds'] <= fastest*1.1)
    model.load_state_dict(initial)
    eval_rows = []
    reference = None
    ix = np.arange(min(1024, len(data.targets['dev']['y'])))
    for micro in (128, 256):
        tick = time.perf_counter()
        p = predict(model, data, micro, ix)
        drift = 0. if reference is None else float(np.max(np.abs(reference-p)))
        assert drift <= 1e-5, ('Evaluation batching drift', drift)
        if reference is None:
            reference = p
        eval_rows.append(dict(microbatch=micro, seconds=time.perf_counter()-tick, probability_drift=drift))
    evaluation_microbatch = min(eval_rows, key=lambda r: r['seconds'])['microbatch']
    seed_all(a.seed+10000)
    optimizer = optimizer_for(model, a.lr)
    update(model, optimizer, data, orders[0], selected)
    save_checkpoint(out/'representative_checkpoint.pt', model, optimizer, rng, dict(updates=1), identity)
    ck = torch.load(out/'representative_checkpoint.pt', map_location='cpu', weights_only=False)
    next_ix = rng.permutation(len(data.targets['train']['y']))[:512]
    update(model, optimizer, data, next_ix, selected)
    expected = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    restore_rng(ck['rng'], rng)
    replay_ix = rng.permutation(len(data.targets['train']['y']))[:512]
    assert np.array_equal(next_ix, replay_ix)
    update(model, optimizer, data, replay_ix, selected)
    drift = max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in model.named_parameters())
    assert drift <= 1e-6, ('Optimizer/RNG resume mismatch', drift)
    after = parameter_hash(model)
    assert after != identity['initial_weights_sha256']
    result = dict(complete=True, pilot=True, from_scratch=True, test_loaded=False, frozen_learnable_parameters=0,
                  identity=identity, selected_microbatch=selected, evaluation_microbatch=evaluation_microbatch,
                  effective_batch=512, benchmarks=benchmarks, evaluation_benchmarks=eval_rows,
                  gradient_checks=gradient_checks, optimizer_resume_max_parameter_drift=drift,
                  peak_reserved_mib=max(torch.cuda.max_memory_reserved()/2**20, max(r['peak_reserved_mib'] for r in benchmarks)),
                  checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
                  seconds=time.time()-started, finished_at=time.time())
    atomic(out/'completion.json', result)
    atomic(out/'progress.json', dict(status='complete', seconds=result['seconds']))


def train(a, out, data, model, identity):
    pilot = read(a.pilot_record/'completion.json')
    assert pilot['complete'] and pilot['pilot']
    for name in ('database', 'variant', 'seed', 'input_manifest_sha256', 'source_manifest_sha256',
                 'initial_weights_sha256', 'common_weights_sha256'):
        assert pilot['identity'][name] == identity[name], ('Pilot identity mismatch', name)
    micro = pilot['selected_microbatch']
    eval_micro = pilot['evaluation_microbatch']
    atomic(out/'pilot_identity_check.json', dict(match=True, pilot=str(a.pilot_record), microbatch=micro,
                                                effective_batch=512, evaluation_microbatch=eval_micro))
    seed_all(a.seed+10000)
    rng = np.random.default_rng(a.seed)
    optimizer = optimizer_for(model, a.lr)
    n = len(data.targets['train']['y'])
    steps = math.ceil(n/512)
    history = []
    best_ap, best_epoch, updates = -1., 0, 0
    first_epoch = 1
    if a.resume:
        ck = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        assert ck['identity'] == identity and ck['lr_schedule'] == PROTOCOL
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng'], rng)
        state = ck['state']
        history = state['history']
        best_ap, best_epoch, updates = state['best_ap'], state['best_epoch'], state['updates']
        first_epoch = state['epoch']+1
        del ck
    if a.start_barrier:
        atomic(out/'progress.json', dict(status='waiting_coexistence_barrier'))
        deadline = time.time()+600
        while not a.start_barrier.exists():
            assert time.time() < deadline, 'Coexistence barrier timeout'
            time.sleep(.5)
    started = time.time()
    atomic(out/'training_started.json', dict(started_at=started, pid=os.getpid()))
    torch.cuda.reset_peak_memory_stats()
    max_epochs = 1 if a.action == 'smoke' else PROTOCOL['max_epochs']
    for epoch in range(first_epoch, max_epochs+1):
        tick = time.time()
        order = rng.permutation(n)
        if a.action == 'smoke':
            order = order[:20*512]
        epoch_steps = math.ceil(len(order)/512)
        total_loss = torch.zeros((), device='cuda')
        for step, start in enumerate(range(0, len(order), 512), 1):
            ix = order[start:start+512]
            lr = rate_for_update(a.lr, updates, steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss, report = update(model, optimizer, data, ix, micro, check=(epoch == 1 and step == 1))
            total_loss += loss*len(ix)
            updates += 1
            if report is not None:
                atomic(out/'first_step_gradients.json', report)
            if step == 1 or step % 20 == 0 or step == epoch_steps:
                atomic(out/'progress.json', dict(status='training', epoch=epoch, step=step, steps=epoch_steps,
                                                 best_ap=best_ap, best_epoch=best_epoch, lr=lr,
                                                 elapsed_seconds=time.time()-started))
        train_seconds = time.time()-tick
        dev_ix = np.arange(min(1024, len(data.targets['dev']['y']))) if a.action == 'smoke' else None
        atomic(out/'progress.json', dict(status='evaluating', epoch=epoch, best_ap=best_ap, best_epoch=best_epoch))
        val_started = time.time()
        pred = predict(model, data, eval_micro, dev_ix)
        labels = data.targets['dev']['y'][:len(pred)]
        metrics = all_scores(labels, pred)
        row = dict(epoch=epoch, dev_ap=metrics['mean_ap'], metrics=metrics,
                   train_loss=float(total_loss)/len(order), train_seconds=train_seconds,
                   evaluation_seconds=time.time()-val_started, epoch_seconds=time.time()-tick, lr=lr)
        assert math.isfinite(row['train_loss']) and math.isfinite(row['dev_ap'])
        history.append(row)
        improved = row['dev_ap'] > best_ap
        if improved:
            best_ap, best_epoch = row['dev_ap'], epoch
        state = dict(epoch=epoch, updates=updates, best_ap=best_ap, best_epoch=best_epoch, history=history)
        if improved:
            save_checkpoint(out/'best.pt', model, optimizer, rng, state, identity)
            np.savez_compressed(out/'predictions_dev.npz', ids=data.targets['dev']['ids'][:len(pred)],
                                labels=labels, probs=pred, epoch=epoch)
            atomic(out/'metrics.json', metrics)
        save_checkpoint(out/'latest.pt', model, optimizer, rng, state, identity)
        row['epoch_seconds_with_checkpoint'] = time.time()-tick
        atomic(out/'history.json', history)
        atomic(out/'progress.json', dict(status='epoch_complete', epoch=epoch, dev_ap=row['dev_ap'],
                                         best_ap=best_ap, best_epoch=best_epoch,
                                         epoch_seconds=row['epoch_seconds_with_checkpoint'],
                                         stopping=stopping(history), elapsed_seconds=time.time()-started))
        print(json.dumps(dict(epoch=epoch, dev_ap=row['dev_ap'], best_ap=best_ap,
                              seconds=row['epoch_seconds_with_checkpoint'])), flush=True)
        if a.action != 'smoke' and stopping(history)['stop']:
            break
    del optimizer
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    assert ck['identity'] == identity and ck['state']['epoch'] == best_epoch
    model.load_state_dict(ck['model'])
    del ck
    selected = dict(np.load(out/'predictions_dev.npz'))
    assert int(selected['epoch']) == best_epoch
    assert np.array_equal(selected['ids'], data.targets['dev']['ids'][:len(selected['probs'])])
    assert np.array_equal(selected['labels'], data.targets['dev']['y'][:len(selected['probs'])])
    reference_metrics = all_scores(selected['labels'], selected['probs'])
    assert reference_metrics == read(out/'metrics.json') and abs(reference_metrics['mean_ap']-best_ap) <= 1e-12
    replay = predict(model, data, eval_micro, np.arange(len(selected['probs'])))
    drift = float(np.max(np.abs(selected['probs']-replay)))
    assert drift <= 1e-5, ('Selected checkpoint replay mismatch', drift)
    result = dict(complete=True, pilot=a.action == 'smoke', from_scratch=True, test_loaded=False,
                  frozen_learnable_parameters=0, **identity, epochs=len(history), best_epoch=best_epoch,
                  dev_ap=best_ap, metrics=reference_metrics, seconds=time.time()-started, finished_at=time.time(),
                  stopping_status=stopping(history),
                  termination='smoke' if a.action == 'smoke' else ('early_stopping' if stopping(history)['stop'] else 'max_epochs'),
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  checkpoint_sha256=sha(out/'best.pt'), predictions_sha256=sha(out/'predictions_dev.npz'),
                  checkpoint_replay=dict(complete=True, max_probability_drift=drift,
                                         selection_predictions_preserved=True, replay_ap=all_scores(selected['labels'], replay)['mean_ap']))
    atomic(out/'completion.json', result)
    atomic(out/'progress.json', dict(status='complete', epochs=len(history), best_epoch=best_epoch, best_ap=best_ap))


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'output', 'preparation', 'pilot-record', 'start-barrier'):
        parser.add_argument('--'+name, type=Path, required=name in ('project','output'))
    parser.add_argument('--database', choices=['mimic', 'eicu'], required=True)
    parser.add_argument('--action', choices=['prepare','probe','smoke','train'], required=True)
    parser.add_argument('--variant', choices=['N0','N1','N2','N3'], default='N0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-gib', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    a.project = a.project.resolve(strict=True)
    out = checked_output(a.project, a.output)
    assert out.is_dir() if a.resume else not out.exists(), 'Refuse ambiguous output reuse'
    assert shutil.disk_usage(a.project).free > (a.expected_gib+10)*2**30
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME'):
        path = checked_output(a.project, os.environ[key])
        path.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=a.resume)
    atomic(out/'storage.json', dict(project=str(a.project), code=str(ROOT.resolve()), output=str(out),
                                   cache=os.environ['XDG_CACHE_HOME'], temp=os.environ['TMPDIR'], expected_gib=a.expected_gib))
    if a.action == 'prepare':
        prepare(a.project, a.database, out)
        return
    assert a.preparation and torch.cuda.is_available()
    atomic(out/'progress.json', dict(status='loading_verified_inputs'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = Data(a.project, a.database, a.preparation)
    model = UnifiedNetwork(data.input_dim, data.prep, a.variant, a.seed)
    identity = dict(database=a.database, variant=a.variant, lr=a.lr, seed=a.seed, protocol=PROTOCOL,
                    source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                    input_manifest_sha256=sha(a.preparation/'completion.json'),
                    initial_weights_sha256=parameter_hash(model), common_weights_sha256=parameter_hash(model, True),
                    parameters=sum(p.numel() for p in model.parameters()), input_dim=data.input_dim,
                    members=model.members)
    assert all(p.requires_grad for p in model.parameters())
    if not a.resume:
        atomic(out/'run_manifest.json', identity)
    else:
        assert read(out/'run_manifest.json') == identity
    model = model.cuda()
    if a.action == 'probe':
        probe(a, out, data, model, identity)
    else:
        assert a.pilot_record
        train(a, out, data, model, identity)


if __name__ == '__main__':
    main()
