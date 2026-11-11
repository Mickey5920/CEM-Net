"""EMA observes the original optimizer; it never supplies a training forward pass."""
import gc
import hashlib
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.mimiciii_ema_common import EARLY, EMA_RULE, budget, expected, atomic, read, sha
from scripts.cem_v1_group_confirm_runtime import update, predict
from scripts.tracehr_noa_ablation_common import Optimizer
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng


def digest(value):
    """Exact nested state comparison without retaining duplicate optimizer tensors."""
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        else: h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


class ParameterEMA:
    def __init__(self, model):
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.updates = 0

    @torch.no_grad()
    def step(self, model):
        beta = min(.999, (self.updates+1)/(self.updates+10))
        assert set(self.shadow) == set(dict(model.named_parameters()))
        for n, p in model.named_parameters():
            self.shadow[n].mul_(beta).add_(p.detach(), alpha=1-beta)
        self.updates += 1

    def state_dict(self):
        return dict(shadow=self.shadow, buffers=self.buffers, updates=self.updates, rule=EMA_RULE)

    def load_state_dict(self, state):
        assert state['rule'] == EMA_RULE and set(state['shadow']) == set(self.shadow)
        for n, v in state['shadow'].items(): self.shadow[n].copy_(v.to(self.shadow[n].device))
        for n, v in state['buffers'].items(): self.buffers[n].copy_(v.to(self.buffers[n].device))
        self.updates = state['updates']

    def assert_static_buffers(self, model):
        assert set(self.buffers) == set(dict(model.named_buffers()))
        assert all(torch.equal(b, self.buffers[n]) for n, b in model.named_buffers())

    @contextmanager
    def apply(self, model):
        """Use EMA only for evaluation/save and restore original tensors exactly."""
        self.assert_static_buffers(model)
        raw = {n: p.detach().clone() for n, p in model.named_parameters()}
        training = model.training
        try:
            with torch.no_grad():
                for n, p in model.named_parameters(): p.copy_(self.shadow[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters(): p.copy_(raw[n])
            model.train(training)
            self.assert_static_buffers(model)


def save_checkpoint(path, model, opt, ema, rng, state, identity, selected=False):
    value = dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.state_dict(),
                 rng=rng_state(rng), state=state, identity=identity, role='raw_resume')
    if selected:
        # A selected checkpoint is inference-only. latest.pt is the resumable state.
        value = dict(model={**{n:b for n,b in model.named_buffers()}, **ema.shadow},
                     raw_model=model.state_dict(), identity=identity, state=state,
                     role='EMA_selected_inference', ema_rule=EMA_RULE, ema_updates=ema.updates)
    tmp = path.with_name(path.name+'.writing')
    torch.save(value, tmp); os.replace(tmp, path)


def cpu_checks():
    m = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad(): m.weight.fill_(2.)
    e = ParameterEMA(m)
    with torch.no_grad(): m.weight.fill_(4.)
    e.step(m); torch.testing.assert_close(e.shadow['weight'], torch.full_like(m.weight, 3.8))
    with torch.no_grad(): m.weight.fill_(6.)
    e.step(m)
    torch.testing.assert_close(e.shadow['weight'], torch.full_like(m.weight, (2/11)*3.8+(9/11)*6))
    before = digest(m.state_dict())
    try:
        with e.apply(m):
            assert digest(m.state_dict()) != before
            raise ValueError('test restoration on failure')
    except ValueError: pass
    assert digest(m.state_dict()) == before and all(p.requires_grad for p in m.parameters())
    f = ParameterEMA(m); f.load_state_dict(e.state_dict())
    assert digest(f.state_dict()) == digest(e.state_dict())
    return ['EMA_formula_and_warmup', 'shadow_restore_including_exception', 'EMA_serialization',
            'six_fixed_budgets_total_584_epochs']


def probe(a, out, data, model, identity, spec):
    started = time.time(); constructor_peak = torch.cuda.max_memory_reserved()/2**20
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n, batch = len(data.targets['train']['y']), spec['effective_batch']
    steps = math.ceil(n/batch); micro = spec['training_microbatch']
    assert (batch, micro, spec['evaluation_microbatch']) == (256, 128, 256)
    baseline = []; evidence = []
    for with_ema in (False, True):
        model.load_state_dict(initial); model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        seed_all(a.seed+10000); rng = np.random.default_rng(a.seed)
        opt = Optimizer(model, spec, 1.); ema = ParameterEMA(model) if with_ema else None
        order = rng.permutation(n)
        timings = []
        for i in range(20):
            opt.schedule((i+(0 if spec['native'] else 1))/steps)
            tick = time.perf_counter()
            loss, report = update(model, opt, data, order[i*batch:(i+1)*batch], micro, a.endpoint, check=i in (0,19))
            if ema is not None: ema.step(model)
            torch.cuda.synchronize(); timings.append(time.perf_counter()-tick)
            record = dict(parameters=digest(model.state_dict()),
                          gradients=digest({n:p.grad for n,p in model.named_parameters()}),
                          optimizer=digest(opt.state_dict()), rng=digest(rng_state(rng)), loss=float(loss))
            if with_ema:
                assert record == baseline[i], ('EMA altered raw trajectory', i+1)
                if i == 0:
                    # The original native schedule starts at learning-rate factor zero.
                    # Connectivity is checked here; actual movement is checked at step 20.
                    atomic(out/'first_step_gradient_check.json', report)
                    assert all(v == 0 for v in report['relative_update'].values())
                    groups = report['loss_paths']['groups']
                    assert all(g['main_connected'] == g['tensors'] and g['aux_connected'] == 0 for g in groups.values())
                if i == 19:
                    assert all(v > 0 for v in report['relative_update'].values())
                    atomic(out/'gradient_check.json', report)
            else: baseline.append(record)
            atomic(out/'progress.json', dict(status='training_probe', ema_enabled=with_ema, updates=i+1))
        evidence.append(dict(ema_enabled=with_ema, median_step_seconds=float(np.median(timings[1:])),
                             peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        if not with_ema: del opt; gc.collect()
    # Evaluation must not perturb any training state or RNG stream.
    before = digest(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(rng)))
    ix = np.arange(512)
    with ema.apply(model): p128 = predict(model, data, 128, ix); p256 = predict(model, data, 256, ix)
    assert float(np.max(np.abs(p128-p256))) <= 1e-5
    assert before == digest(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(rng)))
    save_checkpoint(out/'representative_checkpoint.pt', model, opt, ema, rng, dict(updates=20), identity)
    ck = torch.load(out/'representative_checkpoint.pt', map_location='cpu', weights_only=False)
    next_ix = rng.permutation(n)[:batch]
    opt.schedule((20+(0 if spec['native'] else 1))/steps)
    update(model, opt, data, next_ix, micro, a.endpoint); ema.step(model)
    after = digest(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.state_dict(), rng=rng_state(rng)))
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); ema.load_state_dict(ck['ema']); restore_rng(ck['rng'], rng)
    replay_ix = rng.permutation(n)[:batch]; assert np.array_equal(replay_ix, next_ix)
    opt.schedule((20+(0 if spec['native'] else 1))/steps)
    update(model, opt, data, replay_ix, micro, a.endpoint); ema.step(model)
    assert after == digest(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.state_dict(), rng=rng_state(rng)))
    audit = dict(complete=True, continuous_updates=20, all_raw_parameters_gradients_optimizer_rng_loss_exact=True,
                 evaluation_preserves_raw_and_rng=True, resume_raw_optimizer_ema_rng_exact=True,
                 formula=EMA_RULE, per_step_reference=baseline, benchmarks=evidence, test_loaded=False)
    atomic(out/'ema_path_audit.json', audit)
    result = dict(complete=True, pilot=True, from_scratch=True, test_loaded=False, frozen_learnable_parameters=0,
                  identity=identity, selected_microbatch=micro, evaluation_microbatch=256, effective_batch=batch,
                  optimizer_resume_max_parameter_drift=0., raw_path_exact=True, ema_resume_exact=True,
                  ema_path_audit_sha256=sha(out/'ema_path_audit.json'),
                  peak_reserved_mib=max(constructor_peak, torch.cuda.max_memory_reserved()/2**20,
                                        max(r['peak_reserved_mib'] for r in evidence)),
                  checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
                  seconds=time.time()-started, finished_at=time.time())
    atomic(out/'completion.json', result); atomic(out/'progress.json', dict(status='complete', seconds=result['seconds']))


def train(a, out, data, model, identity, spec):
    pilot = read(a.pilot_record/'completion.json')
    assert pilot['complete'] and pilot['pilot'] and pilot['raw_path_exact'] and pilot['ema_resume_exact']
    assert pilot['identity'] == identity
    atomic(out/'pilot_identity_check.json', dict(match=True, fresh_initialization=True, pilot_weights_loaded=False,
                                               pilot=str(a.pilot_record)))
    seed_all(a.seed+10000); rng = np.random.default_rng(a.seed)
    opt = Optimizer(model, spec, 1.); ema = ParameterEMA(model)
    atomic(out/'optimizer_initial.json', opt.values())
    n, batch = len(data.targets['train']['y']), spec['effective_batch']; steps = math.ceil(n/batch)
    ref = expected()['references'][f'noEvidence_s{a.seed}_t{a.endpoint}']; refdir = Path(ref['output'])
    reference_history = read(refdir/'history.json')
    limit = budget(a.seed, a.endpoint); assert limit == ref['epochs'] == len(reference_history)
    started = time.time(); history = []; updates = 0; best_ap = -1.; best_epoch = 0
    atomic(out/'training_started.json', dict(started_at=started, actual_optimizer_updates=0, epoch_budget=limit))
    for epoch in range(1, limit+1):
        tick = time.time(); order = rng.permutation(n); total = torch.zeros((), device='cuda')
        for step, start in enumerate(range(0, n, batch), 1):
            ix = order[start:start+batch]
            opt.schedule((updates+(0 if spec['native'] else 1))/steps)
            loss, report = update(model, opt, data, ix, spec['training_microbatch'], a.endpoint, check=step==1)
            ema.step(model); updates += 1; assert updates == ema.updates
            total += loss*len(ix)
            if report is not None: atomic(out/'gradient_latest.json', dict(epoch=epoch, groups=report, optimizer=opt.values()))
            if step == 1 or step % 30 == 0 or step == steps:
                atomic(out/'progress.json', dict(status='training', epoch=epoch, step=step, steps=steps,
                       optimizer_updates=updates, ema_updates=ema.updates, best_ap=best_ap, best_epoch=best_epoch,
                       active_epoch_limit=limit, elapsed_seconds=time.time()-started))
        train_seconds = time.time()-tick
        atomic(out/'progress.json', dict(status='evaluating', epoch=epoch, optimizer_updates=updates, active_epoch_limit=limit))
        labels = data.targets['dev']['y'][:,a.endpoint:a.endpoint+1]
        val_tick = time.time(); raw_pred = predict(model, data, 256); raw_metrics = all_scores(labels, raw_pred)
        # Original trajectory comparison is a fidelity check, not a selection option.
        raw_ap_delta = raw_metrics['mean_ap']-reference_history[epoch-1]['dev_ap']
        assert abs(raw_ap_delta) <= 1e-6, ('Original raw trajectory changed', epoch, raw_ap_delta)
        before_rng = digest(rng_state(rng))
        with ema.apply(model):
            pred = predict(model, data, 256)
            member_ap = [float(average_precision_score(labels[:,0], model._diagnostic_member_probs[:,i])) for i in range(8)]
        assert digest(rng_state(rng)) == before_rng
        metrics = all_scores(labels, pred)
        row = dict(epoch=epoch, dev_ap=metrics['mean_ap'], metrics=metrics, raw_metrics=raw_metrics,
                   same_epoch_ema_minus_raw_AP=metrics['mean_ap']-raw_metrics['mean_ap'],
                   raw_reference_AP_delta=raw_ap_delta, train_loss=float(total)/n,
                   train_seconds=train_seconds, evaluation_seconds=time.time()-val_tick,
                   optimizer=opt.values(), optimizer_updates=updates, ema_updates=ema.updates,
                   parameter_source='EMA', main_view='z0', checkpoint_selection_view='z0',
                   member_AP=member_ap, auxiliary_enabled=False)
        assert math.isfinite(row['dev_ap']) and math.isfinite(row['train_loss'])
        history.append(row)
        if row['dev_ap'] > best_ap:
            best_ap, best_epoch = row['dev_ap'], epoch
            state = dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap)
            save_checkpoint(out/'best.pt', model, opt, ema, rng, state, identity, selected=True)
            np.savez_compressed(out/'predictions_dev.npz', ids=data.targets['dev']['ids'], labels=labels, probs=pred,
                                raw_probs_at_EMA_selected_epoch=raw_pred, epoch=epoch, endpoint=a.endpoint,
                                main_view='z0', parameter_source='EMA')
            atomic(out/'metrics.json', metrics)
        save_checkpoint(out/'latest.pt', model, opt, ema, rng,
                        dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap, history=history), identity)
        if epoch in EARLY:
            prior = torch.load(refdir/f'epoch_{epoch:03d}_model.pt', map_location='cpu', weights_only=False)
            drift = max(float((p.detach().cpu()-prior['model'][name]).abs().max()) for name,p in model.named_parameters())
            assert drift <= 1e-6, ('Original early parameters changed', epoch, drift)
            raw_exact = digest(model.state_dict()) == digest(prior['model']); del prior
            early = out/f'epoch_{epoch:03d}_model.pt'; tmp = early.with_name(early.name+'.writing')
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), epoch=epoch, updates=updates,
                            identity=identity, role='raw_and_EMA_audit'), tmp); os.replace(tmp, early)
            atomic(out/f'epoch_{epoch:03d}_diagnostics.json', dict(checkpoint_sha256=sha(early), row=row,
                   gradient=read(out/'gradient_latest.json'), raw_reference_max_parameter_drift=drift,
                   raw_reference_state_exact=raw_exact))
        row['epoch_seconds_with_checkpoint'] = time.time()-tick
        atomic(out/'history.json', history)
        atomic(out/'progress.json', dict(status='epoch_complete', epoch=epoch, best_ap=best_ap, best_epoch=best_epoch,
               dev_ap=row['dev_ap'], raw_dev_ap=raw_metrics['mean_ap'], optimizer_updates=updates,
               active_epoch_limit=limit, epoch_seconds=row['epoch_seconds_with_checkpoint'], elapsed_seconds=time.time()-started))
    del opt, ema; gc.collect()
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    assert ck['role'] == 'EMA_selected_inference' and ck['identity'] == identity and ck['state']['epoch'] == best_epoch
    model.load_state_dict(ck['model']); del ck
    saved = dict(np.load(out/'predictions_dev.npz')); replay = predict(model, data, 256)
    drift = float(np.max(np.abs(replay-saved['probs']))); assert drift <= 1e-5
    scores = all_scores(saved['labels'], saved['probs']); assert abs(scores['mean_ap']-best_ap) < 1e-12
    assert best_epoch == 1+int(np.argmax([r['dev_ap'] for r in history]))
    result = dict(complete=True, pilot=False, from_scratch=True, test_loaded=False, frozen_learnable_parameters=0,
                  **identity, epochs=limit, best_epoch=best_epoch, dev_ap=best_ap, metrics=scores,
                  optimizer_updates=updates, ema_updates=updates, main_view='z0', checkpoint_selection_view='z0',
                  stopping_view='z0', stopping_status=dict(stop=True, reason='predeclared original epoch budget', epoch_budget=limit),
                  termination='fixed_original_budget', checkpoint_file='best.pt', parameter_source='EMA',
                  checkpoint_sha256=sha(out/'best.pt'), predictions_sha256=sha(out/'predictions_dev.npz'),
                  checkpoint_replay=dict(complete=True, max_probability_drift=drift),
                  raw_reference_max_AP_drift=max(abs(r['raw_reference_AP_delta']) for r in history),
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, seconds=time.time()-started, finished_at=time.time())
    atomic(out/'completion.json', result); atomic(out/'progress.json', dict(status='complete', epochs=limit, best_ap=best_ap, best_epoch=best_epoch))
