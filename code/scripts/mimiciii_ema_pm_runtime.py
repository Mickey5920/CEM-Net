"""Reuse the original EMA engine on single-deletion training trajectories."""
import gc, math, os, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.mimiciii_ema_pm_common import EARLY, EMA_RULE, budget, expected, atomic, read, sha, reference
from scripts.mimiciii_ema_runtime import ParameterEMA, digest, save_checkpoint, probe, cpu_checks
from scripts.cem_v1_group_confirm_runtime import update, predict
from scripts.tracehr_noa_ablation_common import Optimizer
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all, rng_state

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
    ref = reference(a.variant, a.seed, a.endpoint); refdir = Path(ref['output'])
    reference_history = read(refdir/'history.json')
    limit = budget(a.seed, a.endpoint); assert ref['epochs'] == len(reference_history)
    compared_epochs = min(limit, len(reference_history))
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
        raw_ap_delta = None
        if epoch <= compared_epochs:
            raw_ap_delta = raw_metrics['mean_ap']-reference_history[epoch-1]['dev_ap']
            assert abs(raw_ap_delta) <= 1e-6, ('Archived deletion raw trajectory changed', epoch, raw_ap_delta)
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
                  raw_reference_max_AP_drift=max(abs(r['raw_reference_AP_delta']) for r in history if r['raw_reference_AP_delta'] is not None),
                  raw_reference_compared_epochs=compared_epochs, archived_raw_epoch_budget=ref['epochs'],
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, seconds=time.time()-started, finished_at=time.time())
    atomic(out/'completion.json', result); atomic(out/'progress.json', dict(status='complete', epochs=limit, best_ap=best_ap, best_epoch=best_epoch))
