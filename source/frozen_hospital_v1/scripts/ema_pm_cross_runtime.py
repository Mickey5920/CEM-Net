"""Unchanged raw optimizer and EMA rule on P/M deletion trajectories."""
import gc, math, os, time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.ema_pm_cross_common import EARLY, budget, atomic, read, sha, bind_probe
from scripts.mimiciii_ema_runtime import ParameterEMA, digest, save_checkpoint
from scripts.tracehr_fte_binding_runtime import update, predict
from scripts.tracehr_noa_ablation_common import Optimizer
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all, rng_state

def probe(a, out, data, model, identity, spec):
    bind_probe()(a, out, data, model, identity, spec)
    atomic(out/'path_audit.json', read(out/'ema_path_audit.json'))

def train(a, out, data, model, identity, spec):
    pilot = read(a.pilot_record/'completion.json')
    assert pilot['complete'] and pilot['pilot'] and pilot['raw_path_exact'] and pilot['ema_resume_exact']
    pi = pilot['identity']
    for key in ('database','variant','endpoint','input_dim','source_manifest_sha256','input_manifest_sha256',
                'ema_rule','epoch_budget','parameters','members','lr_multiplier','ablated_component'):
        assert pi[key] == identity[key], key
    assert {k:v for k,v in pi['protocol'].items() if k!='seed'} == {k:v for k,v in spec.items() if k!='seed'}
    if a.seed == 48: assert pi == identity
    observation = read(out/'initialization_observation.json')
    assert observation['full_initial_tensors_exact'] and observation['deletion']['retained_initials_exact']
    atomic(out/'pilot_identity_check.json', dict(complete=True, execution_path_match=True,
        same_seed_initial_match=(pi['initial_weights_sha256']==identity['initial_weights_sha256']) if a.seed==48 else None,
        reference_seed_initial_match=True, fresh_initialization=True, pilot_weights_loaded=False,
        path_pilot_seed=48, formal_seed=a.seed, pilot=str(a.pilot_record)))
    seed_all(a.seed+10000); rng = np.random.default_rng(a.seed)
    opt = Optimizer(model, spec, 1.); ema = ParameterEMA(model)
    atomic(out/'optimizer_initial.json', opt.values())
    n, batch = len(data.targets['train']['y']), spec['effective_batch']; steps = math.ceil(n/batch)
    limit = budget(a.seed, a.endpoint)
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
        before_rng = digest(rng_state(rng))
        with ema.apply(model):
            pred = predict(model, data, 256)
            member_ap = [float(average_precision_score(labels[:,0], model._diagnostic_member_probs[:,i])) for i in range(8)]
        assert digest(rng_state(rng)) == before_rng
        metrics = all_scores(labels, pred)
        row = dict(epoch=epoch, dev_ap=metrics['mean_ap'], metrics=metrics, raw_metrics=raw_metrics,
            same_epoch_ema_minus_raw_AP=metrics['mean_ap']-raw_metrics['mean_ap'], train_loss=float(total)/n,
            train_seconds=train_seconds, evaluation_seconds=time.time()-val_tick, optimizer=opt.values(),
            optimizer_updates=updates, ema_updates=ema.updates, parameter_source='EMA', main_view='z0',
            checkpoint_selection_view='z0', member_AP=member_ap, auxiliary_enabled=False)
        assert math.isfinite(row['dev_ap']) and math.isfinite(row['train_loss'])
        history.append(row)
        if row['dev_ap'] > best_ap:
            best_ap, best_epoch = row['dev_ap'], epoch
            save_checkpoint(out/'best.pt', model, opt, ema, rng,
                dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap), identity, selected=True)
            np.savez_compressed(out/'predictions_dev.npz', ids=data.targets['dev']['ids'], labels=labels, probs=pred,
                raw_probs_at_EMA_selected_epoch=raw_pred, epoch=epoch, endpoint=a.endpoint, main_view='z0', parameter_source='EMA')
            atomic(out/'metrics.json', metrics)
        save_checkpoint(out/'latest.pt', model, opt, ema, rng,
            dict(epoch=epoch, updates=updates, best_epoch=best_epoch, best_ap=best_ap, history=history), identity)
        if epoch in EARLY:
            early = out/f'epoch_{epoch:03d}_model.pt'; tmp = early.with_name(early.name+'.writing')
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), epoch=epoch, updates=updates,
                identity=identity, role='raw_and_EMA_audit'), tmp); os.replace(tmp, early)
            atomic(out/f'epoch_{epoch:03d}_diagnostics.json', dict(checkpoint_sha256=sha(early), row=row,
                gradient=read(out/'gradient_latest.json')))
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
    result = dict(identity)
    result.update(complete=True, pilot=False, from_scratch=True, test_loaded=False, frozen_learnable_parameters=0,
        epochs=limit, best_epoch=best_epoch, dev_ap=best_ap, metrics=scores, optimizer_updates=updates, ema_updates=updates,
        main_view='z0', checkpoint_selection_view='z0', stopping_view='z0',
        stopping_status=dict(stop=True, reason='predeclared full EMA epoch budget', epoch_budget=limit),
        termination='fixed_original_budget', checkpoint_file='best.pt', parameter_source='EMA',
        checkpoint_sha256=sha(out/'best.pt'), predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True, max_probability_drift=drift),
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, seconds=time.time()-started, finished_at=time.time())
    atomic(out/'completion.json', result)
    atomic(out/'progress.json', dict(status='complete', epochs=limit, best_ap=best_ap, best_epoch=best_epoch))
