"""Bounded rate/parameter-group checks; restore fresh weights and every RNG."""
import copy, gc
from scripts.cem_v1_rate_common import *
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng


def bounded_rate_check(model, data, spec, endpoint, seed):
    state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    initial_hash = parameter_hash(model)
    rng = np.random.default_rng(seed)
    saved_rng = rng_state(rng)
    training = model.training
    base_spec = v1_protocol('V1', 'mimic', seed)
    ratio = spec['fusion_lr']/base_spec['fusion_lr']
    before_opt, after_opt = OriginalOptimizer(model, base_spec), Optimizer(model, spec)
    rate_reference_opt = None
    schedules=[]
    try:
        for epoch in (0., 1/1180, 1., 8., 15., 17., 30., 52.):
            before_opt.schedule(epoch); before=copy.deepcopy(before_opt.values())
            after_opt.schedule(epoch); after=copy.deepcopy(after_opt.values())
            assert before['base']==after['base'], epoch
            assert before['fusion']['betas']==after['fusion']['betas']
            assert before['fusion']['weight_decay']==after['fusion']['weight_decay']
            assert abs(after['fusion']['lr']-ratio*before['fusion']['lr'])<1e-15
            schedules.append(dict(epoch=epoch, original_fusion_lr=before['fusion']['lr'],
                actual_fusion_lr=after['fusion']['lr'], base_and_input_rates_exact=True))
        # Same gradients and native clock: only the fusion parameter update may differ.
        seed_all(seed+10000); model.train(); model.zero_grad(set_to_none=True)
        ix=np.arange(256)
        for start in (0,128):
            x, y=data.batch('train',ix[start:start+128])
            logits=model(x); loss=model.loss(logits,y[:,endpoint])
            (loss*.5).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before_opt.schedule(8.); before_opt.step()
        reference={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        model.load_state_dict(state)
        # The original implementation accepts arbitrary fusion_lr but rejects the
        # tuning metadata flag. This diagnostic oracle uses exactly the target
        # rate and original constructor, with no search or selection. Formal runs
        # retain tuning=True. Equal-rate updates must be bitwise identical.
        oracle_spec=dict(spec,variant='V1',tuning=False)
        rate_reference_opt=OriginalOptimizer(model,oracle_spec)
        rate_reference_opt.schedule(8.);rate_reference_opt.step()
        rate_reference={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        model.load_state_dict(state)
        after_opt.schedule(8.); after_opt.step()
        base_drift=0.; fusion_scaled_update_drift=0.; implementation_drift=0.; fusion_changed=0
        for name,p in model.named_parameters():
            actual=p.detach().cpu()
            implementation_drift=max(implementation_drift,float((actual-rate_reference[name]).abs().max()))
            if name.startswith('fusion.'):
                target=state[name]+ratio*(reference[name]-state[name])
                fusion_scaled_update_drift=max(fusion_scaled_update_drift,float((actual-target).abs().max()))
                fusion_changed+=int(not torch.equal(actual,state[name]))
            else:
                base_drift=max(base_drift,float((actual-reference[name]).abs().max()))
        assert base_drift==0., base_drift
        assert implementation_drift==0., implementation_drift
        assert fusion_changed==len(list(model.fusion.parameters()))
    finally:
        model.load_state_dict(state); model.zero_grad(set_to_none=True)
        after_opt.schedule(0.); restore_rng(saved_rng,rng); model.train(training)
        del before_opt,after_opt,rate_reference_opt; gc.collect(); torch.cuda.empty_cache()
    assert parameter_hash(model)==initial_hash
    return dict(complete=True, ratio=ratio, structural_variant='V1', same_gradient_diagnostic_steps=3,
        original_zero_learning_rate_first_step=True, schedules=schedules, base_parameter_update_max_drift=base_drift,
        fusion_update_scaled_max_drift=fusion_scaled_update_drift, fusion_tensors_updated=fusion_changed,
        same_rate_original_optimizer_max_parameter_drift=implementation_drift,
        all_parameter_groups_covered_once=True, fresh_parameters_and_rng_restored=True,
        formal_weights_reused=False, test_loaded=False)
