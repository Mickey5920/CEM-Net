"""Bounded oracle checks; preserve fresh parameters and every random state."""
import copy, gc
from scripts.cem_v1_group_rate_common import *
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng


def bounded_rate_check(model, data, spec, endpoint, seed):
    state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    initial_hash = parameter_hash(model)
    rng = np.random.default_rng(seed); saved_rng = rng_state(rng); training = model.training
    base_spec = v1_protocol('V1', 'mimic', seed)
    before_opt = OriginalOptimizer(model, base_spec)
    low_opt = OriginalOptimizer(model, dict(base_spec, fusion_lr=.00015))
    after_opt = Optimizer(model, spec)
    schedules=[]
    try:
        membership=copy.deepcopy(after_opt.values()['fusion_groups'])
        for epoch in (0., 1/1180, 1., 8., 15., 17., 30., 52.):
            before_opt.schedule(epoch); before=copy.deepcopy(before_opt.values())
            after_opt.schedule(epoch); after=copy.deepcopy(after_opt.values())
            assert before['base']==after['base'], epoch
            for role,ratio in (('encoder',1.),('readout',.5)):
                group=after['fusion_groups'][role]
                assert group['betas']==before['fusion']['betas']
                assert group['weight_decay']==before['fusion']['weight_decay']
                assert abs(group['lr']-ratio*before['fusion']['lr'])<1e-15
            schedules.append(dict(epoch=epoch, original_fusion_lr=before['fusion']['lr'],
                encoder_lr=after['fusion_groups']['encoder']['lr'],
                readout_lr=after['fusion_groups']['readout']['lr'],base_and_input_rates_exact=True))
        seed_all(seed+10000); model.train(); model.zero_grad(set_to_none=True)
        for start in (0,128):
            x,y=data.batch('train',np.arange(start,start+128))
            loss=model.loss(model(x),y[:,endpoint]);(loss*.5).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        # Two unchanged original optimizers provide per-parameter FP32 oracles.
        before_opt.schedule(8.);before_opt.step()
        reference={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        model.load_state_dict(state)
        low_opt.schedule(8.);low_opt.step()
        low_reference={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        model.load_state_dict(state)
        after_opt.schedule(8.);after_opt.step()
        unchanged_drift=0.;readout_drift=0.;changed={'encoder':0,'readout':0}
        for name,p in model.named_parameters():
            actual=p.detach().cpu()
            reduced=name.startswith('fusion.') and name[len('fusion.'):] in READOUT_NAMES
            target=low_reference[name] if reduced else reference[name]
            drift=float((actual-target).abs().max())
            if reduced:readout_drift=max(readout_drift,drift)
            else:unchanged_drift=max(unchanged_drift,drift)
            if name.startswith('fusion.'):
                changed['readout' if reduced else 'encoder']+=int(not torch.equal(actual,state[name]))
        assert unchanged_drift==readout_drift==0.,(unchanged_drift,readout_drift)
        assert all(changed[role]==membership[role]['tensors'] for role in changed)
    finally:
        model.load_state_dict(state);model.zero_grad(set_to_none=True)
        restore_rng(saved_rng,rng);model.train(training)
        del before_opt,low_opt,after_opt;gc.collect();torch.cuda.empty_cache()
    assert parameter_hash(model)==initial_hash
    return dict(complete=True,structural_variant='V1',encoder_ratio=1.,readout_ratio=.5,
        groups=membership,schedules=schedules,same_gradient_diagnostic_steps=3,
        unchanged_parameter_update_max_drift=unchanged_drift,readout_oracle_max_drift=readout_drift,
        updated_fusion_tensors=changed,all_parameter_groups_covered_once=True,
        fresh_parameters_and_rng_restored=True,formal_weights_reused=False,test_loaded=False)
