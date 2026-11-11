"""Bounded initial-function and learning-path checks; never initialize formal fits from probes."""
import gc
import torch
from models.cem_gate_location import ScalarGate
from scripts.cem_gate_start_common import Optimizer, parameter_hash
from scripts import cem_structure_runtime as runner
from scripts.train_tracehr_unified import seed_all


@torch.no_grad()
def initial_scalar_equivalence(model, data):
    assert model.fusion.gate.alpha.item()==0.
    initial_hash = parameter_hash(model)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    was_training = model.training
    gate = model.fusion.gate
    results = []
    try:
        for training in (False, True):
            model.train(training)
            x,_ = data.batch('train', list(range(128)))
            before_cpu = torch.get_rng_state().clone()
            before_cuda = torch.cuda.get_rng_state_all()
            actual = model(x)
            after_cpu = torch.get_rng_state().clone()
            after_cuda = torch.cuda.get_rng_state_all()
            model.fusion.gate = ScalarGate(gate.bias)
            torch.set_rng_state(before_cpu); torch.cuda.set_rng_state_all(before_cuda)
            reference = model(x)
            assert torch.equal(after_cpu, torch.get_rng_state())
            assert all(torch.equal(a,b) for a,b in zip(after_cuda,torch.cuda.get_rng_state_all()))
            drift = float((actual-reference).abs().max())
            assert drift==0., drift
            results.append(dict(training=training,max_logit_drift=drift,dropout_rng_exact=True))
            model.fusion.gate = gate
    finally:
        model.fusion.gate = gate
        model.train(was_training)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
    assert parameter_hash(model)==initial_hash
    return dict(complete=True,optimizer_updates=0,learned_V1_loaded=False,
        same_seed_archived_V1_retained_parameters_exact=model.info['archived_V1_retained_initialization_verified'],
        alpha_initial=0.,comparisons=results,initial_parameters_unchanged=True)


def bounded_learning_check(model, data, spec, endpoint, seed):
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash = parameter_hash(model)
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state_all()
    was_training = model.training
    opt = None
    rows = []
    try:
        seed_all(seed+10000)
        opt = Optimizer(model, spec)
        gate = model.fusion.gate
        assert sum(p is gate.alpha for g in opt.fusion.param_groups for p in g['params'])==1
        initial_weight = gate.weight.detach().clone()
        for step in range(20):
            opt.schedule(float(step)/max(1,(len(data.targets['train']['y'])+255)//256))
            ix = list(range(step*256,(step+1)*256))
            before_alpha = float(gate.alpha)
            before_weight = gate.weight.detach().clone()
            _,report = runner.update(model,opt,data,ix,128,endpoint,check=True)
            row = dict(step=step+1,learning_rate=opt.fusion.param_groups[0]['lr'],
                alpha_before=before_alpha,alpha_after=float(gate.alpha),
                alpha_gradient=float(gate.alpha.grad),weight_gradient_norm=float(gate.weight.grad.norm()),
                weight_max_update=float((gate.weight-before_weight).abs().max()),
                all_parameters_connected=all(p.grad is not None and p.requires_grad for p in model.parameters()))
            rows.append(row)
            print('X1_learning_check '+str(row),flush=True)
            assert row['all_parameters_connected'],row
            if step==0:
                assert row['alpha_before']==0. and row['weight_gradient_norm']==0.
                assert row['alpha_gradient']!=0. and row['learning_rate']==0. and row['alpha_after']==0.
        assert any(r['alpha_before']!=r['alpha_after'] for r in rows),rows
        assert any(r['weight_gradient_norm']>0. and r['weight_max_update']>0. for r in rows),rows
        assert not torch.equal(initial_weight, gate.weight)
    finally:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        del opt, initial
        gc.collect(); torch.cuda.empty_cache()
    assert parameter_hash(model)==initial_hash
    return dict(complete=True,updates=20,probe_only=True,formal_weights_reused=False,
        initial_parameters_and_rng_restored=True,alpha_in_original_fusion_optimizer=True,
        original_zero_learning_rate_first_step=True,
        first_zero_condition_gradient_expected=True,subsequent_condition_learning_verified=True,steps=rows)
