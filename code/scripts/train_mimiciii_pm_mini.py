"""Reuse the accepted training loops; no test input is loaded during fitting."""
import argparse, gc, hashlib, os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path[:0] = [str(deps), *[str(p) for p in deps.glob('*.whl')]]
import numpy as np
import torch
from scripts.mimiciii_pm_mini_common import *
from scripts.mimiciii_fixed_common import Data, model_for, parameter_hash, stopping
from scripts.train_mimiciii_numeric_baseline import NoEvidence
from scripts import train_mimiciii_numeric_baseline as previous
from scripts.train_tracehr_unified import seed_all
from models.mimiciii_pm_mini import delete_component


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n, p in model.named_parameters()}


def fresh_deletion(a, data, out):
    full = model_for(data, a.preparation, a.endpoint, a.seed)
    model = NoEvidence(full)
    del full
    ref = reference(a.variant, a.seed, a.endpoint)
    assert sha(Path(ref['output'])/'initialization_observation.json') == ref['initialization_sha256']
    obs = read(Path(ref['output'])/'initialization_observation.json')
    hashes = tensor_hashes(model)
    assert hashes == obs['retained_tensor_hashes']
    assert parameter_hash(model) == ref['initial_weights_sha256']
    model, mapping = delete_component(model, a.variant[-1])
    # Store observation before any assertion depending on this run's output.
    atomic(out/'initialization_observation.json', dict(complete=True, optimizer_updates=0,
        reference_main_initial_sha256=ref['initial_weights_sha256'], reference_file_sha256=ref['initialization_sha256'],
        initial_weights_sha256=parameter_hash(model), retained_tensor_hashes=tensor_hashes(model), deletion=mapping))
    atomic(out/'connection_audit.json', dict(complete=True, reference_seed=a.seed, retained_CEM_tensors_exact=True,
        learned_checkpoint_loaded=False, evidence_branch_removed=True, auxiliary_enabled=False,
        all_retained_parameters_trainable=all(p.requires_grad for p in model.parameters()), main_loss_coefficient=.5,
        deletion=mapping))
    return model


def fit_deletion(a, out, data, identity):
    from scripts import cem_v1_group_confirm_runtime as runner
    from scripts.tracehr_noa_ablation_common import Optimizer, diagnostics
    from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
    runner.Optimizer, runner.diagnostics, runner.stopping = Optimizer, diagnostics, stopping
    model = fresh_deletion(a, data, out)
    identity.update(initial_weights_sha256=parameter_hash(model), common_weights_sha256=parameter_hash(model, True),
        parameters=sum(p.numel() for p in model.parameters()), members=8, lr_multiplier=1., selected_structure=a.variant)
    atomic(out/'run_manifest.json', identity)
    model.cuda();gc.collect()
    atomic(out/'G0_equivalence.json', initial_equivalence(model, data))
    if a.action == 'probe':
        runner.probe(a, out, data, model, identity, identity['protocol'])
        result = read(out/'completion.json')
        groups = result['gradient_checks']['128']['loss_paths']['groups']
        assert all(g['main_connected'] == g['tensors'] and g['aux_connected'] == 0 for g in groups.values()), groups
    else:
        runner.train(a, out, data, model, identity, identity['protocol'])


def fit_mini(a, out, data, identity):
    from tabm import TabM
    original = TabM.make
    ref = reference(a.variant, a.seed, a.endpoint)
    def mini_make(*args, **kwargs):
        assert kwargs == dict(n_num_features=data.input_dim, d_out=1, k=32, d_block=256, n_blocks=3, dropout=.2)
        # Check the original API/default initializer before adding the sole architecture change.
        with torch.random.fork_rng(devices=[]):
            baseline = original(*args, **kwargs)
            baseline_hash = parameter_hash(baseline)
            assert baseline_hash == ref['initial_weights_sha256']
            del baseline
        model = original(*args, **dict(kwargs, arch_type='tabm-mini'))
        assert model.backbone.__class__.__name__ != 'MLPBackboneBatchEnsemble'
        atomic(out/'architecture_audit.json', dict(complete=True, official_package='tabm',
            original_TabM_initial_weights_sha256=baseline_hash, arguments=dict(kwargs, arch_type='tabm-mini'),
            baseline_arguments=kwargs, retained_recipe=True, architecture_class=type(model.backbone).__name__))
        return model
    TabM.make = mini_make
    try:
        previous.tabm(a, out, data, identity)
    finally:
        TabM.make = original


def evaluate(a, out, data):
    from scripts.tracehr_oof_common import all_scores
    from scripts.train_tracehr_native_baseline import tabm_predict
    job = a.fit_output.resolve(strict=True)
    c = read(job/'completion.json');identity = read(job/'run_manifest.json')
    assert c['complete'] and not c['pilot'] and c['variant'] == a.variant and c['seed'] == a.seed and c['endpoint'] == a.endpoint
    assert c['source_manifest_sha256'] == sha(ROOT/'source_manifest.json')
    ckpath = job/c.get('checkpoint_file', 'best.pt');assert sha(ckpath) == c['checkpoint_sha256']
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    if a.variant == 'TabM-mini':
        import joblib
        from tabm import TabM
        model = TabM.make(n_num_features=data.input_dim, d_out=1, k=32, d_block=256, n_blocks=3, dropout=.2, arch_type='tabm-mini').cuda()
        model.load_state_dict(ck['model_state_dict'])
        transform = joblib.load(a.project/'runs/mimiciii_baselines_20261024/preparation/tabm_preprocessing.joblib')
        def predict(raw):
            z = np.array(raw, copy=True);d = transform['phi_dim']
            for start in range(0, len(z), 1024):
                block = z[start:start+1024, :d];v = (block-transform['phi_center'])*transform['phi_factor'];v /= np.sqrt(1+(v/3)**2)
                v[:, transform['binary']] = block[:, transform['binary']];z[start:start+1024, :d] = v
            z[:, d:] = transform['numeric_quantile'].transform(z[:, d:])
            return tabm_predict(model, z)
    else:
        model = fresh_deletion(a, data, out).cuda();model.load_state_dict(ck['model'])
        def predict(raw):
            model.eval();parts=[]
            for start in range(0, len(raw), 256):
                x = torch.as_tensor(np.array(raw[start:start+256]), device='cuda')
                parts.append(model.base(x).softmax(-1)[..., 1].mean(1).cpu().numpy())
            return np.concatenate(parts)
    del ck;gc.collect()
    def statehash():
        h=hashlib.sha256()
        for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before = statehash()
    torch.set_grad_enabled(False)
    def forbidden(*args, **kwargs):raise AssertionError('Test evaluation forbids optimizer updates and backward')
    torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):cls.step=forbidden
    saved = dict(np.load(job/'predictions_dev.npz'))
    q = predict(data.raw['dev']);drift = float(np.max(np.abs(q-saved['probs'][:,0])));assert drift <= 1e-5
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids'])
    np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    inp=a.project/expected()['test_inputs'];manifest=expected()['test_manifest']
    for n in ('test_raw.npy','test_targets.npz'):
        assert sha(inp/n)==manifest[n]['sha256']
    raw=np.load(inp/'test_raw.npy',mmap_mode='r');target=dict(np.load(inp/'test_targets.npz'))
    for split in ('train','dev'):assert not set(target['ids'][:,0])&set(data.targets[split]['ids'][:,0])
    q=predict(raw);labels=target['y'][:,a.endpoint:a.endpoint+1]
    metrics=all_scores(labels,q[:,None]);np.savez_compressed(out/'predictions_test.npz',probs=q[:,None],labels=labels,ids=target['ids'])
    assert statehash()==before and sha(ckpath)==c['checkpoint_sha256']
    atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
        model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),metrics=metrics,
        dev_replay_max_drift=drift,best_epoch=c['best_epoch'],selected_using='development AP only',training_output=str(job)))
    atomic(out/'progress.json',dict(status='complete',test_ap=metrics['mean_ap']))


def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True);p.add_argument('--action',choices=['probe','train','evaluate'],required=True)
    p.add_argument('--expected-gib',type=float,required=True);a=p.parse_args();a.project=a.project.resolve(strict=True)
    a.preparation=a.project/'runs/mimiciii_adaptation_20261024/v1/preparation';a.resume=False;a.lr_multiplier=1.;a.start_barrier=None
    out=a.output.resolve();assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        value=Path(os.environ[k]).resolve();assert value.is_relative_to(a.project);value.mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),output=str(out),preparation=str(a.preparation),expected_gib=a.expected_gib,
        code=str(ROOT),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR']))
    torch.set_num_threads(4);assert torch.cuda.is_available();seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats();atomic(out/'progress.json',dict(status='loading_verified_inputs'));data=Data(a.preparation)
    identity=dict(database='mimiciii',variant=a.variant,endpoint=a.endpoint,seed=a.seed,input_dim=data.input_dim,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),protocol=protocol(a.variant,a.seed))
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    if a.action=='evaluate':evaluate(a,out,data)
    elif a.variant=='TabM-mini':fit_mini(a,out,data,identity)
    else:fit_deletion(a,out,data,identity)


if __name__=='__main__':main()
