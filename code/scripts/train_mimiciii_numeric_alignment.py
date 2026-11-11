"""Six from-scratch fits with a single fixed clinical-summary preprocessing change."""
import argparse,gc,hashlib,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path[:0]=[str(deps),*[str(p) for p in deps.glob('*.whl')]]
import numpy as np
import torch
from scripts.mimiciii_numeric_alignment_common import *
from scripts.train_tracehr_unified import seed_all
from scripts.mimiciii_fixed_common import stopping


def evaluate(a,out,data):
    from scripts.tracehr_oof_common import all_scores
    job=a.fit_output.resolve(strict=True);c=read(job/'completion.json');identity=read(job/'run_manifest.json')
    assert c['complete'] and not c['pilot'] and c['variant']==VARIANT
    assert c['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
    ckpath=job/'best.pt';assert sha(ckpath)==c['checkpoint_sha256']
    model=fresh_model(data,a.preparation,a.seed,a.endpoint,out).cuda().eval()
    ck=torch.load(ckpath,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck;gc.collect()
    def statehash():
        h=hashlib.sha256()
        for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=statehash();torch.set_grad_enabled(False)
    def forbidden(*a,**kw):raise AssertionError('Test evaluation forbids weight updates')
    torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):cls.step=forbidden
    def predict(raw,summary):
        parts=[]
        for start in range(0,len(raw),256):
            x=np.array(raw[start:start+256],copy=True);x[:,data.phi:]=summary[start:start+256]
            parts.append(model.base(torch.as_tensor(x,device='cuda')).softmax(-1)[...,1].mean(1).cpu().numpy())
        return np.concatenate(parts)
    saved=dict(np.load(job/'predictions_dev.npz'));q=predict(data.raw['dev'],data.summary['dev'])
    drift=float(np.max(np.abs(q-saved['probs'][:,0])));assert drift<=1e-5
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids']);np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    inp=a.project/expected()['test_inputs']
    for n in ('test_raw.npy','test_targets.npz'):assert sha(inp/n)==expected()['test_manifest'][n]['sha256']
    raw=np.load(inp/'test_raw.npy',mmap_mode='r');target=dict(np.load(inp/'test_targets.npz'))
    for split in ('train','dev'):assert not set(target['ids'][:,0])&set(data.targets[split]['ids'][:,0])
    summary=data.transform_summary(raw[:,data.phi:]);q=predict(raw,summary);labels=target['y'][:,a.endpoint:a.endpoint+1]
    metrics=all_scores(labels,q[:,None]);np.savez_compressed(out/'predictions_test.npz',probs=q[:,None],labels=labels,ids=target['ids'])
    assert statehash()==before and sha(ckpath)==c['checkpoint_sha256']
    atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
        model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),metrics=metrics,
        dev_replay_max_drift=drift,best_epoch=c['best_epoch'],selected_using='development z0 AP only'))
    atomic(out/'progress.json',dict(status='complete',test_ap=metrics['mean_ap']))


def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--variant',choices=[VARIANT],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True);p.add_argument('--action',choices=['probe','train','evaluate'],required=True)
    p.add_argument('--expected-gib',type=float,required=True);a=p.parse_args();a.project=a.project.resolve(strict=True)
    a.preparation=a.project/expected()['input_preparation'];a.resume=False;a.lr_multiplier=1.;a.start_barrier=None
    out=a.output.resolve();assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        value=Path(os.environ[k]).resolve();assert value.is_relative_to(a.project);value.mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),output=str(out),preparation=str(a.preparation),expected_gib=a.expected_gib,
        code=str(ROOT),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],derived_summary_storage='RAM only'))
    torch.set_num_threads(4);assert torch.cuda.is_available();seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats();atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=AlignedData(a.preparation,a.project)
    if a.action=='evaluate':evaluate(a,out,data);return
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    model=fresh_model(data,a.preparation,a.seed,a.endpoint,out)
    identity=dict(database='mimiciii',variant=VARIANT,endpoint=a.endpoint,seed=a.seed,input_dim=data.input_dim,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),
        protocol=protocol(VARIANT,a.seed),initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
        parameters=sum(p.numel() for p in model.parameters()),members=8,lr_multiplier=1.,selected_structure='noEvidence',
        transform_sha256=expected()['transform_sha256'])
    atomic(out/'run_manifest.json',identity)
    from scripts import cem_v1_group_confirm_runtime as runner
    from scripts.tracehr_noa_ablation_common import Optimizer,diagnostics
    from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
    runner.Optimizer,runner.diagnostics,runner.stopping=Optimizer,diagnostics,stopping
    model.cuda();gc.collect();atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    if a.action=='probe':runner.probe(a,out,data,model,identity,identity['protocol'])
    else:runner.train(a,out,data,model,identity,identity['protocol'])


if __name__=='__main__':main()
