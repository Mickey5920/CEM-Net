"""Zero-update, predeclared diagnostics of existing full and deletion checkpoints."""
import argparse, gc, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path[:0]=[str(deps),*[str(x) for x in deps.glob('*.whl')]]
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from scripts.ema_internal_common import *
from scripts.mimiciii_ema_runtime import digest
from models.mimiciii_pm_mini import delete_component
from scripts.train_tracehr_unified import seed_all

def scores(y,p):
    if len(np.unique(y))<2:return dict(n=len(y),positive=int(y.sum()),AP=None,AUROC=None,Brier=None,log_loss=None)
    return dict(n=len(y),positive=int(y.sum()),AP=float(average_precision_score(y,p)),
        AUROC=float(roc_auc_score(y,p)),Brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))

def evidence(raw,nv,phi):
    # Event presence and laboratory-observed indicators are original feature semantics.
    present=np.asarray(raw[:,nv:2*nv]);labs=np.asarray(raw[:,phi+11*np.arange(32)])
    assert np.all((present==0)|(present==1)) and np.all((labs==0)|(labs==1))
    return dict(event_type_coverage=present.sum(1),observed_labs=labs.sum(1))

@torch.no_grad()
def members(model,raw):
    model.eval();values=[]
    for start in range(0,len(raw),256):
        x=torch.as_tensor(np.array(raw[start:start+256],copy=True),device='cuda')
        values.append(model.base(x).softmax(-1)[...,1].cpu().numpy())
    return np.concatenate(values)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);out=a.output.resolve()
    assert out.is_relative_to(a.project) and not out.exists();out.mkdir(parents=True)
    c=expected();prep=a.project/c['input_preparation'];started=time.time()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    atomic(out/'progress.json',dict(status='loading_verified_inputs',optimizer_updates=0))
    data=Data(a.project,prep);phi=data.record['phi_dim']
    if database()=='eicu':
        from scripts.tracehr_multiview_common import inputs
        schema=inputs(a.project,'eicu')['phi']/'completion.json'
        original=read(Path(data.record['input_pointer']['path'])/'completion.json')
        assert sha(schema)==original['identity']['inputs']['phi']
        record=read(schema);nv=record['num_types'];assert record['widths']['phi']==phi
    else:nv=data.record['event_num_types']
    traincov=evidence(data.raw['train'],nv,phi)
    boundaries={k:np.quantile(v,[1/3,2/3]).tolist() for k,v in traincov.items()}
    inp=a.project/c['test_inputs']
    for n in ('test_raw.npy','test_targets.npz'):assert sha(inp/n)==c['test_manifest'][n]['sha256']
    raw=dict(data.raw,test=np.load(inp/'test_raw.npy',mmap_mode='r'))
    targets=dict(data.targets,test=dict(np.load(inp/'test_targets.npz')))
    for split in ('train','dev'):assert not set(targets['test']['ids'][:,0])&set(targets[split]['ids'][:,0])
    cov={s:evidence(raw[s],nv,phi) for s in ('dev','test')}
    records=[];curves=[];group_pairs=[];allpred={}
    for ref in c['diagnostic_checkpoints']:
        s,e=ref['seed'],ref['endpoint'];label=ref['label'];folder=Path(ref['output']);seed_all(s)
        model=full_model(data,prep,s,e,out)
        if label!='fullEMA':model,_=delete_component(model,'M' if database()=='eicu' else 'P')
        ckpath=folder/'best.pt';assert sha(ckpath)==ref['checkpoint_sha256']
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert ck['role']=='EMA_selected_inference';model.load_state_dict(ck['model'],strict=True);del ck
        model.cuda().eval();before=digest(model.state_dict())
        for split in ('dev','test'):
            z=members(model,raw[split]);q=z.mean(1);y=targets[split]['y'][:,e]
            oldpath=Path(ref['predictions'][split]);assert sha(oldpath)==c['protected'][str(oldpath)]
            saved=dict(np.load(oldpath));np.testing.assert_array_equal(saved['ids'],targets[split]['ids'])
            np.testing.assert_array_equal(saved['labels'][:,0],y)
            drift=float(np.max(np.abs(q-saved['probs'][:,0])));assert drift<=1e-5
            aps=[float(average_precision_score(y,z[:,k])) for k in range(8)]
            corr=np.corrcoef((z-y[:,None]).T);off=corr[np.triu_indices(8,1)]
            groups={axis:[dict(group=g,**scores(y[np.digitize(value,boundaries[axis],right=True)==g],q[np.digitize(value,boundaries[axis],right=True)==g]))
                for g in range(3)] for axis,value in cov[split].items()}
            records.append(dict(method=label,seed=s,endpoint=e,split=split,metrics=scores(y,q),member_AP=aps,
                mean_minus_best_member_AP=float(average_precision_score(y,q)-max(aps)),
                mean_member_probability_SD=float(z.std(1).mean()),mean_pairwise_error_correlation=float(off.mean()),
                replay_max_drift=drift,groups=groups))
            allpred[(label,s,e,split)]=q
        assert digest(model.state_dict())==before and sha(ckpath)==ref['checkpoint_sha256']
        history=read(folder/'history.json');best=int(np.argmax([v['dev_ap'] for v in history]));window=history[max(0,best-5):min(len(history),best+6)]
        curves.append(dict(method=label,seed=s,endpoint=e,epochs=len(history),best_epoch=best+1,
            best_AP=history[best]['dev_ap'],last_AP=history[-1]['dev_ap'],
            neighborhood_radius=5,neighborhood_mean_AP=float(np.mean([r['dev_ap'] for r in window])),
            neighborhood_SD_AP=float(np.std([r['dev_ap'] for r in window],ddof=1)),
            same_epoch_ema_minus_raw_AP=history[best].get('same_epoch_ema_minus_raw_AP'),
            dev_AP_curve=[r['dev_ap'] for r in history]))
        del model;gc.collect();torch.cuda.empty_cache()
        atomic(out/'progress.json',dict(status='read_only_analysis',completed_checkpoints=len(curves),total_checkpoints=12,optimizer_updates=0))
    paired=[];other='noM' if database()=='eicu' else 'noP'
    for split in ('dev','test'):
        for e in (0,1):
            for s in SEEDS:
                full=allpred[('fullEMA',s,e,split)];deleted=allpred[(other,s,e,split)];y=targets[split]['y'][:,e]
                paired.append(dict(split=split,seed=s,endpoint=e,full_minus_deletion={k:scores(y,full)[k]-scores(y,deleted)[k] for k in ('AP','AUROC','Brier','log_loss')},
                    mean_absolute_probability_difference=float(np.abs(full-deleted).mean()),
                    prediction_correlation=float(np.corrcoef(full,deleted)[0,1])))
                for axis,value in cov[split].items():
                    bins=np.digitize(value,boundaries[axis],right=True)
                    for g in range(3):
                        ix=bins==g;f=scores(y[ix],full[ix]);d=scores(y[ix],deleted[ix])
                        group_pairs.append(dict(split=split,seed=s,endpoint=e,axis=axis,group=g,n=int(ix.sum()),positive=int(y[ix].sum()),
                            delta_AP=None if f['AP'] is None else f['AP']-d['AP']))
    same_epoch=[]
    for e in (0,1):
        for s in SEEDS:
            f=next(v for v in curves if v['seed']==s and v['endpoint']==e and v['method']=='fullEMA')
            d=next(v for v in curves if v['seed']==s and v['endpoint']==e and v['method']==other)
            assert f['epochs']==d['epochs'];delta=np.array(f['dev_AP_curve'])-d['dev_AP_curve']
            same_epoch.append(dict(seed=s,endpoint=e,epochs=len(delta),full_better_epochs=int((delta>0).sum()),
                mean_delta_AP=float(delta.mean()),at_full_selected_epoch=float(delta[f['best_epoch']-1]),
                at_deletion_selected_epoch=float(delta[d['best_epoch']-1])))
    assert len(records)==24
    result=dict(complete=True,database=database(),records=records,curves=curves,paired=paired,group_pairs=group_pairs,
        common_epoch_dev_comparison=same_epoch,group_boundaries_train_only=boundaries,
        selection_unchanged=True,optimizer_updates=0,checkpoints_unchanged=True,patient_records_exported=False,
        limitations=['Descriptive frozen-model diagnostics do not identify a causal generalization mechanism.',
        'Single-member maxima are descriptive references only; inference always uses the fixed eight-member mean.',
        'No test-driven output, member, checkpoint, group threshold or training recipe selection.'])
    atomic(out/'analysis.json',result)
    atomic(out/'completion.json',dict(complete=True,records=24,optimizer_updates=0,checkpoints_unchanged=True,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),analysis_sha256=sha(out/'analysis.json'),seconds=time.time()-started))
    atomic(out/'progress.json',dict(status='complete',optimizer_updates=0,completed_checkpoints=12))

if __name__=='__main__':main()
