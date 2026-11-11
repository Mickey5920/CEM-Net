"""LightGBM-TD classification recipe, validation AP selection, locked tests."""
import argparse, gc, os, resource, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from scripts.cem_icde72_common import RawData, expected, lightgbm_recipe, read, sha, atomic
from scripts.tracehr_oof_common import all_scores

def predict(model,raw):
    return np.concatenate([model.predict(raw[i:i+1024],num_threads=4) for i in range(0,len(raw),1024)])

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,choices=(48,49,50),required=True);p.add_argument('--endpoint',type=int,choices=(0,1),required=True)
    p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);out=a.output.resolve();c=expected()
    assert dict(variant=a.variant,seed=a.seed,endpoint=a.endpoint) in c['matrix']
    assert c['database']==a.database and lgb.__version__=='4.6.0'
    assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(10+a.expected_gib)*2**30
    for k in ('TMPDIR','XDG_CACHE_HOME'):
        assert Path(os.environ[k]).resolve().is_relative_to(a.project)
    out.mkdir(parents=True);tick=time.time()
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=RawData(a.project);params=lightgbm_recipe(a.seed)
    identity=dict(database=a.database,variant=a.variant,seed=a.seed,endpoint=a.endpoint,input_dim=data.input_dim,
                  source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_contract_sha256=sha(ROOT/'work/cem_icde_evidence_20261028/config.json'),
                  protocol=dict(params=params,max_rounds=1000,early_stopping_rounds=300,selection='validation AP only',
                                recipe='pytabkit LGBM_TD_CLASS',class_weight=None),package_version=lgb.__version__)
    atomic(out/'run_manifest.json',identity)
    if a.action=='evaluate':
        fit=a.fit_output;done=read(fit/'completion.json');lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
        assert lock['complete'] and not lock['test_used_for_selection'] and len(lock['selections'])==len(c['matrix'])
        assert identity==read(fit/'run_manifest.json')
        checkpoint=fit/'best.txt';before=sha(checkpoint)
        assert before==done['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
        model=lgb.Booster(model_file=str(checkpoint));saved=dict(np.load(fit/'predictions_dev.npz'))
        np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids'])
        np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
        drift=float(np.max(np.abs(predict(model,data.raw['dev'])-saved['probs'][:,0])));assert drift<=1e-12
        state=model.model_to_string();data.test();start=time.time();q=predict(model,data.raw['test'])
        assert model.model_to_string()==state and sha(checkpoint)==before
        y=data.targets['test']['y'][:,a.endpoint:a.endpoint+1]
        np.savez_compressed(out/'predictions_test.npz',ids=data.targets['test']['ids'],labels=y,probs=q[:,None])
        atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,tree_updates=0,parameters_updated=False,
              model_sha256=before,dev_replay_max_drift=drift,predictions_sha256=sha(out/'predictions_test.npz'),metrics=all_scores(y,q[:,None]),
              inference_seconds=time.time()-start,rows=len(q),test_loaded=True))
    else:
        if a.action=='train':
            r=read(a.pilot_record/'completion.json');audit=read(a.pilot_record/'path_audit.json')
            assert r['complete'] and r['pilot'] and audit['reload_prediction_exact']
            ref=r['identity'];assert ref['variant']==a.variant and ref['endpoint']==a.endpoint
            assert ref['source_manifest_sha256']==identity['source_manifest_sha256']
            assert ref['input_dim']==data.input_dim and ref['package_version']==lgb.__version__
            atomic(out/'pilot_identity_check.json',dict(complete=True,from_scratch=True,pilot_weights_loaded=False,resource_path_seed=48,seed=a.seed))
        atomic(out/'initialization_observation.json',dict(complete=True,from_scratch=True,tree_updates=0,learned_checkpoint_loaded=False,**identity))
        ytrain=data.targets['train']['y'][:,a.endpoint];ydev=data.targets['dev']['y'][:,a.endpoint]
        assert set(np.unique(ytrain))=={0,1} and set(np.unique(ydev))=={0,1}
        train=lgb.Dataset(data.raw['train'],label=ytrain,params=params,free_raw_data=True)
        valid=lgb.Dataset(data.raw['dev'],label=ydev,reference=train,free_raw_data=True)
        history=[]
        def metric(q,d):return 'AP',float(average_precision_score(d.get_label(),q)),True
        def progress(env):
            ap=float(env.evaluation_result_list[0][2]);history.append(dict(iteration=env.iteration+1,dev_ap=ap,elapsed_seconds=time.time()-tick))
            atomic(out/'progress.json',dict(status='training',iteration=env.iteration+1,dev_ap=ap,elapsed_seconds=time.time()-tick,
                                           active_iteration_limit=20 if a.action=='probe' else 1000))
        progress.order=10
        callbacks=[progress]
        if a.action=='train':callbacks.append(lgb.early_stopping(300,first_metric_only=True,verbose=False,min_delta=0.))
        atomic(out/'progress.json',dict(status='preparing_inputs',input_dim=data.input_dim,train_rows=len(ytrain)))
        model=lgb.train(params,train,num_boost_round=20 if a.action=='probe' else 1000,valid_sets=[valid],valid_names=['validation'],feval=metric,callbacks=callbacks)
        best=model.best_iteration or model.current_iteration()
        q=model.predict(data.raw['dev'],num_iteration=best,num_threads=4)
        checkpoint=out/'best.txt';model.save_model(str(checkpoint),num_iteration=best)
        replay=predict(lgb.Booster(model_file=str(checkpoint)),data.raw['dev'])
        drift=float(np.max(np.abs(q-replay)));assert drift<=1e-12
        if a.action=='train':assert best==1+int(np.argmax([r['dev_ap'] for r in history]))
        np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=ydev[:,None],probs=replay[:,None])
        atomic(out/'history.json',history)
        metrics=all_scores(ydev[:,None],replay[:,None]);pilot=a.action=='probe'
        if pilot:
            assert len(history)==20 and model.num_trees()==20
            atomic(out/'path_audit.json',dict(complete=True,continuous_tree_updates=20,reload_prediction_exact=drift==0,
                 full_training_rows=len(ytrain),full_validation_rows=len(ydev),input_dim=data.input_dim,from_scratch=True,test_loaded=False,
                 seconds_per_iteration=(time.time()-tick)/20,selection_metric='AP',cpu_threads=4))
        atomic(out/'completion.json',dict(complete=True,pilot=pilot,identity=identity,**identity,from_scratch=True,test_loaded=False,
              best_epoch=best,epochs=len(history),dev_ap=metrics['mean_ap'],metrics=metrics,checkpoint_file='best.txt',checkpoint_bytes=checkpoint.stat().st_size,
              checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/'predictions_dev.npz'),checkpoint_replay=dict(complete=True,max_probability_drift=drift),
              seconds=time.time()-tick,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_reserved_mib=0.,
              cpu_threads=4,stopping='1000 rounds or 300 rounds without validation AP improvement',finished_at=time.time()))
    atomic(out/'progress.json',dict(status='complete',seconds=time.time()-tick))

if __name__=='__main__':main()
