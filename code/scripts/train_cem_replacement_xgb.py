"""XGBoost-TD, dense complete evidence, AP-only selection and locked tests."""
import argparse,os,resource,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from scripts.cem_replacement_common import expected,raw_data,xgb_recipe,read,sha,atomic,RUN
from scripts.tracehr_oof_common import all_scores

def predict(model,raw):
    return np.concatenate([model.inplace_predict(raw[i:i+1024]) for i in range(0,len(raw),1024)])

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,choices=(48,49,50),required=True);p.add_argument('--endpoint',type=int,choices=(0,1),required=True)
    p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);out=a.output.resolve();c=expected()
    assert dict(variant=a.variant,seed=a.seed,endpoint=a.endpoint) in c['matrix']
    assert c['database']==a.database and a.variant=='XGBoost_TD' and xgb.__version__=='3.2.0'
    assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(10+a.expected_gib)*2**30
    for k in ('TMPDIR','XDG_CACHE_HOME'):assert Path(os.environ[k]).resolve().is_relative_to(a.project)
    out.mkdir(parents=True);tick=time.time()
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(out/'progress.json',dict(status='loading_verified_inputs'));data=raw_data(a.project);params=xgb_recipe(a.seed)
    identity=dict(database=a.database,variant=a.variant,seed=a.seed,endpoint=a.endpoint,input_dim=data.input_dim,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_contract_sha256=sha(ROOT/'work'/RUN/'config.json'),
        protocol=dict(params=params,max_rounds=1000,early_stopping_rounds=300,selection='validation AP only',
            recipe='pytabkit XGB_TD_CLASS',class_weight=None,missing='NaN only; dense zero is an observed value'),package_version=xgb.__version__)
    atomic(out/'run_manifest.json',identity)
    if a.action=='evaluate':
        fit=a.fit_output;done=read(fit/'completion.json');lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
        assert lock['complete'] and not lock['test_used_for_selection'] and len(lock['selections'])==len(c['matrix'])
        assert identity==read(fit/'run_manifest.json');checkpoint=fit/'best.ubj';before=sha(checkpoint)
        assert before==done['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
        model=xgb.Booster();model.load_model(checkpoint);model.set_param({'nthread':4,'device':'cpu'})
        saved=dict(np.load(fit/'predictions_dev.npz'))
        np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids']);np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
        drift=float(np.max(np.abs(predict(model,data.raw['dev'])-saved['probs'][:,0])));assert drift==0
        state=bytes(model.save_raw());data.test();start=time.time();q=predict(model,data.raw['test'])
        assert bytes(model.save_raw())==state and sha(checkpoint)==before
        y=data.targets['test']['y'][:,a.endpoint:a.endpoint+1]
        np.savez_compressed(out/'predictions_test.npz',ids=data.targets['test']['ids'],labels=y,probs=q[:,None])
        atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,tree_updates=0,parameters_updated=False,
            model_sha256=before,dev_replay_max_drift=drift,predictions_sha256=sha(out/'predictions_test.npz'),metrics=all_scores(y,q[:,None]),
            inference_seconds=time.time()-start,rows=len(q),test_loaded=True))
    else:
        if a.action=='train':
            r=read(a.pilot_record/'completion.json');audit=read(a.pilot_record/'path_audit.json');ref=r['identity']
            assert r['complete'] and r['pilot'] and audit['reload_prediction_exact']
            assert ref['variant']==a.variant and ref['endpoint']==a.endpoint and ref['input_dim']==data.input_dim
            assert ref['source_manifest_sha256']==identity['source_manifest_sha256'] and ref['package_version']==xgb.__version__
            atomic(out/'pilot_identity_check.json',dict(complete=True,from_scratch=True,pilot_weights_loaded=False,resource_path_seed=48,seed=a.seed))
        atomic(out/'initialization_observation.json',dict(complete=True,from_scratch=True,tree_updates=0,learned_checkpoint_loaded=False,**identity))
        ytrain=data.targets['train']['y'][:,a.endpoint];ydev=data.targets['dev']['y'][:,a.endpoint]
        assert set(np.unique(ytrain))=={0,1} and set(np.unique(ydev))=={0,1}
        atomic(out/'progress.json',dict(status='preparing_inputs',input_dim=data.input_dim,train_rows=len(ytrain)))
        train=xgb.QuantileDMatrix(data.raw['train'],label=ytrain,max_bin=256,nthread=4)
        valid=xgb.QuantileDMatrix(data.raw['dev'],label=ydev,max_bin=256,ref=train,nthread=4)
        history=[]
        def metric(q,d):return 'AP',float(average_precision_score(d.get_label(),q))
        class Progress(xgb.callback.TrainingCallback):
            def after_iteration(self,model,epoch,log):
                ap=float(log['validation']['AP'][-1]);history.append(dict(iteration=epoch+1,dev_ap=ap,elapsed_seconds=time.time()-tick))
                atomic(out/'progress.json',dict(status='training',iteration=epoch+1,dev_ap=ap,elapsed_seconds=time.time()-tick));return False
        pilot=a.action=='probe';callbacks=[Progress()]
        if not pilot:callbacks.append(xgb.callback.EarlyStopping(rounds=300,metric_name='AP',data_name='validation',maximize=True,save_best=False))
        model=xgb.train(params,train,num_boost_round=20 if pilot else 1000,evals=[(valid,'validation')],custom_metric=metric,maximize=True,callbacks=callbacks,verbose_eval=False)
        best=20 if pilot else int(model.best_iteration)+1
        if not pilot:assert best==1+int(np.argmax([r['dev_ap'] for r in history]))
        model=model[:best];q=predict(model,data.raw['dev']);checkpoint=out/'best.ubj';model.save_model(checkpoint)
        reloaded=xgb.Booster();reloaded.load_model(checkpoint);reloaded.set_param({'nthread':4,'device':'cpu'})
        replay=predict(reloaded,data.raw['dev']);drift=float(np.max(np.abs(q-replay)));assert drift==0
        np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=ydev[:,None],probs=replay[:,None]);atomic(out/'history.json',history)
        if pilot:
            assert len(history)==20 and model.num_boosted_rounds()==20
            atomic(out/'path_audit.json',dict(complete=True,continuous_tree_updates=20,reload_prediction_exact=True,
                full_training_rows=len(ytrain),full_validation_rows=len(ydev),input_dim=data.input_dim,from_scratch=True,test_loaded=False,
                seconds_per_iteration=(time.time()-tick)/20,selection_metric='AP',cpu_threads=4))
        metrics=all_scores(ydev[:,None],replay[:,None])
        atomic(out/'completion.json',dict(complete=True,pilot=pilot,identity=identity,**identity,from_scratch=True,test_loaded=False,
            best_epoch=best,epochs=len(history),dev_ap=metrics['mean_ap'],metrics=metrics,checkpoint_file='best.ubj',checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/'predictions_dev.npz'),checkpoint_replay=dict(complete=True,max_probability_drift=0.),
            seconds=time.time()-tick,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_reserved_mib=0.,
            cpu_threads=4,stopping='1000 rounds or 300 without validation AP improvement',finished_at=time.time()))
    atomic(out/'progress.json',dict(status='complete',seconds=time.time()-tick))

if __name__=='__main__':main()
