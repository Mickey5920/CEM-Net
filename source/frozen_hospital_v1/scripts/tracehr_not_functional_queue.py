"""Four functional ablations using the proven centralized admission/exit machinery."""
import argparse,os,sys,time,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_not_confirmation_queue import Queue as PreviousQueue
from scripts.tracehr_recovery_queue import atomic,read,sha,process_identity
from scripts.tracehr_not_functional_common import RUN,MATRIX,protocol


class Queue(PreviousQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl
        self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN/'mimic'
        self.temp=self.project/'work'/RUN/'tmp/mimic'
        self.preparation=self.project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation'
        for p in (self.source,self.out,self.cache,self.temp,self.preparation):
            assert p.resolve().is_relative_to(self.project) and p!=self.project
        for p in (self.out,self.cache,self.temp):p.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.locks,self.handles={},{}
        for name,digest in read(self.source/'source_manifest.json').items():assert sha(self.source/name)==digest
        assert a.database=='mimic' and not a.resume_queue and not (self.out/'state.json').exists()
        previous=self.project/'runs/tracehr_not_confirmation_20261017/mimic/confirm_validation_v2'
        assert read(previous/'screen/state.json')['status']=='complete'
        assert read(previous/'start_exit.json')['returncode']==0
        self.state=dict(database='mimic',pid=os.getpid(),status='preflight',phase='probes',jobs={},
            source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),
            expected_gib=a.expected_gib,gpus=[1,0,4],slots_per_gpu=2,formal_job_limit=4,
            coexistence_verified=True,max_workers_per_gpu=2,from_scratch=True,test_loaded=False,
            hyperparameter_search=False,min_free_gpu_mib=8192,eicu_authorized=False,reused={},
            admission_rule='User authorized exactly noA/noR x death/LOS seed42; no extensions')
        for v,e,s in MATRIX:
            for pilot in (True,False):
                name=('probe_' if pilot else '')+f'{v}_t{e}_s{s}'
                self.state['jobs'][name]=dict(id=name,variant=v,endpoint=e,seed=s,pilot=pilot,
                    action='probe' if pilot else 'train',status='pending',
                    output=str(self.out/('pilots' if pilot else 'jobs')/name))
        self.verify_controls();self.save()

    def verify_controls(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        summary=read(self.project/'runs/tracehr_not_confirmation_20261017/results_v1/aggregates.json')
        assert summary['complete'] and summary['protected_files_unchanged'] and not summary['test_loaded']
        prep=read(self.preparation/'completion.json');folder=Path(prep['input_pointer']['path'])
        assert sha(folder/'dev_targets.npz')==prep['input_artifacts']['dev_targets.npz']['sha256']
        target=dict(np.load(folder/'dev_targets.npz'))
        for v in ('C5_noT','B0','C5_noT_noP','C5_noT_noM'):
            for e in (0,1):
                key=f'{v}_t{e}_s42';r=summary['records'][key];path=Path(r['source']);c=read(path/'completion.json')
                assert read(path.parent.parent/(path.name+'_exit.json'))['returncode']==0
                assert c['complete'] and c['from_scratch'] and not c['test_loaded'] and c['seed']==42 and c['endpoint']==e
                assert sha(path/c.get('checkpoint_file','best.pt'))==r['checkpoint_sha256']==c['checkpoint_sha256']
                assert sha(path/'predictions_dev.npz')==r['predictions_sha256']==c['predictions_sha256']
                pr=dict(np.load(path/'predictions_dev.npz'))
                assert np.array_equal(pr['ids'],target['ids']) and np.array_equal(pr['labels'],target['y'][:,e:e+1])
                ap=float(average_precision_score(pr['labels'][:,0],pr['probs'][:,0]));assert abs(ap-r['AP'])<1e-12
                self.state['reused'][key]=dict(path=str(path),dev_ap=ap,best_epoch=c['best_epoch'],
                    initial_weights_sha256=c['initial_weights_sha256'],checkpoint_sha256=c['checkpoint_sha256'],
                    predictions_sha256=c['predictions_sha256'],source_manifest_sha256=c['source_manifest_sha256'])
        atomic(self.out/'reused_controls.json',self.state['reused'])

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==4 and {(j['variant'],j['endpoint'],j['seed']) for j in formal}==set(MATRIX)
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):return protocol(j['variant'],'mimic',j['seed'])

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-B',str(self.source/'scripts/train_tracehr_not_functional.py'),
             '--project',str(self.project),'--database','mimic','--output',str(out),'--preparation',str(self.preparation),
             '--variant',j['variant'],'--seed',str(j['seed']),'--endpoint',str(j['endpoint']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train':cmd+=['--pilot-record',self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}_s{j['seed']}"]['output']]
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        r=super().validate(j)
        edge=read(Path(j['output'])/'connection_audit.json')
        assert edge['main_loss_coefficient']==.5
        assert edge['auxiliary_enabled']==(j['variant']=='C5_noT_noR')
        assert edge['main_residual_enabled']==(j['variant']=='C5_noT_noA')
        assert edge['auxiliary_loss_coefficient']==(.5 if edge['auxiliary_enabled'] else 0.)
        assert edge['parameter_count_removed']==(514 if j['variant']=='C5_noT_noA' else 0)
        return r

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):
            self.state.update(status='needs_attention',error='Attempt failed; no recipe change or automatic retry');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            probes=[j['result'] for j in jobs if j['pilot']]
            estimate=sum(x['checkpoint_bytes'] for x in probes)*11/2**30+.5
            assert estimate<self.a.expected_gib
            self.state.update(projected_artifact_gib=estimate,phase='training',status='running',
                min_free_gpu_mib=math.ceil(max(x['peak_reserved_mib'] for x in probes)*1.35+1536))
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            records={j['id']:j['result'] for j in jobs if not j['pilot']}
            effects={key:dict(dev_ap=r['dev_ap'],
                full_noT_minus_ablation=self.state['reused'][f"C5_noT_t{r['endpoint']}_s42"]['dev_ap']-r['dev_ap'],
                delta_vs_B0=r['dev_ap']-self.state['reused'][f"B0_t{r['endpoint']}_s42"]['dev_ap']) for key,r in records.items()}
            atomic(self.out/'phase_summary.json',dict(complete=True,records=records,reused=self.state['reused'],effects=effects,
                selection_view='z0',additional_training_authorized=False))
            self.state.update(status='complete',phase='four_functional_ablations_complete',completed_at=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true');Queue(p.parse_args()).main()
