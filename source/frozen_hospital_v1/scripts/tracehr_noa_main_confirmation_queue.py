"""30 endpoint fits / 21 original worker jobs, twelve bounded same-seed identity probes."""
import argparse,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_not_stability_queue import Queue as PreviousQueue
from scripts.tracehr_recovery_queue import atomic,read,sha,process_identity
from scripts.tracehr_noa_main_confirmation_common import RUN,MATRIX,FAMILIES,baseline_command,protocol

class Queue(PreviousQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl;self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN/'mimic';self.temp=self.project/'work'/RUN/'tmp/mimic'
        self.preparation=self.project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation'
        for f in (self.source,self.out,self.cache,self.temp,self.preparation):assert f.resolve().is_relative_to(self.project) and f!=self.project
        for f in (self.out,self.cache,self.temp):f.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.locks,self.handles={},{}
        assert not (self.out/'state.json').exists() and not a.resume_queue and a.database=='mimic'
        for f,h in read(self.source/'source_manifest.json').items():assert sha(self.source/f)==h
        audit=read(self.out.parent/'baseline-audit.json');assert audit['complete'] and audit['protected_files_unchanged']
        assert audit['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.state=dict(database='mimic',pid=os.getpid(),status='preflight',phase='probes',jobs={},
            source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),expected_gib=a.expected_gib,
            gpus=[0,1,4],slots_per_gpu=2,max_workers_per_gpu=2,max_cpu_workers=2,formal_job_limit=21,endpoint_fit_limit=30,
            coexistence_verified=True,from_scratch=True,test_loaded=False,hyperparameter_search=False,
            min_free_gpu_mib=8192,eicu_authorized=False,other_fitting_authorized=False,reused=audit['controls'],
            baselines=audit['baselines'],contracts=audit['contracts'],baseline_audit_sha256=sha(self.out.parent/'baseline-audit.json'),
            baseline_artifact_bound_gib=audit['baseline_artifact_bound_gib'])
        for v,e,s in MATRIX:
            for pilot in (True,False):
                name=('probe_' if pilot else '')+f'{v}_t{e}_s{s}'
                self.state['jobs'][name]=dict(id=name,kind='neural',variant=v,endpoint=e,endpoint_count=1,seed=s,pilot=pilot,
                    action=('audit' if v=='B0' else 'probe') if pilot else 'train',status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))
        for s in (48,49,50):
            for family in FAMILIES:
                name=f'{family}_s{s}'
                self.state['jobs'][name]=dict(id=name,kind='baseline',variant=family,family=family,endpoint=None,endpoint_count=2,
                    seed=s,pilot=False,action='native_train',status='pending',output=str(self.out/'jobs'/name))
        self.save()

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==21 and sum(j['endpoint_count'] for j in formal)==30
        assert {(j['variant'],j['endpoint'],j['seed']) for j in formal if j['kind']=='neural'}==set(MATRIX)
        assert {(j['family'],j['seed']) for j in formal if j['kind']=='baseline'}=={(f,s) for f in FAMILIES for s in (48,49,50)}
        counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')}
        endpoints={s:sum(j['endpoint_count'] for j in formal if j['status']==s) for s in counts}
        self.state.update(updated_at=time.time(),counts=counts,endpoint_counts=endpoints)
        atomic(self.out/'state.json',self.state)

    def spec(self,j):return protocol(j['variant'],'mimic',j['seed'])

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            NUMPY_MADVISE_HUGEPAGE='0',TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if j['kind']=='neural':
            cmd=[sys.executable,'-B',str(self.source/'scripts'/('train_tracehr_noa_main_native.py' if j['variant']=='B0' else 'train_tracehr_noa_main_confirmation.py')),'--project',str(self.project),
                '--database','mimic','--output',str(out),'--preparation',str(self.preparation),'--variant',j['variant'],
                '--seed',str(j['seed']),'--endpoint',str(j['endpoint']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
            if not j['pilot']:cmd+=['--pilot-record',self.state['jobs']['probe_'+j['id']]['output']]
        else:
            contract=self.state['contracts'][j['family']]
            assert sha(contract['entry'])==contract['entry_sha256']
            cmd=baseline_command(j['family'],j['seed'],out,sys.executable)
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        out=Path(j['output'])
        if j['kind']=='neural':
            return super().validate(j)
        import numpy as np
        from sklearn.metrics import average_precision_score
        r=read(out/'completion.json');m=read(out/'run_manifest.json');contract=self.state['contracts'][j['family']]
        assert r['complete'] and not r['pilot'] and not r['test_loaded'] and r['seed']==j['seed'] and r['role']==j['family']
        assert j['command']==baseline_command(j['family'],j['seed'],out,j['command'][0])
        expected=dict(contract['manifest'],seed=j['seed']);assert m==expected,('Native manifest changed',j['id'])
        assert sha(contract['entry'])==m['source_sha256']==contract['entry_sha256']
        h=read(out/'history.json');pr=dict(np.load(out/'predictions_dev.npz'))
        reference=dict(np.load(Path(contract['reference']['path'])/'predictions_dev.npz'))
        assert np.array_equal(pr['ids'],reference['ids']) and np.array_equal(pr['labels'],reference['labels'])
        assert pr['probs'].shape==pr['labels'].shape and np.isfinite(pr['probs']).all()
        aps=[float(average_precision_score(pr['labels'][:,e],pr['probs'][:,e])) for e in (0,1)]
        assert abs(float(np.mean(aps))-r['dev_ap'])<1e-12
        for e in (0,1):
            hist=[x for x in h if x['endpoint']==e];assert len(hist)==r['epochs'][e]
            assert [x['epoch'] for x in hist]==list(range(1,len(hist)+1))
            best=-1.;best_epoch=0;stale=0;patience={'tabm':16,'lightgbm':75,'xgboost':50}[j['family']]
            tolerance=1e-9 if j['family']=='tabm' else 0.
            for i,row in enumerate(hist):
                if row['dev_ap']>best+tolerance:best=row['dev_ap'];best_epoch=row['epoch'];stale=0
                else:stale+=1
                assert stale<patience or i==len(hist)-1,'Native early stopping passed its first trigger'
            assert best_epoch==r['best_epoch'][e]
            assert stale>=patience or len(hist)==(256 if j['family']=='tabm' else 1500)
            # XGBoost logs its callable metric rounded to six decimal places.
            assert abs(aps[e]-hist[best_epoch-1]['dev_ap'])<(1e-6 if j['family']=='xgboost' else 1e-10)
        checkpoints={f.name:sha(f) for f in out.glob('best_task*') if f.is_file()};assert len(checkpoints)==2
        return dict(r,AP=aps,predictions_sha256=sha(out/'predictions_dev.npz'),checkpoint_hashes=checkpoints,
            native_manifest_and_command_verified=True,original_stopping_verified=True)

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):
            self.state.update(status='needs_attention',error='Execution/acceptance failure; preserve records and inspect before any repair');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            probes=[j['result'] for j in jobs if j['pilot']]
            estimate=sum(r['checkpoint_bytes'] for r in probes)*11/2**30+self.state['baseline_artifact_bound_gib']+.5
            assert estimate<self.a.expected_gib,(estimate,self.a.expected_gib)
            for variant in ('C5_noT_noA','B0'):
                for e in (0,1):
                    hashes=[self.state['jobs'][f'probe_{variant}_t{e}_s{s}']['result']['identity']['initial_weights_sha256'] for s in (48,49,50)]
                    hashes += [self.state['reused'][f'{variant}_t{e}_s{s}']['initial_weights_sha256'] for s in (43,44,45,46,47)]
                    assert len(set(hashes))==8
            self.state.update(projected_artifact_gib=estimate,seed_routing_verified=True,phase='training',status='running')
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            self.state.update(status='complete',phase='thirty_endpoints_complete',completed_at=time.time())
            atomic(self.out/'phase_summary.json',dict(complete=True,formal_jobs=21,endpoint_fits=30,records={j['id']:j['result'] for j in jobs if not j['pilot']},additional_training_authorized=False,eicu_started=False))

    def initializing(self,j):
        if j['kind']=='neural':return self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native')
        # Native trees publish their manifest after input assembly. TabM's first
        # history row additionally confirms completion of its preprocessing.
        file='history.json' if j['family']=='tabm' else 'run_manifest.json'
        return not (Path(j['output'])/file).exists()

    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.initializing(j) for j in running):self.state['admission_wait']='serial input/model initialization';return
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['pilot']==(self.state['phase']=='probes')]
        if not pending:return
        # CPU tree fits do not occupy a GPU worker slot. Bound them separately.
        cpu=[j for j in pending if j['kind']=='baseline' and j['family']!='tabm']
        cpu_running=[j for j in running if j.get('gpu') is None]
        ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/2**20
        if cpu and len(cpu_running)<2 and ram>=52 and os.getloadavg()[0]<=2*os.cpu_count() and shutil.disk_usage(self.project).free>(self.a.expected_gib+10)*2**30:
            j=cpu[0];slot=next(i for i in (0,1) if not any(x['slot']==i for x in cpu_running))
            self.launch(j,None,slot);return
        gpu_pending=[j for j in pending if j not in cpu]
        if not gpu_pending:self.state['admission_wait']='CPU slots or host RAM reserve';return
        candidate=gpu_pending[0];gpu_rows=subprocess.check_output(['nvidia-smi','--id=0,1,4','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        choices=[]
        for row in gpu_rows.strip().splitlines():
            gpu,free=map(int,row.split(','));own=sum(j.get('gpu')==gpu for j in running)
            if own<(1 if self.state['phase']=='probes' else 2):choices.append((own,-free,gpu))
        self.state['min_free_gpu_mib']=8192 if candidate['pilot'] else (4096 if candidate['kind']=='baseline' else math.ceil(self.state['jobs']['probe_'+candidate['id']]['result']['peak_reserved_mib']*1.35+1536))
        for _,_,gpu in sorted(choices):
            slot=next(i for i in (0,1) if not any(j.get('gpu')==gpu and j['slot']==i for j in running))
            if self.resources(gpu):self.launch(candidate,gpu,slot);return
        self.state['admission_wait']='Authorized GPU slots or measured resource reserve'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True);p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true');Queue(p.parse_args()).main()
