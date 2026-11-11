"""Six RM comparisons following the completed readonly gate; fixed resource scope."""
import argparse,os,sys,time,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue,read,sha,atomic,process_identity
from scripts.tracehr_r_member_common import RUN,MATRIX,VARIANTS,protocol,stopping,expected

class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl
        self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True);self.out=a.output.resolve()
        self.cache=self.project/'cache'/RUN/'mimic';self.temp=self.project/'work'/RUN/'tmp/mimic'
        self.preparation=self.project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation'
        for p in (self.source,self.out,self.cache,self.temp,self.preparation):assert p.resolve().is_relative_to(self.project) and p!=self.project
        for p in (self.out,self.cache,self.temp):p.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.locks,self.handles={},{}
        assert a.database=='mimic' and not (self.out/'state.json').exists()
        for n,h in read(self.source/'source_manifest.json').items():assert sha(self.source/n)==h
        check=read(self.out.parent/'controls-audit.json')
        assert check['complete'] and check['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.state=dict(database='mimic',pid=os.getpid(),status='preflight',phase='probes',jobs={},
            source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),expected_gib=a.expected_gib,
            gpus=[0,1,4],slots_per_gpu=2,max_workers_per_gpu=2,formal_job_limit=6,from_scratch=True,test_loaded=False,
            hyperparameter_search=False,min_free_gpu_mib=12000,controls_audit_sha256=sha(self.out.parent/'controls-audit.json'),
            eicu_authorized=False,second_batch_authorized=False,reused_noR=[])
        for v in VARIANTS:
            for e in (0,1):self.add_job(f'probe_{v}_t{e}_s43',v,e,43,'probe')
        for v,e,s in MATRIX:self.add_job(f'{v}_t{e}_s{s}',v,e,s,'train')
        self.save()

    def add_job(self,name,v,e,s,kind):
        pilot=kind!='train'
        self.state['jobs'][name]=dict(id=name,variant=v,endpoint=e,seed=s,kind=kind,pilot=pilot,
            status='pending',reused=False,pilot_id=f'probe_{v}_t{e}_s43',output=str(self.out/('pilots' if pilot else 'jobs')/name))

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==6 and {(j['variant'],j['endpoint'],j['seed']) for j in formal}==set(MATRIX)
        fresh=[j for j in formal if not j['reused']]
        counts=lambda rows:{s:sum(j['status']==s for j in rows) for s in ('pending','running','completed','failed')}
        self.state.update(updated_at=time.time(),counts=counts(formal),new_fit_counts=counts(fresh))
        atomic(self.out/'state.json',self.state)

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),
            XDG_CACHE_HOME=str(self.cache/'runtime'),TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        assert j['kind'] in ('probe','train')
        entry='train_tracehr_r_member.py'
        cmd=[sys.executable,'-B',str(self.source/'scripts'/entry),'--project',str(self.project),'--output',str(out),
            '--preparation',str(self.preparation),'--seed',str(j['seed']),'--endpoint',str(j['endpoint']),'--expected-gib',str(self.a.expected_gib)]
        if j['kind']!='r_equivalence':cmd+=['--database','mimic','--variant',j['variant'],'--action',j['kind']]
        if j['kind']=='train':cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child;j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        out=Path(j['output']);r=read(out/'completion.json')
        assert r['complete'] and not r['test_loaded']
        assert j['kind'] in ('probe','train')
        identity=r.get('identity',r)
        for k,v in dict(database='mimic',variant=j['variant'],seed=j['seed'],endpoint=j['endpoint'],source_manifest_sha256=self.state['source_manifest_sha256']).items():assert identity[k]==v,(k,identity[k],v)
        assert identity['protocol']==protocol(j['variant'],'mimic',j['seed'])
        assert r['from_scratch'] and r['frozen_learnable_parameters']==0 and r['pilot']==j['pilot']
        edge=read(out/'connection_audit.json')
        assert edge['complete'] and edge['retained_random_tensors_match_noA'] and not edge['learned_checkpoint_loaded'] and not edge['stop_gradient']
        assert edge['reference_seed']==j['seed'] and edge['main_loss_coefficient']==.5 and edge['auxiliary_loss_coefficient']==0.
        eq=read(out/'G0_equivalence.json');assert eq['complete'] and eq['auxiliary_absent']
        if j['pilot']:
            rm=read(out/'RM_input_audit.json');assert rm['complete'] and rm['weights_unchanged'] and rm['phi_gradient_zero'] and rm['R_to_M_gradient_norm']>0
            assert r['selected_microbatch']==128 and r['evaluation_microbatch']==256 and r['optimizer_resume_max_parameter_drift']<=1e-6
        else:
            assert sha(out/'best.pt')==r['checkpoint_sha256'] and sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            h=read(out/'history.json');spec=identity['protocol']
            assert 52<=len(h)==r['epochs']<=256 and r['dev_ap']==max(x['dev_ap'] for x in h)
            assert r['best_epoch']==next(x['epoch'] for x in h if x['dev_ap']==r['dev_ap'])
            assert r['stopping_status']==stopping(h,spec)
            assert not any(stopping(h[:i],spec)['stop'] for i in range(1,len(h)))
            assert all(x['main_view']==x['checkpoint_selection_view']==x['stopping_view']=='z0' and x['auxiliary_dev_ap'] is None for x in h)
            for epoch in (1,5,8,10,12,15,18,20):
                diag=read(out/f'epoch_{epoch:03d}_diagnostics.json')
                assert sha(out/f'epoch_{epoch:03d}_model.pt')==diag['checkpoint_sha256']
                paths=diag['gradient']['groups']['loss_paths'];assert paths['auxiliary_loss_removed'] and paths['auxiliary_base_gradient_absent']
                assert all(g['aux_connected']==0 for g in paths['groups'].values())
        return r

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):self.state.update(status='needs_attention',error='Execution or validation failure; no automatic retry or recipe change');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['kind']=='probe'):
            estimate=.5
            for j in jobs:
                if j['kind']=='probe':estimate+=j['result']['checkpoint_bytes']/2**30
                if j['kind']=='train' and not j['reused']:estimate+=self.state['jobs'][j['pilot_id']]['result']['checkpoint_bytes']*11/2**30
            assert estimate<self.a.expected_gib,('Retention bound exceeds declared disk budget',estimate)
            self.state.update(projected_artifact_gib=estimate,phase='training',status='running')
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            atomic(self.out/'phase_summary.json',dict(complete=True,comparisons=6,new_fits=6,reused_noR=self.state['reused_noR'],eicu_started=False,additional_training_authorized=False))
            self.state.update(status='complete',phase='six_RM_complete',completed_at=time.time())

    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):return
        kind={'r_equivalence':'r_equivalence','probes':'probe','training':'train'}[self.state['phase']]
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['kind']==kind]
        if not pending:return
        rows=subprocess.check_output(['nvidia-smi','--id=0,1,4','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        candidates=[]
        for row in rows.strip().splitlines():
            g,free=map(int,row.split(','));own=sum(j['gpu']==g for j in running)
            if own<(2 if kind=='train' else 1):candidates.append((own,-free,g))
        for _,_,g in sorted(candidates):
            slot=next(i for i in (0,1) if not any(j['gpu']==g and j['slot']==i for j in running))
            j=pending[0]
            self.state['min_free_gpu_mib']=math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.35+1536) if kind=='train' else (14000 if kind=='r_equivalence' else 8192)
            if self.resources(g):self.launch(j,g,slot);return
        self.state['admission_wait']='authorized GPU slot or measured host/GPU resource headroom'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True);p.add_argument('--expected-gib',type=float,required=True)
    Queue(p.parse_args()).main()
