"""Append 45 workers / 72 endpoint results after the six immutable full-model fits."""
import argparse,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue,atomic,read,sha,process_identity,same_process
from scripts.mimiciii_baselines_common import RUN,CLINICAL,NUMERIC,expected,matrix,numeric_protocol

class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl;self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN;self.temp=self.project/'work'/RUN/'tmp'
        for p in (self.out,self.source,self.cache,self.temp):assert p.is_relative_to(self.project) and p!=self.project
        self.out.mkdir(parents=True,exist_ok=True);self.owner=(self.out/'owner.lock').open('a+')
        fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (self.out/'state.json').exists(),'Never repeat an active or completed writer'
        for n,h in read(self.source/'source_manifest.json').items():assert sha(self.source/n)==h,n
        check=read(self.out.parent/'verification.json');assert check['complete'] and check['endpoint_fits']==72
        assert check['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.handles={};self.locks={};self.available_since={}
        self.state=dict(database='mimiciii',pid=os.getpid(),process_identity=process_identity(os.getpid()),
            source_manifest_sha256=sha(self.source/'source_manifest.json'),phase='waiting_predecessor',status='waiting',
            jobs={},started_at=time.time(),gpus=[0,1,4],slots_per_gpu=2,max_workers_per_gpu=2,
            expected_gib=a.expected_gib,formal_workers=45,endpoint_fits=72,test_loaded=False,hyperparameter_search=False,
            min_free_gpu_mib=40960,predecessor=expected()['predecessor'])
        for row in matrix():
            v,s,e=row['variant'],row['seed'],row['endpoint'];name=f'{v}_t{e}_s{s}'
            pilot_seed=48 if v in CLINICAL else s
            pilot_id=f'probe_{v}_t{e}_s{pilot_seed}'
            if pilot_id not in self.state['jobs']:
                self.state['jobs'][pilot_id]=dict(row,id=pilot_id,seed=pilot_seed,pilot=True,status='pending',action='probe',
                    output=str(self.out/'pilots'/pilot_id))
            self.state['jobs'][name]=dict(row,id=name,pilot=False,pilot_id=pilot_id,status='pending',action='train',
                output=str(self.out/'jobs'/name))
        self.save()
    def save(self):
        jobs=list(self.state['jobs'].values());formal=[j for j in jobs if not j['pilot']]
        assert len(formal)==45 and sum(len(j['endpoints']) for j in formal)==72
        statuses=('pending','running','completed','failed')
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in statuses},
            endpoint_counts={s:sum(len(j['endpoints']) for j in formal if j['status']==s) for s in statuses},
            probe_counts={s:sum(j['status']==s for j in jobs if j['pilot']) for s in statuses})
        atomic(self.out/'state.json',self.state)
    def predecessor(self):
        previous=self.project/expected()['predecessor'];state=read(previous/'screen/state.json')
        assert state['source_manifest_sha256']==expected()['parent_manifest_sha256']
        if state['status']=='needs_attention':raise RuntimeError('Predecessor requires attention; no baseline start')
        if state['status']!='complete' or not (previous/'start_exit.json').exists():return False
        assert state['counts']==dict(pending=0,running=0,completed=6,failed=0)
        assert read(previous/'start_exit.json')['returncode']==0 and not same_process(state['pid'],state['process_identity'])
        for j in state['jobs'].values():
            if j['pilot']:continue
            c=read(Path(j['output'])/'completion.json');assert c['complete'] and not c['pilot'] and c['from_scratch']
            assert c['checkpoint_sha256']==sha(Path(j['output'])/'best.pt')
        atomic(self.out/'predecessor-complete.json',dict(complete=True,queue=str(previous/'screen'),
            state_sha256=sha(previous/'screen/state.json'),exit_sha256=sha(previous/'start_exit.json'),results_retrained=False))
        return True
    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):self.state['status']='needs_attention';return
        if self.state['phase']=='waiting_predecessor' and self.predecessor():
            self.state.update(phase='probes',status='preflight',predecessor_completed=True)
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            estimate=.75+sum(v['bytes'] for v in expected()['inputs'].values())/2**30
            for j in jobs:
                if j['pilot']:estimate+=j['result']['checkpoint_bytes']*1.2/2**30;continue
                p=self.state['jobs'][j['pilot_id']]['result']
                multiplier=12 if j['variant']=='noEvidence' else 3
                estimate+=p['checkpoint_bytes']*multiplier*1.2/2**30
            assert estimate<self.a.expected_gib,('Storage bound exceeded',estimate)
            self.state.update(phase='formal',status='running',projected_artifact_gib=estimate,formal_released_at=time.time())
        if all(j['status']=='completed' for j in jobs if not j['pilot']):self.state.update(status='complete',finished_at=time.time())
    def launch(self,j,gpu,slot):
        assert gpu in (0,1,4) and slot in (0,1);out=Path(j['output']);assert not out.exists()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        clinical=j['variant'] in CLINICAL
        script='train_mimiciii_clinical_baseline.py' if clinical else 'train_mimiciii_numeric_baseline.py'
        cmd=[sys.executable,'-B',str(self.source/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),
            str(self.source/'scripts'/script),'--project',str(self.project),'--output',str(out),
            '--variant',j['variant'],'--seed',str(j['seed']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if not clinical:cmd+=['--endpoint',str(j['endpoint'])]
        if not j['pilot']:cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        spec,receipt,log=[self.out/(j['id']+suffix) for suffix in ('_attempt.json','_exit.json','.log')]
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child;j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()
    def admit(self):
        if self.state['status'] in ('complete','needs_attention') or self.state['phase']=='waiting_predecessor':return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        loading=(None,'loading_verified_inputs','randomly_initializing_native','initializing_official_package')
        if any(self.progress(j).get('status') in loading for j in running):self.state['admission_wait']='serialized initialization';return
        probes=self.state['phase']=='probes'
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['pilot']==probes]
        for j in pending:
            if probes:need=40960 if j['variant'] in CLINICAL else 24576
            else:need=max(8192,math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
            if j['variant']=='B0':need=max(24576,need)
            self.state['min_free_gpu_mib']=need
            for slot in range(1 if probes else 2):
                for gpu in (0,1,4):
                    if any(r['gpu']==gpu and r['slot']==slot for r in running):continue
                    if not self.resources(gpu):self.available_since.pop((gpu,j['id']),None);continue
                    if j['variant']=='B0':
                        key=(gpu,j['id']);self.available_since.setdefault(key,time.time())
                        if time.time()-self.available_since[key]<10:continue
                    self.launch(j,gpu,slot);return
    def validate(self,j):
        out=Path(j['output']);r=read(out/'completion.json');i=r.get('identity',r)
        assert r['complete'] and r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded']
        for k,v in dict(database='mimiciii',variant=j['variant'],seed=j['seed'],source_manifest_sha256=self.state['source_manifest_sha256']).items():assert i[k]==v,k
        assert r['frozen_learnable_parameters']==0
        if j['variant'] not in CLINICAL:assert i['endpoint']==j['endpoint'] and i['protocol']==numeric_protocol(j['variant'],j['seed'])
        if j['pilot']:
            assert r['checkpoint_bytes']>0 and r['peak_reserved_mib']>0
            if j['variant']=='B0':assert r['updates']==20 and r['continuous_native_entry_parity'] and max(r[k] for k in ('max_gradient_drift','max_parameter_drift','max_logit_drift'))<=1e-6
            elif j['variant']=='TabM':assert r['epochs']==2 and read(out/'gradient_check.json')['all_learnable_parameters_connected']
            else:assert r['optimizer_resume_max_parameter_drift']<=1e-6
        else:
            import numpy as np
            from sklearn.metrics import average_precision_score
            path=out/r.get('checkpoint_file','best.pt');assert sha(path)==r['checkpoint_sha256']
            assert sha(out/'predictions_dev.npz')==r['predictions_sha256'] and r['checkpoint_replay']['complete']
            assert r['checkpoint_replay']['max_probability_drift']<=1e-5
            pred=dict(np.load(out/'predictions_dev.npz',allow_pickle=True))
            target=dict(np.load(self.project/expected()['numeric_preparation']/'dev_targets.npz'))
            np.testing.assert_array_equal(pred['ids'],target['ids']);np.testing.assert_array_equal(pred['labels'],target['y'][:,j['endpoints']])
            aps=[float(average_precision_score(pred['labels'][:,e],pred['probs'][:,e])) for e in range(len(j['endpoints']))]
            np.testing.assert_allclose(aps,np.atleast_1d(r['dev_ap']),atol=1e-12,rtol=0)
            if j['variant'] in ('noEvidence','B0'):
                from scripts.tracehr_centered_native_common import stopping
                history=read(out/'history.json');spec=numeric_protocol(j['variant'],j['seed'])
                assert all(not stopping(history[:k],spec)['stop'] for k in range(1,len(history)))
                assert stopping(history,spec)==r['stopping_status'] and (r['stopping_status']['stop'] or r['epochs']==256)
            elif j['variant'] in CLINICAL:assert r['epochs']==50
            else:
                history=read(out/'history.json');best=-1.;stale=0
                for k,row in enumerate(history):
                    assert stale<16
                    if row['dev_ap']>best+1e-9:best=row['dev_ap'];stale=0
                    else:stale+=1
                assert stale==16 or len(history)==256
        return r

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-gib',type=float,required=True);a=p.parse_args();a.database='mimiciii'
    q=Queue(a);q.main()
    if q.state['status']!='complete':raise SystemExit(2)
