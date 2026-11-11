"""Bounded, persistent queue: 6 pilots, 6 fresh fits, then 6 fixed tests."""
import argparse, math, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue, process_identity, same_process
from scripts.mimiciii_numeric_alignment_common import *


class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl;self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN;self.temp=self.project/'work/t_m3qa'
        for p in (self.out,self.source,self.cache,self.temp):assert p.is_relative_to(self.project) and p!=self.project
        self.out.mkdir(parents=True,exist_ok=True);self.owner=(self.out/'owner.lock').open('a+')
        fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (self.out/'state.json').exists(),'Do not duplicate an active/completed writer'
        for n,h in read(self.source/'source_manifest.json').items():assert sha(self.source/n)==h,n
        check=read(self.out.parent/'verification.json');assert check['complete'] and check['formal_fits']==6
        assert check['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.handles={};self.locks={}
        self.state=dict(database='mimiciii',pid=os.getpid(),process_identity=process_identity(os.getpid()),
            source_manifest_sha256=sha(self.source/'source_manifest.json'),phase='probes',status='preflight',
            jobs={},started_at=time.time(),gpus=[0,1,4],slots_per_gpu=2,max_workers_per_gpu=2,
            expected_gib=a.expected_gib,formal_fits=6,hyperparameter_search=False,min_free_gpu_mib=24576)
        for row in matrix():
            name=f"{row['variant']}_t{row['endpoint']}_s{row['seed']}"
            for action,prefix,folder in [('probe','probe_','pilots'),('train','','jobs'),('evaluate','test_','test')]:
                key=prefix+name
                self.state['jobs'][key]=dict(row,id=key,pilot=action=='probe',action=action,pilot_id='probe_'+name,
                    fit_id=name,status='pending',output=str(self.out/folder/key))
        self.save()

    def save(self):
        rows=list(self.state['jobs'].values());statuses=('pending','launching','running','completed','failed')
        assert sum(r['action']=='train' for r in rows)==6
        self.state.update(updated_at=time.time())
        for action,key in [('train','counts'),('probe','probe_counts'),('evaluate','evaluation_counts')]:
            self.state[key]={s:sum(j['status']==s and j['action']==action for j in rows) for s in statuses}
        atomic(self.out/'state.json',self.state)

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):self.state['status']='needs_attention';return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['action']=='probe'):
            estimate=1.
            for j in jobs:
                if j['action']=='probe':estimate+=j['result']['checkpoint_bytes']*1.2/2**30
                elif j['action']=='train':
                    pilot=self.state['jobs'][j['pilot_id']]['result']
                    estimate+=pilot['checkpoint_bytes']*(3 if j['variant']=='TabM-mini' else 12)*1.2/2**30
            assert estimate<self.a.expected_gib,('storage estimate exceeds declaration',estimate)
            self.state.update(phase='formal',status='running',projected_artifact_gib=estimate,formal_released_at=time.time())
        if self.state['phase']=='formal' and all(j['status']=='completed' for j in jobs if j['action']=='train'):
            atomic(self.out/'fixed-checkpoint-lock.json',dict(complete=True,created_at=time.time(),
                selections={j['id']:dict(checkpoint_sha256=j['result']['checkpoint_sha256'],best_epoch=j['result']['best_epoch'],
                    dev_ap=j['result']['dev_ap']) for j in jobs if j['action']=='train'},test_used_for_selection=False))
            self.state.update(phase='evaluation',status='evaluating',all_6_fits_successfully_exited=True)
        if self.state['phase']=='evaluation' and all(j['status']=='completed' for j in jobs if j['action']=='evaluate'):
            self.state.update(status='complete',finished_at=time.time())

    def launch(self,j,gpu,slot):
        assert gpu in (0,1,4) and slot in (0,1);out=Path(j['output']);assert not out.exists()
        cmd=[sys.executable,'-B',str(self.source/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),
            str(self.source/'scripts/train_mimiciii_numeric_alignment.py'),'--project',str(self.project),'--output',str(out),
            '--variant',j['variant'],'--seed',str(j['seed']),'--endpoint',str(j['endpoint']),
            '--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train':cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        if j['action']=='evaluate':cmd+=['--fit-output',self.state['jobs'][j['fit_id']]['output']]
        spec,receipt,log=[self.out/(j['id']+s) for s in ('_attempt.json','_exit.json','.log')]
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child;j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):
            self.state['admission_wait']='serialized initialization';return
        action={'probes':'probe','formal':'train','evaluation':'evaluate'}[self.state['phase']]
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['action']==action]
        for j in pending:
            need=24576 if action=='probe' else max(8192,math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
            self.state['min_free_gpu_mib']=need
            for slot in range(1 if action=='probe' else 2):
                for gpu in (0,1,4):
                    if any(r['gpu']==gpu and r['slot']==slot for r in running):continue
                    if self.resources(gpu):self.launch(j,gpu,slot);return

    def validate(self,j):
        import numpy as np
        from sklearn.metrics import average_precision_score
        out=Path(j['output']);r=read(out/'completion.json');i=r.get('identity',r)
        assert r['complete']
        for k,v in dict(database='mimiciii',variant=j['variant'],seed=j['seed'],endpoint=j['endpoint'],source_manifest_sha256=self.state['source_manifest_sha256']).items():assert i[k]==v,k
        assert i['protocol']==protocol(j['variant'],j['seed'])
        audit=read(out/'connection_audit.json')
        assert audit['P_retained'] and audit['M_retained'] and audit['retained_CEM_tensors_exact']
        assert audit['event_coordinates_unchanged'] and audit['quantile_transform_exact'] and not audit['double_scaling']
        assert audit['transform_sha256']==expected()['transform_sha256'] and not audit['transform_refitted']
        assert i['parameters']==12902328 and i['members']==8
        if j['action']=='evaluate':
            assert r['optimizer_updates']==0 and not r['parameters_updated'] and r['dev_replay_max_drift']<=1e-5
            assert r['model_sha256']==self.state['jobs'][j['fit_id']]['result']['checkpoint_sha256']
            split='test';pred=out/'predictions_test.npz';target_path=self.project/expected()['test_inputs']/'test_targets.npz'
        else:
            assert r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded'] and r['frozen_learnable_parameters']==0
            if j['pilot']:
                assert r['checkpoint_bytes']>0 and r['peak_reserved_mib']>0
                if j['variant']=='TabM-mini':
                    assert r['epochs']==2 and read(out/'gradient_check.json')['all_learnable_parameters_connected']
                    assert read(out/'architecture_audit.json')['arguments']['arch_type']=='tabm-mini'
                else:
                    assert r['optimizer_resume_max_parameter_drift']<=1e-6 and r['selected_microbatch']==128 and r['evaluation_microbatch']==256
                    assert read(out/'initialization_observation.json')['retained_CEM_tensors_exact']
                return r
            assert sha(out/r.get('checkpoint_file','best.pt'))==r['checkpoint_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            history=read(out/'history.json');assert r['best_epoch']==1+int(np.argmax([h['dev_ap'] for h in history]))
            if j['variant']=='TabM-mini':
                best=-1.;stale=0
                for h in history:
                    assert stale<16
                    if h['dev_ap']>best+1e-9:best=h['dev_ap'];stale=0
                    else:stale+=1
                assert stale==16 or len(history)==256
            else:
                from scripts.mimiciii_fixed_common import stopping
                spec=protocol(j['variant'],j['seed'])
                assert all(not stopping(history[:k],spec)['stop'] for k in range(1,len(history)))
                assert stopping(history,spec)==r['stopping_status'] and (r['stopping_status']['stop'] or r['epochs']==256)
                assert r['main_view']==r['checkpoint_selection_view']==r['stopping_view']=='z0'
                for epoch in (1,5,8,10,12,15,18,20):
                    assert sha(out/f'epoch_{epoch:03d}_model.pt')==read(out/f'epoch_{epoch:03d}_diagnostics.json')['checkpoint_sha256']
            split='dev';pred=out/'predictions_dev.npz';target_path=self.project/'runs/mimiciii_adaptation_20261024/v1/preparation/dev_targets.npz'
        assert sha(pred)==r['predictions_sha256'];v=dict(np.load(pred));target=dict(np.load(target_path))
        np.testing.assert_array_equal(v['ids'],target['ids']);np.testing.assert_array_equal(v['labels'][:,0],target['y'][:,j['endpoint']])
        ap=float(average_precision_score(v['labels'][:,0],v['probs'][:,0]));assert abs(ap-(r['dev_ap'] if split=='dev' else r['metrics']['mean_ap']))<1e-12
        return r


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-gib',type=float,required=True);a=p.parse_args();a.database='mimiciii'
    q=Queue(a);q.main()
    if q.state['status']!='complete':raise SystemExit(2)
