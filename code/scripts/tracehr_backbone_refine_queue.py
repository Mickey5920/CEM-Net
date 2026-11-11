"""Resource-bounded D0--D3, cross-host selection, then D4--D7; never touch test."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue, atomic, read, sha, process_identity
from scripts.tracehr_backbone_refine_common import RUN, STAGE1, STAGE2, VARIANTS, protocol, stopping, select_structure


class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl = a,fcntl
        self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN/a.database
        self.temp=self.project/'work'/RUN/'tmp'/a.database;self.preparation=self.out.parent/'preparation'
        for path in (self.source,self.out,self.cache,self.temp,self.preparation):
            assert path.is_relative_to(self.project) and path!=self.project
        for path in (self.out,self.cache,self.temp):path.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.locks,self.handles={},{}
        for name,digest in read(self.source/'source_manifest.json').items():assert sha(self.source/name)==digest
        assert read(self.preparation/'completion.json')['preparation_complete']
        if a.resume_queue:
            self.state=read(self.out/'state.json')
            assert self.state['source_manifest_sha256']==sha(self.source/'source_manifest.json')
            assert self.state['database']==a.database and self.state['status']!='complete'
            assert not any(j['status']=='launching' for j in self.state['jobs'].values())
            self.state.update(pid=os.getpid(),resumed_at=time.time())
        else:
            assert not (self.out/'state.json').exists()
            self.state=dict(database=a.database,pid=os.getpid(),status='preflight',phase='stage1_probes',jobs={},
                source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),
                expected_gib=a.expected_gib,gpus=[0,4] if a.database=='mimic' else [1,4],
                slots_per_gpu=2,coexistence_verified=False,formal_job_limit=16,stage1_job_limit=8,
                from_scratch=True,test_loaded=False,min_free_gpu_mib=8192)
            for v in STAGE1:
                for endpoint in (0,1):self.add(f'probe_{v}_t{endpoint}',v,endpoint,'probe')
            for endpoint in (0,1):
                for v in VARIANTS:self.add(f'{v}_t{endpoint}_s42',v,endpoint,'train')
        self.save()

    def add(self,name,variant,endpoint,action):
        assert name not in self.state['jobs']
        pilot=action!='train'
        self.state['jobs'][name]=dict(id=name,variant=variant,endpoint=endpoint,seed=42,pilot=pilot,action=action,
            status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))

    def save(self):
        jobs=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(jobs)==16
        statuses=('pending','running','completed','failed')
        groups={}
        for v in VARIANTS:
            rows=[j for j in jobs if j['variant']==v]
            groups[v]='completed' if all(j['status']=='completed' for j in rows) else (
                'failed' if any(j['status']=='failed' for j in rows) else (
                'running' if any(j['status'] in ('running','completed') for j in rows) else 'pending'))
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in jobs) for s in statuses},
            stage1_counts={s:sum(j['status']==s and j['variant'] in STAGE1 for j in jobs) for s in statuses},
            stage2_counts={s:sum(j['status']==s and j['variant'] in STAGE2 for j in jobs) for s in statuses},
            group_status=groups,group_counts={s:list(groups.values()).count(s) for s in statuses})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):
        return protocol(j['variant'],self.state.get('selected_structure'))

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE=self.a.database,
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-B',str(self.source/'scripts/train_tracehr_backbone_refine.py'),
             '--project',str(self.project),'--database',self.a.database,'--output',str(out),
             '--preparation',str(self.preparation),'--variant',j['variant'],'--endpoint',str(j['endpoint']),
             '--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['variant'] in STAGE2:cmd+=['--selected-structure',self.state['selected_structure']]
        if j['action']!='probe':cmd+=['--pilot-record',self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}"]['output']]
        if j['action']=='smoke':cmd+=['--start-barrier',str(self.out/'coexistence_release.json')]
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],
                        source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        out=Path(j['output']);r=read(out/'completion.json')
        assert r['complete'] and r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded']
        assert r['frozen_learnable_parameters']==0
        identity=r.get('identity',r)
        for key,value in dict(database=self.a.database,variant=j['variant'],endpoint=j['endpoint'],seed=42,
                             source_manifest_sha256=self.state['source_manifest_sha256']).items():assert identity[key]==value
        assert identity['protocol']==self.spec(j)
        if j['action']=='probe':
            assert r['optimizer_resume_max_parameter_drift']<=1e-6
            assert r['effective_batch']==self.spec(j)['effective_batch']
            assert r['selected_microbatch'] in (128,256)
        else:
            assert sha(out/'best.pt')==r['checkpoint_sha256']
            assert sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            if not j['pilot']:
                spec=self.spec(j)
                assert spec['min_epochs']<=r['epochs']<=spec['extension_epochs']
                assert r['stopping_status']==stopping(read(out/'history.json'),spec)
        return r

    def phase(self):
        jobs=self.state['jobs']
        if any(j['status']=='failed' for j in jobs.values()):
            self.state.update(status='needs_attention',error='A verified attempt failed; inspect before releasing dependent work');return
        if self.state['phase']=='stage1_probes':
            probes=[j for j in jobs.values() if j['action']=='probe']
            if all(j['status']=='completed' for j in probes):
                for endpoint in (0,1):
                    common=[jobs[f'probe_{v}_t{endpoint}']['result']['identity']['common_weights_sha256'] for v in ('D1','D2','D3')]
                    assert len(set(common))==1, ('Native common initialization mismatch',endpoint)
                largest=max(probes,key=lambda j:j['result']['peak_reserved_mib'])
                estimate=max(j['result']['checkpoint_bytes'] for j in probes)*60/2**30+3
                assert estimate<self.a.expected_gib, ('Projected storage exceeds declaration',estimate)
                free={g:int(subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)) for g in self.state['gpus']}
                self.state.update(projected_artifact_gib=estimate,common_initialization_verified=True,phase='coexistence',coexistence_gpu=max(free,key=free.get))
                for name in ('smoke_0','smoke_1'):self.add(name,largest['variant'],largest['endpoint'],'smoke')
        if self.state['phase']=='coexistence':
            smokes=[jobs[n] for n in ('smoke_0','smoke_1')]
            if all(self.progress(j).get('status')=='waiting_coexistence_barrier' for j in smokes) and not (self.out/'coexistence_release.json').exists():
                atomic(self.out/'coexistence_release.json',dict(released_at=time.time(),pids=[j['pid'] for j in smokes]))
            if all(j['status']=='completed' for j in smokes):
                overlap=min(j['result']['finished_at'] for j in smokes)-max(read(Path(j['output'])/'training_started.json')['started_at'] for j in smokes)
                assert overlap>0
                self.state.update(coexistence_verified=True,coexistence_overlap_seconds=overlap,phase='stage1',status='running')
        if self.state['phase'] in ('stage1','common_selection_barrier'):
            stage1=[j for j in jobs.values() if not j['pilot'] and j['variant'] in STAGE1]
            if all(j['status']=='completed' for j in stage1):
                records={j['id']:j['result'] for j in stage1}
                summary=dict(database=self.a.database,source_manifest_sha256=self.state['source_manifest_sha256'],results=records)
                if not (self.out/'stage1_summary.json').exists():atomic(self.out/'stage1_summary.json',summary)
                self.state.update(phase='common_selection_barrier',status='waiting_for_other_database')
                if (self.out/'common_selection.json').exists():
                    choice=read(self.out/'common_selection.json')
                    assert choice['source_manifest_sha256']==self.state['source_manifest_sha256']
                    assert choice['records'][self.a.database]==records
                    assert select_structure(choice['records'])==choice['decision']
                    selected=choice['decision']['selected']
                    self.state.update(selected_structure=selected,selection=choice['decision'],phase='stage2',status='running')
                    for v in STAGE2:
                        for endpoint in (0,1):self.add(f'probe_{v}_t{endpoint}',v,endpoint,'probe')
        if self.state['phase']=='stage2':
            formal=[j for j in jobs.values() if not j['pilot']]
            if all(j['status']=='completed' for j in formal):
                results={v:dict(mean_ap=sum(jobs[f'{v}_t{e}_s42']['result']['dev_ap'] for e in (0,1))/2,
                               endpoint_AP=[jobs[f'{v}_t{e}_s42']['result']['dev_ap'] for e in (0,1)]) for v in VARIANTS}
                atomic(self.out/'phase_summary.json',dict(database=self.a.database,results=results,selected_structure=self.state['selected_structure'],evaluation_usage='development selection'))
                self.state.update(status='complete',phase='D0_D7_complete',completed_at=time.time())

    def eligible(self,j):
        if j['status']!='pending':return False
        phase=self.state['phase']
        if phase=='stage1_probes':return j['variant'] in STAGE1 and j['action']=='probe'
        if phase=='coexistence':return j['action']=='smoke'
        if phase=='stage1':return j['variant'] in STAGE1 and not j['pilot']
        if phase=='stage2' and j['variant'] in STAGE2:
            return j['action']=='probe' or self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}"]['status']=='completed'
        return False

    def admit(self):
        if self.state['status'] in ('complete','needs_attention','waiting_for_other_database'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','initializing','randomly_initializing_native') for j in running):return
        pending=[j for j in self.state['jobs'].values() if self.eligible(j)]
        slots=([(self.state['coexistence_gpu'],s) for s in (0,1)] if self.state['phase']=='coexistence' else
               [(g,s) for s in range(2 if self.state['coexistence_verified'] else 1) for g in self.state['gpus']])
        for j in pending:
            probe=self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}"]
            bound=probe['result']['peak_reserved_mib']*1.35+1536 if probe['status']=='completed' else 8192
            if self.state['phase']=='stage2' and probe['status']!='completed':
                reference=self.state['jobs'][f"probe_{self.state['selected_structure']}_t{j['endpoint']}"]['result']
                bound=max(bound,reference['peak_reserved_mib']*1.5+1536)
            self.state['min_free_gpu_mib']=math.ceil(bound)
            for gpu,slot in slots:
                if any(jj['gpu']==gpu and jj['slot']==slot for jj in running):continue
                if self.resources(gpu):self.launch(j,gpu,slot);return


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true')
    Queue(p.parse_args()).main()
