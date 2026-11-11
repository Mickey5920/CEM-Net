"""Six new structure fits per database, plus two verified reused E2 controls."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue, atomic, read, sha, process_identity
from scripts.tracehr_structure_common import RUN, VARIANTS, protocol, stopping, reference_paths


class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl
        self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.cache=self.project/'cache'/RUN/a.database
        self.temp=self.project/'work'/RUN/'tmp'/a.database;self.preparation=self.out.parent/'preparation'
        for f in (self.source,self.out,self.cache,self.temp,self.preparation):assert f.is_relative_to(self.project) and f!=self.project
        for f in (self.out,self.cache,self.temp):f.mkdir(parents=True,exist_ok=True)
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
            self.state=dict(database=a.database,pid=os.getpid(),status='preflight',phase='probes',jobs={},
                source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),
                expected_gib=a.expected_gib,gpus=[0,1,4] if a.database=='mimic' else [6,4,5],
                slots_per_gpu=2,coexistence_verified=False,formal_job_limit=6,
                comparison_settings=8,G0_reused={},from_scratch=True,test_loaded=False,
                hyperparameter_search=False,min_free_gpu_mib=10000)
            for v in VARIANTS:
                for e in (0,1):self.add(f'probe_{v}_t{e}',v,e,'probe')
            for v in VARIANTS:
                for e in (0,1):self.add(f'{v}_t{e}_s42',v,e,'train')
        self.save()

    def add(self,name,variant,endpoint,action):
        assert name not in self.state['jobs'] and variant in VARIANTS
        pilot=action!='train'
        self.state['jobs'][name]=dict(id=name,variant=variant,endpoint=endpoint,seed=42,pilot=pilot,action=action,
            status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))

    def save(self):
        jobs=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(jobs)==6 and all(j['variant']!='G0' for j in jobs)
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in jobs) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):return protocol(j['variant'],self.a.database)

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE=self.a.database,
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-B',str(self.source/'scripts/train_tracehr_structure.py'),
             '--project',str(self.project),'--database',self.a.database,'--output',str(out),
             '--preparation',str(self.preparation),'--variant',j['variant'],'--endpoint',str(j['endpoint']),
             '--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action'] in ('smoke','train'):cmd+=['--pilot-record',self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}"]['output']]
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
        eq=read(out/'G0_equivalence.json')
        assert eq['complete'] and eq['disabled_fusion_max_logit_drift']==0
        if j['action']=='probe':
            assert r['optimizer_resume_max_parameter_drift']<=1e-6 and r['effective_batch']==256
            assert r['selected_microbatch']==128 and r['long_sequence_stress']['complete']
        else:
            assert sha(out/'best.pt')==r['checkpoint_sha256'] and sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            if not j['pilot']:
                spec=self.spec(j);assert spec['min_epochs']<=r['epochs']<=256
                assert r['stopping_status']==stopping(read(out/'history.json'),spec)
        return r

    def reuse_G0(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        prep,refs=reference_paths(self.project,self.a.database)
        new=read(self.preparation/'completion.json')
        source=self.project/('work/tracehr_centered_native_20261014/v1' if self.a.database=='mimic'
                             else 'work/tracehr_execution_refine_20261014/eicu_v1')
        for endpoint in (0,1):
            key=str(endpoint)
            if key in self.state['G0_reused']:continue
            f=refs[key];exit_path=f.parent.parent/f'E2_t{endpoint}_s42_exit.json'
            if not (f/'completion.json').exists() or not exit_path.exists():continue
            r=read(f/'completion.json');exit_record=read(exit_path)
            assert exit_record['returncode']==0 and r['complete'] and not r['pilot'] and r['from_scratch'] and not r['test_loaded']
            assert r['frozen_learnable_parameters']==0 and r['seed']==42 and r['endpoint']==endpoint
            assert r['source_manifest_sha256']==sha(source/'source_manifest.json')
            for name,digest in read(source/'source_manifest.json').items():assert sha(source/name)==digest
            for name in ('preprocessing.npz','native_summary_scaling.npz','native_configs.json'):
                assert sha(prep/name)==sha(self.preparation/name)
            assert r['initial_weights_sha256']==new['base_initial_weights_sha256'][key]
            assert sha(f/'best.pt')==r['checkpoint_sha256'] and sha(f/'predictions_dev.npz')==r['predictions_sha256']
            prediction=dict(np.load(f/'predictions_dev.npz'))
            raw_folder=Path(new['input_pointer']['path'])
            target=dict(np.load(raw_folder/'dev_targets.npz'))
            assert np.array_equal(prediction['ids'],target['ids'])
            assert np.array_equal(prediction['labels'],target['y'][:,endpoint:endpoint+1])
            ap=float(average_precision_score(prediction['labels'][:,0],prediction['probs'][:,0]))
            assert abs(ap-r['dev_ap'])<1e-12
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            self.state['G0_reused'][key]=dict(complete=True,reused=True,original_variant='E2',dev_ap=ap,
                epochs=r['epochs'],best_epoch=r['best_epoch'],path=str(f),completion_sha256=sha(f/'completion.json'),
                checkpoint_sha256=r['checkpoint_sha256'],predictions_sha256=r['predictions_sha256'],
                initial_weights_sha256=r['initial_weights_sha256'],exit_receipt=str(exit_path),from_scratch=True,test_loaded=False)
            atomic(self.out/'G0_reuse.json',self.state['G0_reused'])

    def phase(self):
        jobs=self.state['jobs']
        if any(j['status']=='failed' for j in jobs.values()):
            self.state.update(status='needs_attention',error='A verified attempt failed; inspect before releasing dependent work');return
        self.reuse_G0()
        if self.state['phase']=='probes':
            probes=[j for j in jobs.values() if j['action']=='probe']
            if all(j['status']=='completed' for j in probes):
                for endpoint in (0,1):
                    initial=[jobs[f'probe_{v}_t{endpoint}']['result']['identity']['initial_weights_sha256'] for v in VARIANTS]
                    assert len(set(initial))==1
                    counts=[jobs[f'probe_{v}_t{endpoint}']['result']['identity']['parameters'] for v in VARIANTS]
                    assert len(set(counts))==1
                largest=max(probes,key=lambda j:j['result']['peak_reserved_mib'])
                cache=sum(x['bytes'] for x in read(self.preparation/'completion.json')['trajectory_artifacts'].values())/2**30
                estimate=max(j['result']['checkpoint_bytes'] for j in probes)*28/2**30+cache+4
                assert estimate<self.a.expected_gib,('Projected storage exceeds declaration',estimate)
                free={g:int(subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)) for g in self.state['gpus']}
                self.state.update(projected_artifact_gib=estimate,common_initialization_verified=True,
                    matched_G1_G3_parameter_count=True,phase='coexistence',coexistence_gpu=max(free,key=free.get))
                for name in ('smoke_0','smoke_1'):self.add(name,largest['variant'],largest['endpoint'],'smoke')
        if self.state['phase']=='coexistence':
            smokes=[jobs[n] for n in ('smoke_0','smoke_1')]
            if all(self.progress(j).get('status')=='waiting_coexistence_barrier' for j in smokes) and not (self.out/'coexistence_release.json').exists():
                atomic(self.out/'coexistence_release.json',dict(released_at=time.time(),pids=[j['pid'] for j in smokes]))
            if all(j['status']=='completed' for j in smokes):
                overlap=min(j['result']['finished_at'] for j in smokes)-max(read(Path(j['output'])/'training_started.json')['started_at'] for j in smokes)
                assert overlap>0
                self.state.update(coexistence_verified=True,coexistence_overlap_seconds=overlap,phase='training',status='running')
        if self.state['phase'] in ('training','waiting_G0'):
            formal=[j for j in jobs.values() if not j['pilot']]
            if all(j['status']=='completed' for j in formal):
                if len(self.state['G0_reused'])<2:
                    self.state.update(phase='waiting_G0',status='waiting_for_existing_E2');return
                atomic(self.out/'phase_summary.json',dict(database=self.a.database,
                    G0_reused=self.state['G0_reused'],records={j['id']:j['result'] for j in formal},
                    O_channel_started=False,hyperparameter_search=False))
                self.state.update(status='complete',phase='G0_G3_complete',completed_at=time.time())

    def eligible(self,j):
        return j['status']=='pending' and ((self.state['phase']=='probes' and j['action']=='probe') or
            (self.state['phase']=='coexistence' and j['action']=='smoke') or
            (self.state['phase']=='training' and j['action']=='train'))

    def admit(self):
        if self.state['status'] in ('complete','needs_attention','waiting_for_existing_E2'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):return
        pending=[j for j in self.state['jobs'].values() if self.eligible(j)]
        slots=([(self.state['coexistence_gpu'],s) for s in (0,1)] if self.state['phase']=='coexistence' else
               [(g,s) for s in range(2 if self.state['coexistence_verified'] else 1) for g in self.state['gpus']])
        for j in pending:
            probe=self.state['jobs'][f"probe_{j['variant']}_t{j['endpoint']}"]
            self.state['min_free_gpu_mib']=math.ceil(probe['result']['peak_reserved_mib']*1.35+1536) if probe['status']=='completed' else 10000
            for gpu,slot in slots:
                if any(jj['gpu']==gpu and jj['slot']==slot for jj in running):continue
                if self.resources(gpu):self.launch(j,gpu,slot);return


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--expected-gib',type=float,required=True);p.add_argument('--resume-queue',action='store_true')
    Queue(p.parse_args()).main()
