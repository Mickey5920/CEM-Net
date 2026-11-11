"""Exactly two C4 fits; reuse controls, keep z0 selection, no automatic ablations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_structure_queue import Queue as StructureQueue
from scripts.tracehr_recovery_queue import atomic,read,sha,process_identity
from scripts.tracehr_direct_supervision_common import RUN,protocol


class Queue(StructureQueue):
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
        assert a.database=='mimic' and not a.resume_queue
        assert not (self.out/'state.json').exists(),'Never duplicate an existing queue'
        previous=read(self.project/'runs/tracehr_structure_20261014/mimic/g_v1/screen/state.json')
        assert previous['status']=='complete' and previous['coexistence_verified']
        self.state=dict(database='mimic',pid=os.getpid(),status='preflight',phase='probes',jobs={},
            source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),
            expected_gib=a.expected_gib,gpus=[1,0,4],slots_per_gpu=2,formal_job_limit=2,
            coexistence_verified=False,single_worker_per_gpu=True,
            from_scratch=True,test_loaded=False,hyperparameter_search=False,min_free_gpu_mib=10000,
            eicu_authorized=False,trajectory_ablation_authorized=False,reused={})
        for e in (0,1):
            for action in ('probe','train'):
                name=f'probe_C4_t{e}' if action=='probe' else f'C4_t{e}_s42'
                self.state['jobs'][name]=dict(id=name,variant='C4',endpoint=e,seed=42,pilot=action=='probe',
                    action=action,status='pending',output=str(self.out/('pilots' if action=='probe' else 'jobs')/name))
        self.verify_reused(previous)
        self.save()

    def verify_reused(self,previous):
        import numpy as np
        from sklearn.metrics import average_precision_score
        prep=read(self.preparation/'completion.json');folder=Path(prep['input_pointer']['path'])
        for name,r in prep['input_artifacts'].items():assert sha(folder/name)==r['sha256']
        target=dict(np.load(folder/'dev_targets.npz'))
        for v in ('G0','G2'):
            for e in (0,1):
                ref=previous['G0_reused'][str(e)] if v=='G0' else previous['jobs'][f'G2_t{e}_s42']
                path=Path(ref['path'] if v=='G0' else ref['output'])
                c=read(path/'completion.json')
                assert read(Path(ref['exit_receipt']))['returncode']==0
                assert c['complete'] and c['from_scratch'] and not c['test_loaded'] and c['seed']==42
                assert sha(path/'best.pt')==c['checkpoint_sha256']
                assert sha(path/'predictions_dev.npz')==c['predictions_sha256']
                pr=dict(np.load(path/'predictions_dev.npz'))
                assert np.array_equal(pr['ids'],target['ids'])
                assert np.array_equal(pr['labels'],target['y'][:,e:e+1])
                ap=float(average_precision_score(pr['labels'][:,0],pr['probs'][:,0]))
                assert abs(ap-c['dev_ap'])<1e-12
                self.state['reused'][f'{v}_t{e}']=dict(path=str(path),dev_ap=ap,best_epoch=c['best_epoch'],
                    checkpoint_sha256=c['checkpoint_sha256'],predictions_sha256=c['predictions_sha256'])
        for v in ('C1','C2','B0'):
            for e in (0,1):
                paths={'C1':f'runs/tracehr_context_cut_20261016/mimic/c1_v1/screen/jobs/C1_t{e}_s42',
                       'C2':f'runs/tracehr_trajectory_contribution_20261016/mimic/c2_v1/screen/jobs/C2_t{e}_s42',
                       'B0':f'runs/tracehr_execution_refine_20261014/mimic/v1/screen/jobs/F0_t{e}_s42'}
                path=self.project/paths[v]
                c=read(path/'completion.json');pr=dict(np.load(path/'predictions_dev.npz'))
                receipt=path.parent.parent/((v if v in ('C1','C2') else 'F0')+f'_t{e}_s42_exit.json')
                assert read(receipt)['returncode']==0
                assert c['complete'] and not c['test_loaded'] and c['from_scratch'] and c['seed']==42
                assert sha(path/c.get('checkpoint_file','best.pt'))==c['checkpoint_sha256']
                assert sha(path/'predictions_dev.npz')==c['predictions_sha256']
                assert np.array_equal(pr['ids'],target['ids']) and np.array_equal(pr['labels'],target['y'][:,e:e+1])
                ap=float(average_precision_score(pr['labels'][:,0],pr['probs'][:,0]));assert abs(ap-c['dev_ap'])<1e-12
                self.state['reused'][f'{v}_t{e}']=dict(path=str(path),dev_ap=ap,best_epoch=c['best_epoch'],checkpoint_sha256=c['checkpoint_sha256'],predictions_sha256=c['predictions_sha256'])
        atomic(self.out/'reused_controls.json',self.state['reused'])

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==2 and {j['endpoint'] for j in formal}=={0,1}
        assert all(j['variant']=='C4' for j in formal)
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):return protocol('C4','mimic')

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-B',str(self.source/'scripts/train_tracehr_direct_supervision.py'),
             '--project',str(self.project),'--database','mimic','--output',str(out),'--preparation',str(self.preparation),
             '--variant','C4','--endpoint',str(j['endpoint']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train':cmd+=['--pilot-record',self.state['jobs'][f"probe_C4_t{j['endpoint']}"]['output']]
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        result=super().validate(j)
        edge=read(Path(j['output'])/'connection_audit.json')
        assert edge['complete'] and edge['retained_random_tensors_match_C1'] and not edge['learned_checkpoint_loaded']
        assert edge['parameter_count_same_as_C1'] and not edge['stop_gradient']
        assert edge['main_view']==edge['checkpoint_selection_view']==edge['stopping_view']=='z0'
        if not j['pilot']:
            hist=read(Path(j['output'])/'history.json')
            assert result['dev_ap']==max(x['dev_ap'] for x in hist)
            assert result['best_epoch']==next(x['epoch'] for x in hist if x['dev_ap']==result['dev_ap'])
            assert all(x['main_view']==x['checkpoint_selection_view']==x['stopping_view']=='z0' for x in hist)
        return result

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):
            self.state.update(status='needs_attention',error='Attempt failed; no automatic recipe change or retry');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            probes=[j['result'] for j in jobs if j['pilot']]
            estimate=max(x['checkpoint_bytes'] for x in probes)*12/2**30+.5
            assert estimate<self.a.expected_gib
            self.state.update(projected_artifact_gib=estimate,phase='training',status='running',
                              min_free_gpu_mib=math.ceil(max(x['peak_reserved_mib'] for x in probes)*1.35+1536))
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            atomic(self.out/'phase_summary.json',dict(complete=True,reused=self.state['reused'],
                records={j['id']:j['result'] for j in jobs if not j['pilot']},
                eicu_started=False,ablation_started=False,selection_view="z0",
                screening_gate=dict(
                    death_above_B0=self.state['jobs']['C4_t0_s42']['result']['dev_ap']>self.state['reused']['B0_t0']['dev_ap'],
                    los_at_least_C2=self.state['jobs']['C4_t1_s42']['result']['dev_ap']>=self.state['reused']['C2_t1']['dev_ap'])))
            self.state.update(status='complete',phase='C4_two_endpoints_complete',completed_at=time.time())

    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):return
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and
                 ((self.state['phase']=='probes' and j['pilot']) or (self.state['phase']=='training' and not j['pilot']))]
        if not pending:return
        gpu_rows=subprocess.check_output(['nvidia-smi','--id=0,1,4','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        process_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        candidates=[]
        for row in gpu_rows.strip().splitlines():
            g,uid,free=[x.strip() for x in row.split(',')];g=int(g)
            count=sum(x.split(',')[0].strip()==uid for x in process_rows.strip().splitlines())
            own=sum(j['gpu']==g for j in running)
            if own<1:candidates.append((own,count,-int(free),g))
        for _,_,_,g in sorted(candidates):
            slot=next((i for i in (0,1) if not any(j['gpu']==g and j['slot']==i for j in running)),None)
            if slot is not None and self.resources(g):self.launch(pending[0],g,slot);return
        self.state['admission_wait']='authorized GPU slot or measured resource headroom'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true')
    Queue(p.parse_args()).main()
