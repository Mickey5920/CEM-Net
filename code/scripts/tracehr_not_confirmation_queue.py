"""Six authorized C5 ablations; fixed recipe, reused controls, no extension."""
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
from scripts.tracehr_not_confirmation_common import RUN,protocol,VARIANTS,MATRIX,stopping
from scripts.tracehr_execution_refine_common import protocol as native_protocol


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
            expected_gib=a.expected_gib,gpus=[1,0,4],slots_per_gpu=2,formal_job_limit=12,
            coexistence_verified=False,max_workers_per_gpu=2,
            from_scratch=True,test_loaded=False,hyperparameter_search=False,min_free_gpu_mib=10000,
            eicu_authorized=False,ablation_authorized=True,reused={},admission_rule='User approved first twelve noT confirmation fits; no automatic extension')
        for variant,e,seed in MATRIX:
            for pilot in (True,False):
                action=('audit' if variant=='B0' else 'probe') if pilot else 'train'
                name=('probe_' if pilot else '')+f'{variant}_t{e}_s{seed}'
                self.state['jobs'][name]=dict(id=name,variant=variant,endpoint=e,seed=seed,pilot=pilot,
                    action=action,status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))
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
        for v in ('C1','C2','C4','C5','B0'):
            for e in (0,1):
                paths={'C5':f'runs/tracehr_auxiliary_isolation_20261017/mimic/c5_v1/screen/jobs/C5_t{e}_s42',
                       'C4':f'runs/tracehr_direct_supervision_20261016/mimic/c4_v2/screen/jobs/C4_t{e}_s42',
                       'C1':f'runs/tracehr_context_cut_20261016/mimic/c1_v1/screen/jobs/C1_t{e}_s42',
                       'C2':f'runs/tracehr_trajectory_contribution_20261016/mimic/c2_v1/screen/jobs/C2_t{e}_s42',
                       'B0':f'runs/tracehr_execution_refine_20261014/mimic/v1/screen/jobs/F0_t{e}_s42'}
                path=self.project/paths[v]
                c=read(path/'completion.json');pr=dict(np.load(path/'predictions_dev.npz'))
                receipt=path.parent.parent/((v if v in ('C1','C2','C4','C5') else 'F0')+f'_t{e}_s42_exit.json')
                assert read(receipt)['returncode']==0
                assert c['complete'] and not c['test_loaded'] and c['from_scratch'] and c['seed']==42
                assert sha(path/c.get('checkpoint_file','best.pt'))==c['checkpoint_sha256']
                assert sha(path/'predictions_dev.npz')==c['predictions_sha256']
                assert np.array_equal(pr['ids'],target['ids']) and np.array_equal(pr['labels'],target['y'][:,e:e+1])
                ap=float(average_precision_score(pr['labels'][:,0],pr['probs'][:,0]));assert abs(ap-c['dev_ap'])<1e-12
                self.state['reused'][f'{v}_t{e}']=dict(path=str(path),dev_ap=ap,best_epoch=c['best_epoch'],checkpoint_sha256=c['checkpoint_sha256'],predictions_sha256=c['predictions_sha256'])
        for e in (0,1):
            path=self.project/f'runs/tracehr_c5_ablation_20261017/mimic/ablation_v1/screen/jobs/C5_noT_t{e}_s42'
            c=read(path/'completion.json');pr=dict(np.load(path/'predictions_dev.npz'))
            assert read(path.parent.parent/f'C5_noT_t{e}_s42_exit.json')['returncode']==0
            assert c['complete'] and c['from_scratch'] and not c['test_loaded'] and c['seed']==42
            assert sha(path/'best.pt')==c['checkpoint_sha256'] and sha(path/'predictions_dev.npz')==c['predictions_sha256']
            assert np.array_equal(pr['ids'],target['ids']) and np.array_equal(pr['labels'],target['y'][:,e:e+1])
            ap=float(average_precision_score(pr['labels'][:,0],pr['probs'][:,0]));assert abs(ap-c['dev_ap'])<1e-12
            self.state['reused'][f'C5_noT_t{e}']=dict(path=str(path),dev_ap=ap,best_epoch=c['best_epoch'],
                initial_weights_sha256=c['initial_weights_sha256'],checkpoint_sha256=c['checkpoint_sha256'],
                predictions_sha256=c['predictions_sha256'])
        atomic(self.out/'reused_controls.json',self.state['reused'])

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==12 and {(j['variant'],j['endpoint'],j['seed']) for j in formal}==set(MATRIX)
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):
        if j['variant']=='B0':
            result=native_protocol('F0');result['seed']=j['seed'];return result
        return protocol(j['variant'],'mimic',j['seed'])

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-B',str(self.source/('scripts/train_tracehr_not_native.py' if j['variant']=='B0' else 'scripts/train_tracehr_not_confirmation.py')),
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
        out=Path(j['output']);r=read(out/'completion.json');identity=r.get('identity',r)
        assert r['complete'] and r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded']
        assert r['frozen_learnable_parameters']==0
        for k,v in dict(database='mimic',variant=j['variant'],endpoint=j['endpoint'],seed=j['seed'],
                        source_manifest_sha256=self.state['source_manifest_sha256']).items():assert identity[k]==v,(k,identity[k],v)
        assert identity['protocol']==self.spec(j)
        if j['variant']=='B0':
            audit=read(out/'seed_audit.json');assert audit['complete'] and audit['initial_parameters_match_same_seed_fresh_native']
            assert audit['seed']==j['seed'] and audit['initial_weights_sha256']==identity['initial_weights_sha256']
            if j['pilot']:
                assert r['continuous_native_entry_parity'] and r['updates']==20
                assert max(r['max_gradient_drift'],r['max_parameter_drift'],r['max_logit_drift'])<=1e-6
        else:
            edge=read(out/'connection_audit.json')
            assert edge['complete'] and edge['retained_random_tensors_match_noT'] and not edge['learned_checkpoint_loaded']
            assert not edge['stop_gradient'] and not edge['auxiliary_base_dependency']
            assert edge['reference_seed']==j['seed'] and edge['main_view']==edge['checkpoint_selection_view']==edge['stopping_view']=='z0'
            eq=read(out/'G0_equivalence.json');assert eq['complete'] and eq['disabled_fusion_max_logit_drift']==0
            if j['pilot']:
                assert r['optimizer_resume_max_parameter_drift']<=1e-6 and r['selected_microbatch']==128
                assert r['long_sequence_stress']['complete']
        if not j['pilot']:
            assert sha(out/r.get('checkpoint_file','best.pt'))==r['checkpoint_sha256']
            assert sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            history=read(out/'history.json');assert 52<=r['epochs']<=256
            assert abs(r['dev_ap']-max(x['dev_ap'] for x in history))<1e-7
            assert r['stopping_status']==stopping(history,self.spec(j))
            if j['variant']!='B0':
                assert r['best_epoch']==next(x['epoch'] for x in history if x['dev_ap']==r['dev_ap'])
                assert all(x['main_view']==x['checkpoint_selection_view']==x['stopping_view']=='z0' for x in history)
        return r

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):
            self.state.update(status='needs_attention',error='Attempt failed; no automatic recipe change or retry');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            probes=[j['result'] for j in jobs if j['pilot']]
            # Each formal fit must share its own seed-specific identity with its probe.
            # The two new seeds must not silently reuse the seed42 RNG path.
            for e in (0,1):
                for v in ('C5_noT','B0'):
                    hashes=[self.state['jobs'][f'probe_{v}_t{e}_s{s}']['result']['identity']['initial_weights_sha256'] for s in (43,44)]
                    assert len(set(hashes))==2
            self.state['seed_routing_verified']=True
            estimate=sum(x['checkpoint_bytes'] for x in probes)*11/2**30+.5
            assert estimate<self.a.expected_gib
            self.state.update(projected_artifact_gib=estimate,phase='training',status='running',
                              min_free_gpu_mib=math.ceil(max(x['peak_reserved_mib'] for x in probes)*1.35+1536))
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            records={j['id']:j['result'] for j in jobs if not j['pilot']}
            effects={key:dict(dev_ap=r['dev_ap'],
                       delta_vs_C5=r['dev_ap']-self.state['reused'][f"C5_t{r['endpoint']}"]['dev_ap'],
                       delta_vs_B0=r['dev_ap']-self.state['reused'][f"B0_t{r['endpoint']}"]['dev_ap'])
                     for key,r in records.items()}
            atomic(self.out/'phase_summary.json',dict(complete=True,reused=self.state['reused'],
                records=records,effects=effects,eicu_started=False,additional_training_authorized=False,
                selection_view='z0',admission_rule=self.state['admission_rule']))
            self.state.update(status='complete',phase='first_twelve_confirmation_complete',completed_at=time.time())

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
            if own<(1 if self.state['phase']=='probes' else 2):candidates.append((own,count,-int(free),g))
        for _,_,_,g in sorted(candidates):
            slot=next((i for i in (0,1) if not any(j['gpu']==g and j['slot']==i for j in running)),None)
            candidate=pending[0]
            if self.state['phase']=='training':
                probe=self.state['jobs']['probe_'+candidate['id']]['result']
                self.state['min_free_gpu_mib']=math.ceil(probe['peak_reserved_mib']*1.35+1536)
            else:
                self.state['min_free_gpu_mib']=12000 if candidate['variant']=='B0' else 8192
            if slot is not None and self.resources(g):self.launch(candidate,g,slot);return
        self.state['admission_wait']='authorized GPU slot or measured resource headroom'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true')
    Queue(p.parse_args()).main()
