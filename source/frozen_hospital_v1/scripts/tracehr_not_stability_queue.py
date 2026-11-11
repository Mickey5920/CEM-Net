"""Exactly 28 authorized fits using existing admission, process and exit machinery."""
import argparse, os, sys, time, math, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_not_confirmation_queue import Queue as PreviousQueue
from scripts.tracehr_recovery_queue import atomic,read,sha,process_identity
from scripts.tracehr_not_stability_common import RUN,MATRIX,protocol,stopping_for

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
        for rel in ('tracehr_not_confirmation_20261017/mimic/confirm_validation_v2',
                    'tracehr_not_functional_20261018/mimic/functional_v1'):
            previous=self.project/'runs'/rel
            assert read(previous/'screen/state.json')['status']=='complete'
            assert read(previous/'start_exit.json')['returncode']==0
        audit=read(self.out.parent/'baseline-audit.json')
        assert audit['complete'] and audit['protected_files_unchanged'] and not audit['test_loaded']
        assert audit['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        assert len(audit['baselines'])==12 and len(audit['controls'])==20
        self.state=dict(database='mimic',pid=os.getpid(),status='preflight',phase='probes',jobs={},
            source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),
            expected_gib=a.expected_gib,gpus=[0,1,4],slots_per_gpu=2,formal_job_limit=28,
            coexistence_verified=True,max_workers_per_gpu=2,from_scratch=True,test_loaded=False,
            hyperparameter_search=False,min_free_gpu_mib=12000,eicu_authorized=False,
            reused=audit['controls'],baselines=audit['baselines'],baseline_audit_sha256=sha(self.out.parent/'baseline-audit.json'),
            admission_rule='Only noT/B0 seeds45-47 and noP/noM/noA/noR seeds43-44 x death/LOS; no extension')
        for v,e,s in MATRIX:
            for pilot in (True,False):
                name=('probe_' if pilot else '')+f'{v}_t{e}_s{s}'
                action=('audit' if v=='B0' else 'probe') if pilot else 'train'
                self.state['jobs'][name]=dict(id=name,variant=v,endpoint=e,seed=s,pilot=pilot,
                    action=action,status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))
        atomic(self.out/'reused_controls.json',self.state['reused']);self.save()

    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==28 and {(j['variant'],j['endpoint'],j['seed']) for j in formal}==set(MATRIX)
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def spec(self,j):return protocol(j['variant'],'mimic',j['seed'])

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE='mimic',
            NUMPY_MADVISE_HUGEPAGE='0',TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        entry='train_tracehr_not_stability_native.py' if j['variant']=='B0' else (
            'train_tracehr_not_stability_functional.py' if j['variant'].endswith(('noA','noR')) else 'train_tracehr_not_stability.py')
        cmd=[sys.executable,'-B',str(self.source/'scripts'/entry),'--project',str(self.project),'--database','mimic',
            '--output',str(out),'--preparation',str(self.preparation),'--variant',j['variant'],'--seed',str(j['seed']),
            '--endpoint',str(j['endpoint']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train':cmd+=['--pilot-record',self.state['jobs']['probe_'+j['id']]['output']]
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
            if j['seed'] in (43,44):
                old=self.state['reused'][f"C5_noT_t{j['endpoint']}_s{j['seed']}"]
                assert edge['reference_random_initial_hash']==old['initial_weights_sha256'],'Same-seed ablation mapping differs'
            eq=read(out/'G0_equivalence.json');assert eq['complete'] and eq['disabled_fusion_max_logit_drift']==0
            if j['variant'].endswith(('noA','noR')):
                assert edge['main_loss_coefficient']==.5
                assert edge['auxiliary_enabled']==j['variant'].endswith('noR')
                assert edge['main_residual_enabled']==j['variant'].endswith('noA')
                assert edge['auxiliary_loss_coefficient']==(.5 if edge['auxiliary_enabled'] else 0.)
                assert edge['parameter_count_removed']==(514 if j['variant'].endswith('noA') else 0)
            if j['pilot']:
                assert r['optimizer_resume_max_parameter_drift']<=1e-6 and r['selected_microbatch']==128
                assert r['long_sequence_stress']['complete']
        if not j['pilot']:
            assert sha(out/r.get('checkpoint_file','best.pt'))==r['checkpoint_sha256']
            assert sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            history=read(out/'history.json');assert 52<=len(history)==r['epochs']<=256
            assert abs(r['dev_ap']-max(x['dev_ap'] for x in history))<1e-7
            assert r['stopping_status']==stopping_for(j['variant'],history,self.spec(j))
            # Confirm first-trigger stopping rather than accepting a late matching status.
            assert not any(stopping_for(j['variant'],history[:i],self.spec(j))['stop'] for i in range(1,len(history)))
            pilot=self.state['jobs']['probe_'+j['id']]['result']['identity']
            for k in ('initial_weights_sha256','protocol','source_manifest_sha256','input_manifest_sha256'):
                assert identity[k]==pilot[k],(j['id'],k)
            if j['variant']!='B0':
                assert r['best_epoch']==next(x['epoch'] for x in history if x['dev_ap']==r['dev_ap'])
                assert all(x['main_view']==x['checkpoint_selection_view']==x['stopping_view']=='z0' for x in history)
        return r

    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):
            self.state.update(status='needs_attention',error='Attempt failed; preserve evidence, no recipe change or automatic retry');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            for e in (0,1):
                for v in ('C5_noT','B0'):
                    hashes=[self.state['jobs'][f'probe_{v}_t{e}_s{s}']['result']['identity']['initial_weights_sha256'] for s in (45,46,47)]
                    hashes += [self.state['reused'][f'{v}_t{e}_s{s}']['initial_weights_sha256'] for s in (42,43,44)]
                    assert len(set(hashes))==6
            probes=[j['result'] for j in jobs if j['pilot']]
            estimate=sum(x['checkpoint_bytes'] for x in probes)*11/2**30+.5
            assert estimate<self.a.expected_gib
            self.state.update(projected_artifact_gib=estimate,seed_routing_verified=True,phase='training',status='running',
                min_free_gpu_mib=math.ceil(max(x['peak_reserved_mib'] for x in probes)*1.35+1536))
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            records={j['id']:j['result'] for j in jobs if not j['pilot']}
            atomic(self.out/'phase_summary.json',dict(complete=True,records=records,reused=self.state['reused'],
                selection_view='z0',second_batch_authorized=False,eicu_authorized=False,additional_training_authorized=False))
            self.state.update(status='complete',phase='first_twenty_eight_complete',completed_at=time.time())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True);p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true');Queue(p.parse_args()).main()
