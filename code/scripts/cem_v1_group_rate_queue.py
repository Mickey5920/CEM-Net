"""Two authorized MIMIC fits, two bounded identity probes, no automatic extension."""
import argparse,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue,atomic,read,sha,process_identity
from scripts import tracehr_noa_main_confirmation_queue as native_validator
from scripts.cem_v1_group_rate_common import *
# The inherited native acceptance code compares the new database-specific command.

class Queue(BaseQueue):
    def __init__(self,a):
        import fcntl
        self.a,self.fcntl=a,fcntl;self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True)
        self.out=a.output.resolve();self.db=a.database
        self.cache=self.project/'cache'/RUN/self.db;self.temp=self.project/'work'/RUN/'tmp'/self.db
        self.preparation=self.project/PREPARATION
        for f in (self.source,self.out,self.cache,self.temp,self.preparation):assert f.resolve().is_relative_to(self.project) and f!=self.project
        for f in (self.out,self.cache,self.temp):f.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.locks,self.handles={},{}
        assert not (self.out/'state.json').exists()
        for n,h in read(self.source/'source_manifest.json').items():assert sha(self.source/n)==h
        audit=read(self.out.parent/'controls-audit.json');assert audit['complete'] and not audit['test_loaded']
        assert audit['source_manifest_sha256']==sha(self.source/'source_manifest.json')
        self.state=dict(database=self.db,pid=os.getpid(),status='preflight',phase='probes',jobs={},started_at=time.time(),
            source_manifest_sha256=sha(self.source/'source_manifest.json'),expected_gib=a.expected_gib,
            gpus=[0,1,4] if self.db=='mimic' else [4,5,6],slots_per_gpu=2,max_workers_per_gpu=2,max_cpu_workers=2,
            formal_job_limit=2,endpoint_fit_limit=2,
            from_scratch=True,test_loaded=False,hyperparameter_search=True,prespecified_group_rate_only=True,additional_fitting_authorized=False,
            min_free_gpu_mib=16384,controls=audit['controls'],contracts=audit['contracts'],controls_audit_sha256=sha(self.out.parent/'controls-audit.json'))
        for v,e,s in matrix(self.db):
            probe_seed=48;pilot_id=f'probe_{v}_t{e}_s{probe_seed}'
            if pilot_id not in self.state['jobs']:self.add(pilot_id,v,e,probe_seed,True,None)
            self.add(f'{v}_t{e}_s{s}',v,e,s,False,pilot_id)
        self.save()
    def add(self,name,v,e,s,pilot,pilot_id,kind='neural'):
        self.state['jobs'][name]=dict(id=name,kind=kind,variant=v,family=v if kind=='baseline' else None,endpoint=e,
            endpoint_count=2 if kind=='baseline' else 1,seed=s,pilot=pilot,pilot_id=pilot_id,
            action=('audit' if v=='B0' else 'probe') if pilot else ('native_train' if kind=='baseline' else 'train'),
            status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))
    def save(self):
        formal=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(formal)==self.state['formal_job_limit'] and sum(j['endpoint_count'] for j in formal)==self.state['endpoint_fit_limit']
        assert {(j['variant'],j['endpoint'],j['seed']) for j in formal if j['kind']=='neural'}==set(matrix(self.db))
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in formal) for s in ('pending','running','completed','failed')},
            endpoint_counts={s:sum(j['endpoint_count'] for j in formal if j['status']==s) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)
    def spec(self,j):return protocol(j['variant'],self.db,j['seed'])
    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),TRACEHR_PROJECT=str(self.project),TRACEHR_DATABASE=self.db,
            NUMPY_MADVISE_HUGEPAGE='0',TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),XDG_CACHE_HOME=str(self.cache/'runtime'),
            TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if j['kind']=='neural':
            entry='train_tracehr_noa_validation_native.py' if j['variant']=='B0' else 'train_cem_v1_group_rate.py'
            cmd=[sys.executable,'-B',str(self.source/'scripts'/entry),'--project',str(self.project),'--database',self.db,
                '--output',str(out),'--preparation',str(self.preparation),'--variant',j['variant'],'--seed',str(j['seed']),
                '--endpoint',str(j['endpoint']),'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
            if not j['pilot']:cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        else:
            contract=self.state['contracts'][j['family']];assert sha(contract['entry'])==contract['entry_sha256']
            cmd=baseline_command(j['family'],j['seed'],out,sys.executable)
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json');log=self.out/(j['id']+'.log')
        # Process-local memory policy also applies to each worker, never global THP settings.
        bootstrap=self.source/'work'/RUN/'memory_entry.py'
        cmd=[cmd[0],'-B',str(bootstrap),*cmd[2:]]
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child;j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()
    def validate(self,j):
        if j['kind']=='baseline':
            original=j['command'];j['command']=[original[0],'-B',*original[3:]]
            try:return native_validator.Queue.validate(self,j)
            finally:j['command']=original
        out=Path(j['output']);r=read(out/'completion.json');ident=r.get('identity',r)
        assert r['complete'] and r['pilot']==j['pilot'] and r['from_scratch'] and not r['test_loaded'] and r['frozen_learnable_parameters']==0
        for k,v in dict(database=self.db,variant=j['variant'],seed=j['seed'],endpoint=j['endpoint'],source_manifest_sha256=self.state['source_manifest_sha256']).items():assert ident[k]==v,(k,ident.get(k),v)
        assert ident['protocol']==self.spec(j)
        if j['variant']=='B0':
            seed=read(out/'seed_audit.json');assert seed['complete'] and seed['initial_parameters_match_same_seed_fresh_native']
            assert seed['initial_weights_sha256']==ident['initial_weights_sha256']
            if j['pilot']:assert r['continuous_native_entry_parity'] and r['updates']==20 and max(r['max_gradient_drift'],r['max_parameter_drift'],r['max_logit_drift'])<=1e-6
        else:
            edge=read(out/'connection_audit.json');assert edge['complete'] and not edge['learned_checkpoint_loaded'] and not edge['stop_gradient']
            assert edge['reference_seed']==j['seed'] and edge['main_loss_coefficient']==.5 and edge['auxiliary_loss_coefficient']==0.
            assert not edge['auxiliary_enabled'] and edge['main_residual_enabled']
            eq=read(out/'G0_equivalence.json');assert eq['complete'] and eq['disabled_fusion_max_logit_drift']==0
            cem=self.state['controls'][f'{CEM}_t{j["endpoint"]}_s{j["seed"]}']
            assert edge['retained_CEM_tensors_exact']
            assert edge['reference_CEM_random_initial_hash']==cem['initial_weights_sha256']
            assert ident['parameters']==cem['parameters']+edge['added_parameters']
            assert edge['added_parameters']>0 and set(edge['protocol_differences'])<=PROTOCOL_FIELDS
            assert edge['fusion_position']==self.spec(j)['fusion_position']
            s3=self.state['controls'][f'S3_t{j["endpoint"]}_s{j["seed"]}']
            assert edge['retained_S3_tensors_exact'] and edge['archived_S3_retained_initialization_verified']
            assert edge['reference_S3_initial_weights_sha256']==s3['initial_weights_sha256']
            assert ident['parameters']==s3['parameters']-edge['removed_S3_parameters']+edge['new_S3_parameters']
            from models.cem_gate_location import expected_new_names,expected_removed_names
            original=read(Path(s3['source'])/'initialization_observation.json')['retained_tensor_hashes']
            original_fusion={n[len('fusion.'):] for n in original if n.startswith('fusion.')}
            assert set(edge['removed_S3_parameter_names'])=={'fusion.'+n for n in expected_removed_names('V1',original_fusion)}
            assert set(edge['new_S3_parameter_names'])=={'fusion.'+n for n in expected_new_names('V1')}
            observer=read(out/'S3_initialization_observation.json')
            assert not observer['tensor_mismatches'] and observer['new_names']==edge['new_S3_parameter_names']
            assert observer['removed_names']==edge['removed_S3_parameter_names']
            u1=self.state['controls'][f'U1_t{j["endpoint"]}_s48']
            u=observer['U1'];info=read(out/'native_initialization.json')
            assert u['reference_U1_initial_weights_sha256']==u['reconstructed_U1_initial_weights_sha256']==u1['initial_weights_sha256']
            assert not u['tensor_mismatches'] and info['archived_U1_retained_initialization_verified']
            assert u['removed_names']==['fusion.gate.weight'] and u['new_names']==[]
            assert ident['parameters']==u1['parameters']-192
            v=read(out/'V1_initialization_observation.json')
            ref=self.state['controls'][f'V1_t{j["endpoint"]}_s48']
            assert v['optimizer_updates']==0 and not v['learned_checkpoint_loaded']
            assert v['actual_initial_weights_sha256']==v['expected_initial_weights_sha256']==ref['initial_weights_sha256']==ident['initial_weights_sha256']
            assert v['parameters']==ref['parameters']==ident['parameters']
            assert not v['tensor_mismatches'] and not v['removed_names'] and not v['new_names']
            assert v['structural_variant']=='V1' and info['archived_V1_all_initial_parameters_verified']
            original_protocol=read(Path(ref['source'])/'run_manifest.json')['protocol']
            diffs={k for k in set(original_protocol)|set(ident['protocol']) if original_protocol.get(k)!=ident['protocol'].get(k)}
            assert diffs==RATE_FIELDS,diffs
            if j['pilot']:
                check=read(out/'rate_check.json');assert check==r['rate_check']
                assert check['complete'] and check['fresh_parameters_and_rng_restored'] and not check['formal_weights_reused']
                assert check['encoder_ratio']==1. and check['readout_ratio']==.5
                assert check['unchanged_parameter_update_max_drift']==check['readout_oracle_max_drift']==0.
                assert set(check['groups']['readout']['parameter_names'])==READOUT_NAMES
                assert check['all_parameter_groups_covered_once'] and len(check['schedules'])==8
            assert edge['event_direct_enabled']==self.spec(j)['event_direct_enabled']
            assert edge['interaction_event_skip_enabled']==self.spec(j)['interaction_event_skip_enabled']
            if j['pilot']:assert r['longest_evidence_stress']['all_retained_tensors_have_main_gradients']
            if j['pilot']:assert r['optimizer_resume_max_parameter_drift']<=1e-6 and r['selected_microbatch']==128 and r['evaluation_microbatch']==256
        if not j['pilot']:
            assert sha(out/r.get('checkpoint_file','best.pt'))==r['checkpoint_sha256'] and sha(out/'predictions_dev.npz')==r['predictions_sha256']
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift']<=1e-5
            h=read(out/'history.json');assert len(h)==r['epochs'] and abs(r['dev_ap']-max(x['dev_ap'] for x in h))<1e-7
            stop=native_stopping if j['variant']=='B0' else stopping
            assert r['stopping_status']==stop(h,self.spec(j)) and not any(stop(h[:i],self.spec(j))['stop'] for i in range(1,len(h)))
            assert len(h)==256 if self.db=='eicu' else 52<=len(h)<=256
            pilot=self.state['jobs'][j['pilot_id']]['result']['identity']
            if j['seed']==48:assert pilot['initial_weights_sha256']==ident['initial_weights_sha256']
            else:assert pilot['initial_weights_sha256']!=ident['initial_weights_sha256']
            if j['variant']!='B0':
                assert all(x['main_view']==x['checkpoint_selection_view']==x['stopping_view']=='z0' and x['auxiliary_dev_ap'] is None for x in h)
                groups=self.state['jobs'][j['pilot_id']]['result']['rate_check']['groups']
                for values in [read(out/'optimizer_initial.json')]+[x['optimizer'] for x in h]:
                    current=values['fusion_groups']
                    assert set(current)=={'encoder','readout'}
                    assert current['readout']['lr']==.5*current['encoder']['lr']
                    assert current['encoder']['lr']==values['fusion']['lr']
                    for role in current:
                        for field in ('parameter_names','tensors','parameters','betas','weight_decay'):
                            assert current[role][field]==groups[role][field],(role,field)
                for epoch in (1,5,8,10,12,15,18,20):
                    d=read(out/f'epoch_{epoch:03d}_diagnostics.json');assert sha(out/f'epoch_{epoch:03d}_model.pt')==d['checkpoint_sha256']
                    paths=d['gradient']['groups']['loss_paths'];assert paths['auxiliary_loss_removed'] and paths['auxiliary_base_gradient_absent']
                    assert all(g['aux_connected']==0 for g in paths['groups'].values())
        return r
    def phase(self):
        jobs=list(self.state['jobs'].values())
        if any(j['status']=='failed' for j in jobs):self.state.update(status='needs_attention',error='Execution/validation failure; no automatic retry or recipe change');return
        if self.state['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['pilot']):
            estimate=.75
            estimate+=sum(j['result']['checkpoint_bytes'] for j in jobs if j['pilot'])/2**30
            estimate+=sum(self.state['jobs'][j['pilot_id']]['result']['checkpoint_bytes']*11/2**30 for j in jobs if not j['pilot'] and j['kind']=='neural')
            assert estimate<self.a.expected_gib,(estimate,self.a.expected_gib)
            self.state.update(projected_artifact_gib=estimate,status='running',phase='training')
        if self.state['phase']=='training' and all(j['status']=='completed' for j in jobs if not j['pilot']):
            self.state.update(status='complete',phase='authorized_batch_complete',completed_at=time.time())
            atomic(self.out/'phase_summary.json',dict(complete=True,database=self.db,endpoint_fits=self.state['endpoint_fit_limit'],additional_training_authorized=False))
    def initializing(self,j):return native_validator.Queue.initializing(self,j)
    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.initializing(j) for j in running):self.state['admission_wait']='serial input/model initialization';return
        pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['pilot']==(self.state['phase']=='probes')]
        if not pending:return
        cpu=[j for j in pending if j['kind']=='baseline' and j['family']!='tabm'];cpu_running=[j for j in running if j.get('gpu') is None]
        ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/2**20
        if cpu and len(cpu_running)<2 and ram>=72 and os.getloadavg()[0]<=2*os.cpu_count() and shutil.disk_usage(self.project).free>(self.a.expected_gib+10)*2**30:
            self.launch(cpu[0],None,next(i for i in (0,1) if not any(x['slot']==i for x in cpu_running)));return
        candidates=[j for j in pending if j not in cpu]
        if not candidates:self.state['admission_wait']='CPU slot/RAM';return
        j=candidates[0];rows=subprocess.check_output(['nvidia-smi','--id='+','.join(map(str,self.state['gpus'])),'--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        choices=[]
        for row in rows.strip().splitlines():
            g,free=map(int,row.split(','));own=sum(x.get('gpu')==g for x in running)
            if own<(1 if self.state['phase']=='probes' else 2):choices.append((own,-free,g))
        self.state['min_free_gpu_mib']=(18000 if self.db=='eicu' else 8192) if j['pilot'] else (11000 if j['kind']=='baseline' else math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.35+1536))
        for _,_,gpu in sorted(choices):
            slot=next(i for i in (0,1) if not any(x.get('gpu')==gpu and x['slot']==i for x in running))
            if self.resources(gpu):self.launch(j,gpu,slot);return
        self.state['admission_wait']='authorized GPU slots or measured resource reserve'
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic'],required=True);p.add_argument('--expected-gib',type=float,required=True)
    Queue(p.parse_args()).main()
