"""Persistent two-worker queue for exactly ten formal jobs per database."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
RUN='tracehr_joint_recovery_20261011'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def process_identity(pid):
    try:
        line=Path(f'/proc/{pid}/stat').read_text();tail=line[line.rfind(')')+2:].split()
        return dict(start_ticks=tail[19],state=tail[0],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(pid,record):
    now=process_identity(pid)
    return bool(now and now['state']!='Z' and record and now['start_ticks']==record['start_ticks'] and now['boot']==record['boot'])


def select_config(records):
    assert set(records)=={'mimic','eicu'}
    rows=[]
    for config in ('C0','C1','C2','C3'):
        scores=[]
        for db in ('mimic','eicu'):
            r=records[db][config]
            assert r['complete'] and not r['pilot'] and r['from_scratch'] and not r['test_loaded']
            assert (r['database'],r['branches'],r['fusion'],r['config'],r['seed'])==(db,'AVR','F4l',config,42)
            assert r['frozen_learnable_parameters']==0 and 40<=r['epochs']<=80 and math.isfinite(r['dev_ap'])
            scores.append(r['dev_ap'])
        delta=[scores[i]-records[db]['C0']['dev_ap'] for i,db in enumerate(('mimic','eicu'))]
        rows.append(dict(config=config,mean_ap=sum(scores)/2,database_ap=scores,delta_vs_C0=delta,
                         eligible=all(x>=-.002-1e-12 for x in delta)))
    rows.sort(key=lambda r:(-r['mean_ap'],r['config']))
    selected=next(r for r in rows if r['eligible'])
    return dict(selected=selected,ranking=rows,
        rule='equal two-database mean AP; reject any database loss >0.002 vs C0; defer repeat-only candidates outside the authorized 20')


class Queue:
    def __init__(self,a):
        import fcntl
        self.a=a;self.project=a.project.resolve(strict=True);self.source=a.source.resolve(strict=True);self.out=a.output.resolve()
        self.cache=self.project/'cache'/RUN/a.database;self.temp=self.project/'work'/RUN/'tmp'/a.database
        for p in (self.source,self.out,self.cache,self.temp):assert p.is_relative_to(self.project) and p!=self.project
        for p in (self.out,self.cache,self.temp):p.mkdir(parents=True,exist_ok=True)
        self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.fcntl=fcntl;self.locks={};self.handles={}
        for name,digest in read(self.source/'source_manifest.json').items():assert sha(self.source/name)==digest
        if a.resume_queue:
            self.state=read(self.out/'state.json')
            assert self.state['source_manifest_sha256']==sha(self.source/'source_manifest.json')
            assert self.state['database']==a.database and self.state['status']!='complete'
            assert not any(j['status']=='launching' for j in self.state['jobs'].values()),'Ambiguous launch: inspect wrapper ownership before recovery'
            self.state.update(pid=os.getpid(),resumed_at=time.time())
        else:
            assert not (self.out/'state.json').exists(),'Queue already exists; use inspected recovery'
            self.state=dict(database=a.database,pid=os.getpid(),status='preflight',phase='baseline_preflight',jobs={},
                source_manifest_sha256=sha(self.source/'source_manifest.json'),started_at=time.time(),expected_gib=a.expected_gib,
                gpus=[0,4] if a.database=='mimic' else [1,4],slots_per_gpu=2,coexistence_verified=False,
                formal_job_limit=10,from_scratch=True,test_loaded=False,min_free_gpu_mib=14000)
            for name,branches,fusion,config in [('C0','AVR','F4l','C0'),('A_solo','A','solo','C0'),
                ('V_solo','V','solo','C0'),('R_solo','R','solo','C0'),('C1','AVR','F4l','C1'),
                ('C2','AVR','F4l','C2'),('C3','AVR','F4l','C3')]:
                self.add('probe_'+name,branches,fusion,config,True,'probe')
                self.add(name,branches,fusion,config,False,'train','probe_'+name)
            for h in ('H0','H1','H2'):self.add(h,'AVR',h,None,False,'train','probe_'+h)
        self.save()

    def add(self,name,branches,fusion,config,pilot,action,pilot_id=None):
        assert name not in self.state['jobs']
        self.state['jobs'][name]=dict(id=name,branches=branches,fusion=fusion,config=config,pilot=pilot,action=action,
            pilot_id=pilot_id,status='pending',output=str(self.out/('pilots' if pilot else 'jobs')/name))

    def save(self):
        jobs=[j for j in self.state['jobs'].values() if not j['pilot']]
        assert len(jobs)==10
        self.state.update(updated_at=time.time(),counts={s:sum(j['status']==s for j in jobs) for s in ('pending','running','completed','failed')})
        atomic(self.out/'state.json',self.state)

    def progress(self,j):
        p=Path(j['output'])/'progress.json'
        return read(p) if p.exists() else {}

    def resources(self,gpu):
        if shutil.disk_usage(self.project).free<(self.a.expected_gib+10)*2**30:
            self.state['admission_wait']='disk_reserve';return False
        ram=int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1])/2**20
        if ram<(52 if self.a.database=='mimic' else 72):
            self.state['admission_wait']='host_initialization_ram';return False
        if os.getloadavg()[0]>2*os.cpu_count():
            self.state['admission_wait']='host_cpu_load';return False
        if gpu not in self.locks:
            folder=self.project/'work/tracehr_fullstream_20261003/locks';folder.mkdir(parents=True,exist_ok=True)
            f=(folder/f'gpu{gpu}.lock').open('a+')
            try:self.fcntl.flock(f,self.fcntl.LOCK_EX|self.fcntl.LOCK_NB)
            except BlockingIOError:f.close();self.state['admission_wait']='owned_gpu_lock';return False
            self.locks[gpu]=f
        free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<self.state['min_free_gpu_mib']:
            self.state['admission_wait']='gpu_headroom';return False
        self.state['admission_wait']=None
        return True

    def launch(self,j,gpu,slot):
        out=Path(j['output']);assert not out.exists() and out.resolve().is_relative_to(self.out)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=str(gpu),
            TRACEHR_TABULAR_DEPS=str(self.project/'work/tracehr_tabular_20261004/deps'),TEMP=str(self.temp),TMP=str(self.temp),TMPDIR=str(self.temp),
            XDG_CACHE_HOME=str(self.cache/'runtime'),TORCH_HOME=str(self.cache/'runtime/torch'),MPLCONFIGDIR=str(self.cache/'runtime/matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1')
        cmd=[sys.executable,'-B',str(self.source/'scripts/train_tracehr_recovery.py'),'--project',str(self.project),
            '--database',self.a.database,'--output',str(out),'--branches',j['branches'],'--fusion',j['fusion'],
            '--config',j['config'],'--action',j['action'],'--expected-gib',str(self.a.expected_gib)]
        if j['action']=='train':cmd+=['--pilot-record',self.state['jobs'][j['pilot_id']]['output']]
        if j['pilot'] and j['action']=='train':cmd+=['--smoke','--start-barrier',str(self.out/'coexistence_release.json')]
        spec=self.out/(j['id']+'_attempt.json');receipt=self.out/(j['id']+'_exit.json')
        atomic(spec,dict(command=cmd,cwd=str(self.source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=self.state['source_manifest_sha256']))
        log=self.out/(j['id']+'.log')
        j.update(status='launching',gpu=gpu,slot=slot,command=cmd,log=str(log),exit_receipt=str(receipt));self.save()
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B',str(self.source/'scripts/tracehr_recovery_attempt.py'),'--spec',str(spec)],
                cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[j['id']]=child
        j.update(status='running',pid=child.pid,process_identity=process_identity(child.pid),started_at=time.time());self.save()

    def validate(self,j):
        result=read(Path(j['output'])/'completion.json')
        assert result['complete'] and result['pilot']==j['pilot'] and result['from_scratch'] and not result['test_loaded']
        assert result['frozen_learnable_parameters']==0
        ident=result['identity'] if j['action']=='probe' else result
        assert (ident['database'],ident['branches'],ident['fusion'],ident['config'],ident['seed'])==(self.a.database,j['branches'],j['fusion'],j['config'],42)
        assert ident['source_manifest_sha256']==self.state['source_manifest_sha256']
        if j['action']=='train':
            assert result['epochs']==1 if j['pilot'] else 40<=result['epochs']<=80
            assert result['predictions_sha256']==sha(Path(j['output'])/'predictions_dev.npz')
            assert result['checkpoint_sha256']==sha(Path(j['output'])/'best.pt')
            assert result['checkpoint_replay']['complete']
            if not j['pilot']:
                from scripts.tracehr_joint_stopping import stopping_status
                stop=stopping_status(read(Path(j['output'])/'history.json'),min_epochs=40,patience=20,min_delta=.0001)
                assert result['stopping_status']==stop
                assert stop['stop'] if result['termination']=='early_stopping' else result['epochs']==80
        else:
            assert result['selected_microbatch']==128 and result['optimizer_resume_max_parameter_drift']<=1e-6
        return result

    def poll(self):
        for j in self.state['jobs'].values():
            if j['status']!='running':continue
            handle=self.handles.get(j['id'])
            if handle is not None and handle.poll() is None:continue
            if handle is None and same_process(j['pid'],j['process_identity']):continue
            try:
                receipt=read(j['exit_receipt'])
                assert receipt['job_id']==j['id'] and receipt['returncode']==0,receipt
                assert receipt['source_manifest_sha256']==self.state['source_manifest_sha256']
                result=self.validate(j)
                j.update(status='completed',result=result,finished_at=time.time())
            except Exception as exc:j.update(status='failed',error=repr(exc),finished_at=time.time())

    def phase(self):
        jobs=self.state['jobs'];phase=self.state['phase']
        if any(j['status']=='failed' for j in jobs.values()):
            self.state['status']='needs_attention';return
        base=['probe_C0','probe_A_solo','probe_V_solo','probe_R_solo']
        if phase=='baseline_preflight' and all(jobs[k]['status']=='completed' for k in base):
            hashes=jobs['probe_C0']['result']['initial_branch_hashes']
            for e in 'AVR':assert jobs['probe_'+e+'_solo']['result']['initial_branch_hashes'][e]==hashes[e],('Initial branch mismatch',e)
            previous=read(self.project/'runs/tracehr_joint_redesign_20261010'/self.a.database/'p2_v3/jobs/F4l_s42/run_manifest.json')
            assert previous['initial_weights_sha256']==jobs['probe_C0']['result']['initial_weights_sha256'],'C0 initial weights differ from archived F4l'
            self.state['C0_initial_weights_match_archived_F4l']=True
            peak=max(jobs[k]['result']['peak_reserved_mib'] for k in base)
            estimate=max(jobs[k]['result']['checkpoint_bytes'] for k in base)*34*1.2/2**30
            assert estimate<self.a.expected_gib,('Disk estimate too large',estimate)
            self.state.update(projected_artifact_gib=estimate,min_free_gpu_mib=max(8192,math.ceil(peak*1.5+2048)),phase='coexistence_preflight')
            free={g:int(subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)) for g in self.state['gpus']}
            self.state['coexistence_gpu']=max(free,key=free.get)
            for name in ('smoke_0','smoke_1'):self.add(name,'AVR','F4l','C0',True,'train','probe_C0')
        elif phase=='coexistence_preflight':
            smokes=[jobs[k] for k in ('smoke_0','smoke_1')]
            if all(self.progress(j).get('status')=='waiting_coexistence_barrier' for j in smokes) and not (self.out/'coexistence_release.json').exists():
                atomic(self.out/'coexistence_release.json',dict(released_at=time.time(),pids=[j['pid'] for j in smokes]))
            if all(j['status']=='completed' for j in smokes):
                starts=[read(Path(j['output'])/'training_started.json')['started_at'] for j in smokes]
                ends=[j['result']['finished_at'] for j in smokes];overlap=min(ends)-max(starts)
                assert overlap>0
                self.state.update(coexistence_verified=True,coexistence_overlap_seconds=overlap,phase='S1_S2',status='running')
        if self.state['phase'] in ('S1_S2','common_selection_barrier'):
            if all(jobs[c]['status']=='completed' for c in ('C0','C1','C2','C3')):
                summary=dict(database=self.a.database,source_manifest_sha256=self.state['source_manifest_sha256'],
                    results={c:jobs[c]['result'] for c in ('C0','C1','C2','C3')})
                atomic(self.out/'recipe_summary.json',summary);self.state['phase']='common_selection_barrier'
                if (self.out/'common_selection.json').exists():
                    choice=read(self.out/'common_selection.json');assert select_config(choice['records'])==choice['decision']
                    assert choice['records'][self.a.database]==summary['results']
                    config=choice['decision']['selected']['config'];self.state['selection']=choice['decision']
                    for h in ('H0','H1','H2'):
                        jobs[h]['config']=config;self.add('probe_'+h,'AVR',h,config,True,'probe')
                    self.state['phase']='S3';self.state['status']='running'
        if self.state['phase']=='S3' and all(j['status']=='completed' for j in jobs.values() if not j['pilot']):
            self.state.update(status='complete',phase='twenty_job_campaign_complete',completed_at=time.time())

    def eligible(self,j):
        if j['status']!='pending' or j['config'] is None:return False
        phase=self.state['phase'];jobs=self.state['jobs'];name=j['id']
        if phase=='baseline_preflight':return name in ('probe_C0','probe_A_solo','probe_V_solo','probe_R_solo')
        if phase=='coexistence_preflight':return name in ('smoke_0','smoke_1')
        if not self.state['coexistence_verified']:return False
        if name.removeprefix('probe_') in ('C1','C2','C3') and jobs['C0']['status']!='completed':return False
        if j['pilot_id'] and jobs[j['pilot_id']]['status']!='completed':return False
        if not j['pilot']:
            probe=jobs[j['pilot_id']]['result']
            bound=math.ceil(probe['peak_reserved_mib']*1.5+2048)
            self.state['min_free_gpu_mib']=max(self.state['min_free_gpu_mib'],bound)
        return True

    def admit(self):
        if self.state['status'] in ('complete','needs_attention'):return
        running=[j for j in self.state['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'validating_original_inputs','randomly_initializing_encoders') for j in running):return
        pending=[j for j in self.state['jobs'].values() if self.eligible(j)]
        if not pending:return
        if self.state['phase']=='coexistence_preflight':slots=[(self.state['coexistence_gpu'],s) for s in (0,1)]
        else:slots=[(g,s) for s in range(2 if self.state['coexistence_verified'] else 1) for g in self.state['gpus']]
        for gpu,slot in slots:
            if any(j['gpu']==gpu and j['slot']==slot for j in running):continue
            if self.resources(gpu):self.launch(pending[0],gpu,slot);return

    def main(self):
        try:
            while True:
                self.poll();self.phase();self.admit();self.save()
                if self.state['status']=='complete':return
                if self.state['status']=='needs_attention' and not any(j['status']=='running' for j in self.state['jobs'].values()):return
                time.sleep(2)
        except Exception as exc:self.state.update(status='needs_attention',error=repr(exc));self.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic','eicu'],required=True);p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume-queue',action='store_true');Queue(p.parse_args()).main()
