"""User-authorized idle-first admission; preserves all live fits and frozen workers."""
import argparse
import ast
import importlib.util
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def inventory(project):
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    rows=[]
    for index,gpu in enumerate(tree.findall('gpu')):
        processes=[]
        for item in gpu.findall('./processes/process_info'):
            pid=int(item.findtext('pid'))
            try: command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (OSError,ValueError): command=''
            processes.append(dict(pid=pid,project_worker=str(project) in command,identity_known=bool(command)))
        used=int(gpu.findtext('./fb_memory_usage/used').split()[0])
        utilization=int(gpu.findtext('./utilization/gpu_util').split()[0])
        rows.append(dict(index=index,name=gpu.findtext('product_name'),processes=processes,
                         used_mib=used,utilization=utilization,fully_idle=not processes and utilization==0 and used<=256))
    return rows


def load_recovery_class(source,runtime):
    sys.path.insert(0,str(source))
    from scripts.ema_cross_common import read,sha,atomic,expected
    from scripts.ema_cross_queue import Queue as Original,process_identity,same_process
    recovery=runtime/'ema_cross_completion_recovery.py'
    envelope=read(runtime/'envelope.json')
    assert sha(recovery)==envelope['files'][recovery.name]
    spec=importlib.util.spec_from_file_location('verified_completion_recovery',recovery)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Reuse the audited recovery methods without invoking its predecessor-specific constructor.
    body=next(n for n in ast.parse(recovery.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_queue')
    cls=next(n for n in body.body if isinstance(n,ast.ClassDef) and n.name=='Queue')
    namespace=dict(vars(module),Original=Original,read=read,sha=sha,atomic=atomic,expected=expected,
                   process_identity=process_identity,same_process=same_process)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(recovery),'exec'),namespace)
    return namespace['Queue']


def main():
    parser=argparse.ArgumentParser()
    for name in ('project','source','output','runtime','policy-runtime'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--database',choices=('eicu',),required=True)
    parser.add_argument('--expected-gib',type=float,required=True)
    a=parser.parse_args();a.project=a.project.resolve(strict=True);a.source=a.source.resolve(strict=True)
    for path in (a.source,a.output,a.runtime,a.policy_runtime):assert path.resolve().is_relative_to(a.project)
    Recovered=load_recovery_class(a.source,a.runtime)
    from scripts.ema_cross_common import read,sha
    from scripts.ema_cross_queue import process_identity,same_process

    class Queue(Recovered):
        def __init__(self,a):
            import fcntl
            self.a,self.fcntl=a,fcntl;self.project=a.project;self.source=a.source;self.out=a.output.resolve()
            self.cache=self.project/'cache/tracehr_ema_cross_20261024';self.temp=self.project/'work/t_ema_cross'
            self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state=read(self.out/'state.json');old_pid=self.state['pid'];old_identity=self.state['process_identity']
            handoff=read(self.out.parent/'idle-gpu-handoff.json')
            assert handoff['old_queue_pid']==old_pid and handoff['old_queue_identity']==old_identity
            assert not same_process(old_pid,old_identity) and handoff['complete']
            assert self.state['source_manifest_sha256']==sha(self.source/'source_manifest.json')
            assert self.state['database']=='eicu' and self.state['gpus']==[4,5,6]
            for name,h in read(self.source/'source_manifest.json').items():assert sha(self.source/name)==h,name
            policy=read(a.policy_runtime/'envelope.json')
            assert sha(a.policy_runtime/'envelope.json')==handoff['runtime_envelope_sha256']
            for name,h in policy['files'].items():assert sha(a.policy_runtime/name)==h,name
            self.handles={};self.locks={};self.idle_since={};self.leases={}
            for job in self.state['jobs'].values():
                assert job['status']!='launching'
                if job['status']=='running':
                    assert same_process(job['pid'],job['process_identity']) or Path(job['exit_receipt']).exists()
            for gpu in sorted({j['gpu'] for j in self.state['jobs'].values() if j['status']=='running'}):
                handle=(self.project/'work/tracehr_fullstream_20261003/locks'/f'gpu{gpu}.lock').open('a+')
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);self.locks[gpu]=handle
            self.state.update(pid=os.getpid(),process_identity=process_identity(os.getpid()),
                gpus=policy['visible_gpus'],fallback_gpus=[4,5,6],idle_gpu_preference=[2,3],
                gpu_policy='Fully idle GPUs first; retain acquired idle GPUs for at most two owned workers; then original fallback pool',
                idle_gpu_handoff=str(self.out.parent/'idle-gpu-handoff.json'),idle_gpu_runtime=str(a.policy_runtime),
                active_scheduler_receipt=str(self.out.parent/'scheduler_v5_exit.json'))
            self.save()

        def admit(self):
            if self.state['status'] in ('complete','needs_attention','awaiting_path_review'):return
            running=[j for j in self.state['jobs'].values() if j['status']=='running']
            if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native') for j in running):
                self.state['admission_wait']='serialized initialization';return
            action={'probes':'probe','formal':'train','evaluation':'evaluate'}[self.state['phase']]
            pending=[j for j in self.state['jobs'].values() if j['status']=='pending' and j['action']==action]
            if not pending:return
            now=time.time();rows={r['index']:r for r in inventory(self.project)}
            for gpu,row in rows.items():
                if row['fully_idle'] and not any(j['gpu']==gpu for j in running):self.idle_since.setdefault(gpu,now)
                else:self.idle_since.pop(gpu,None)
            preferred=[2,3]+[g for g in self.state['gpus'] if g not in (2,3)]
            idle=[g for g in preferred if g in self.idle_since and now-self.idle_since[g]>=10]
            if self.idle_since and not idle:
                self.state['admission_wait']='verifying_continuously_idle_gpu';return
            owned=[g for g in self.leases if g not in idle and all(p['project_worker'] for p in rows[g]['processes'])]
            fallback=[g for g in [4,5,6] if g not in idle+owned]
            for j in pending:
                self.state['min_free_gpu_mib']=max(8192,math.ceil(self.state['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
                for gpu in idle+owned+fallback:
                    on_gpu=[r for r in running if r['gpu']==gpu]
                    if len(on_gpu)>=2 or sum(p['project_worker'] for p in rows[gpu]['processes'])>=2:continue
                    slot=next(s for s in (0,1) if not any(r['slot']==s for r in on_gpu))
                    if not self.resources(gpu):continue
                    # Repeat the GPU/process check under the acquired project lock before launch.
                    current=next(r for r in inventory(self.project) if r['index']==gpu)
                    if gpu in idle and not current['fully_idle']:continue
                    if gpu in owned and not all(p['project_worker'] for p in current['processes']):continue
                    if sum(p['project_worker'] for p in current['processes'])>=2:continue
                    reason='fully_idle' if gpu in idle else 'owned_idle_gpu_lease' if gpu in owned else 'original_fallback_pool'
                    if reason=='fully_idle':self.leases[gpu]=dict(acquired_at=now,evidence=current,idle_seconds=now-self.idle_since[gpu])
                    j['gpu_admission']=dict(reason=reason,observed_at=time.time(),evidence=current,
                        existing_queue_workers=len(on_gpu),idle_lease=self.leases.get(gpu),max_workers_per_gpu=2)
                    self.state['idle_gpu_leases']=self.leases
                    self.launch(j,gpu,slot);return
            self.state['admission_wait']='waiting_for_idle_or_authorized_capacity'

    queue=Queue(a);queue.main()
    if queue.state['status']!='complete':raise SystemExit(2)


if __name__=='__main__':main()
