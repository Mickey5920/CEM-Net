"""Continue unchanged CNN workers after explicitly authorized FT cancellation."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import time

RUN='cem_sequence_comparators_20261102'
CNN=('TCN','ModernTCN','LITEMV')


def load_base(project):
    path=project/'work'/RUN/'partition_v1/cem_sequence_partition_runtime.py'
    spec=importlib.util.spec_from_file_location('frozen_partition_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cnn_phases(self):
    base=self.base
    for g in self.state['groups'].values():
        jobs=[j for j in g['jobs'].values() if j['variant'] in CNN];out=Path(g['output'])
        assert not any(j['status'] in ('pending','running') for j in g['jobs'].values() if j['variant']=='FTTransformer')
        if any(j['status']=='failed' for j in jobs):g['status']='needs_attention';continue
        g.update(phase='partitioned',status='running')
        for variant in CNN:
            fits=[j for j in jobs if j['action']=='train' and j['variant']==variant]
            assert {j['id'] for j in fits}==base.method_ids(variant)
            if not all(j['status']=='completed' for j in fits):continue
            selected=base.selections(fits);path=out/'partition_locks'/(variant+'.json')
            if path.exists():assert base.read(path)['selections']==selected
            else:
                for j in fits:
                    assert base.read(j['exit_receipt'])['returncode']==0
                    assert base.sha(Path(j['output'])/'best.pt')==selected[j['id']]['checkpoint_sha256']
                base.save(path,dict(complete=True,variant=variant,test_used_for_selection=False,
                  source_manifest_sha256=g['source_manifest_sha256'],handoff_sha256=self.state['partition_handoff_sha256'],
                  selections=selected,created_at=time.time()))
        if not all(j['status']=='completed' for j in jobs):continue
        path=out/'partition_receipts/cnn.json'
        if not path.exists():
            for j in jobs:assert base.read(j['exit_receipt'])['returncode']==0
            snapshot=copy.deepcopy(g);snapshot.update(jobs={j['id']:copy.deepcopy(j) for j in jobs},status='partition_complete',partition='cnn')
            base.save(path,dict(complete=True,partition='cnn',variants=list(CNN),formal_fits=30,fixed_tests=30,
              host_queue_exit_required=False,host_queue_pid=self.state['pid'],
              handoff_path=str(self.out/'partition_v1/handoff.json'),handoff_sha256=self.state['partition_handoff_sha256'],
              runtime_sha256=self.state['partition_runtime_sha256'],group=snapshot,created_at=time.time()))
        g.update(status='complete',phase='complete',finished_at=time.time(),authorized_active_fits=30,authorized_active_tests=30)


def queue_class(project,host):
    import fcntl
    base=load_base(project);Parent=base.queue_class(project,host)
    from scripts.cem_evidence_host_queue import IdlePolicy
    from scripts.tracehr_recovery_queue import same_process
    class Policy(IdlePolicy):
        def candidates(self,rows,now):return [g for g in super().candidates(rows,now) if host!='eicu' or g!=3]

    class Queue(Parent):
        def __init__(self,a):
            self.a=a;self.project=project;self.base=base
            self.out=project/'runs'/RUN/('host_'+host+'_v3');self.change=self.out/'ft_retirement_v1'
            proof=base.read(self.change/'handoff.json')
            assert proof['complete'] and proof['cnn_workers_stopped']==0 and proof['training_restarts']==0
            assert base.sha(__file__)==proof['runtime_sha256']
            assert base.sha(self.change/'adopted-state.json')==proof['adopted_state_sha256']
            assert base.sha(proof['predecessor_exit_path'])==proof['predecessor_exit_sha256']
            assert base.read(proof['predecessor_exit_path'])['returncode']==-15
            assert base.sha(base.__file__)==proof['parent_runtime_sha256']
            self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state=base.read(self.change/'adopted-state.json');self.handles={};self.locks={};cards=set()
            for path,h in proof['protected'].items():assert base.sha(path)==h,path
            for g in self.state['groups'].values():
                for j in g['jobs'].values():
                    if j['status']!='running':continue
                    assert j['variant'] in CNN
                    f=Path(j['exit_receipt']);owner=base.read(f.with_name(f.stem+'_owner.json'))
                    assert owner==proof['cnn_workers'][g['database']+':'+j['id']]['owner']
                    assert (f.exists() and base.read(f)['returncode']==0) or same_process(owner['child_pid'],owner['child_identity'])
                    cards.add(j['gpu'])
            for gpu in cards:
                f=project/'work/tracehr_fullstream_20261003/locks'/f'gpu{gpu}.lock';handle=f.open('a+')
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);self.locks[gpu]=handle
            self.idle=Policy([2,4,6] if host=='eicu' else [0,1,2,3])
            self.idle.since={int(k):v for k,v in self.state.get('idle_observations',{}).items()}
            self.idle.leases={int(k):v for k,v in self.state.get('idle_leases',{}).items() if host!='eicu' or int(k)!=3}
            self.idle.last=self.state['updated_at']
            self.state.update(pid=os.getpid(),active_scheduler_receipt=str(self.change/'exit.json'),
              retirement_handoff_sha256=base.sha(self.change/'handoff.json'),retirement_runtime_sha256=base.sha(__file__),
              active_variants=list(CNN),retired_variants=['FTTransformer'],authorized_active_fits=30 if host=='eicu' else 60,
              scheduler_adopted_at=time.time())
            self.save()

        phases=cnn_phases
    return Queue


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--host',choices=('mimic','eicu'),required=True)
    p.add_argument('--self-test',action='store_true');a=p.parse_args();project=a.project.resolve(strict=True)
    cls=queue_class(project,a.host)
    if a.self_test:
        import json
        print(json.dumps(dict(complete=True,unchanged_parent_evaluator_and_admission=True,training_started=False)))
    else:cls(a).main()


if __name__=='__main__':main()
