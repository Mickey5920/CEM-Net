"""One host owner; inherited GPU leases and fixed-checkpoint lifecycle."""
import argparse
import copy,fcntl,os,time
from pathlib import Path
from scripts import cem_icde72_queue as previous
from scripts.cem_sequence_comparator_common import RUN,read,sha

previous.prior.RUN=RUN


class Queue(previous.Queue):
    def __init__(self,a):
        self.a=a;self.project=a.project.resolve(strict=True);self.out=self.project/'runs'/RUN/('host_'+a.host+'_v3')
        self.out.mkdir(parents=True,exist_ok=True);self.owner=(self.out/'owner.lock').open('a+')
        fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (self.out/'state.json').exists()
        old=self.project/'runs'/RUN/('host_'+a.host+'_v2');handoff=read(old/'kernel-handoff.json')
        assert handoff['complete'] and handoff['active_workers']==0 and read(old/'start_exit.json')['returncode']==-15
        assert sha(old/'kernel-predecessor-state.json')==handoff['state_sha256']
        self.handles={};self.locks={};self.idle=previous.prior.IdlePolicy([2,3] if a.host=='eicu' else [0,1,2,3])
        self.state=dict(pid=os.getpid(),host=a.host,started_at=time.time(),status='waiting_for_verified_groups',
            preferred_gpus=self.idle.preferred,other_gpu_idle_seconds=900,max_workers_per_gpu=2,max_cpu_workers=1,
            expected_groups=1 if a.host=='eicu' else 2,authorized_formal_fits=40 if a.host=='eicu' else 80,
            deferred_variants=['TVNet'],groups={},policy_checks=previous.prior.policy_checks(),prior_handoff_sha256=sha(old/'kernel-handoff.json'))

    def discover(self):
        previous_keys=set(self.state['groups'])
        super().discover()
        for key,g in self.state['groups'].items():
            assert len(g['matrix'])==40 and 'TVNet' not in g['variants']
            if key not in previous_keys:
                old=read(Path(g['preflight_parent_output'])/'state.json')
                for jid,j in list(g['jobs'].items()):
                    if j['action']=='probe' and j['variant']!='FTTransformer':
                        parent=old['jobs'][jid];assert parent['status']=='completed' and read(parent['exit_receipt'])['returncode']==0
                        assert parent['result']['identity']['source_manifest_sha256']==g['preflight_parent_manifest']
                        g['jobs'][jid]=copy.deepcopy(parent)
                        g['jobs'][jid]['reused_completed_preflight']=True

    def env(self,g,gpu):
        env=super().env(g,gpu)
        env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        return env


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--host',choices=('mimic','eicu'),required=True)
    Queue(p.parse_args()).main()
