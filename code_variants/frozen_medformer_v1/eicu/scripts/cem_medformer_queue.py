"""Independent queue, sharing the existing cross-run GPU leases."""
import argparse
from pathlib import Path
from scripts import cem_icde72_queue as previous
from scripts.cem_medformer_common import RUN,read

previous.prior.RUN=RUN


class Policy(previous.prior.IdlePolicy):
    def __init__(self,host):
        super().__init__([2,4,6] if host=='eicu' else [0,1,2,3])
        self.excluded={3} if host=='eicu' else set()

    def candidates(self,rows,now):
        return [g for g in super().candidates(rows,now) if g not in self.excluded]


class Queue(previous.Queue):
    def __init__(self,a):
        super().__init__(a)
        self.idle=Policy(a.host)
        self.state.update(expected_groups=1 if a.host=='eicu' else 2,
            authorized_formal_fits=10 if a.host=='eicu' else 20,
            preferred_gpus=self.idle.preferred,excluded_gpus=sorted(self.idle.excluded))

    def discover(self):
        super().discover()
        for g in self.state['groups'].values():
            assert g['variants']==['Medformer'] and len(g['matrix'])==10

    def admit(self,rows):
        waiting=[]
        for g in self.state['groups'].values():
            cfg=read(Path(g['source'])/'work'/RUN/'config.json')
            if g['phase']=='evaluation' and not (self.project/cfg['hourly_cache']/'test_completion.json').exists():
                waiting.append((g,g['status']));g['status']='waiting_for_existing_CNN_test_cache'
        super().admit(rows)
        for g,status in waiting:g['status']=status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--host',choices=('mimic','eicu'),required=True)
    Queue(p.parse_args()).main()
