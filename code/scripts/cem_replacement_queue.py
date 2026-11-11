"""Reuse existing GPU leases, CPU admission, probe gates and locked evaluation."""
import argparse
from pathlib import Path
from scripts import cem_icde72_queue as previous
from scripts.cem_replacement_common import RUN

previous.prior.RUN=RUN
class Queue(previous.Queue):
    def __init__(self,a):
        super().__init__(a)
        self.state['expected_groups']=1 if a.host=='eicu' else 2
        self.state['authorized_formal_fits']=12 if a.host=='eicu' else 24
    def discover(self):
        super().discover()
        for g in self.state['groups'].values():
            assert len(g['matrix'])==12
            for j in g['jobs'].values():j['device']='cpu' if j['variant']=='XGBoost_TD' else 'gpu'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--host',choices=('mimic','eicu'),required=True)
    Queue(p.parse_args()).main()
