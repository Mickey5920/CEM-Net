"""Reuse the existing host owner, leases, attempts and phase transitions."""
import argparse
import fcntl
import math
import shutil
import time
from pathlib import Path
from scripts import cem_evidence_host_queue as host
from scripts.cem_icde72_queue import Queue as Previous

host.RUN = 'cem_matched_controls_20261105'


class Queue(Previous):
    def __init__(self, a):
        super().__init__(a)
        self.state.update(expected_groups=1 if a.host == 'eicu' else 2,
                          authorized_formal_fits=20 if a.host == 'eicu' else 40,
                          max_workers_per_gpu=1, max_concurrent_workers=4)

    def admit(self, rows):
        running = [j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status'] == 'running']
        if len(running) >= 4:
            return
        if any(self.progress(j).get('status') in (None, 'loading_verified_inputs', 'randomly_initializing_native') for j in running):
            self.state['admission_wait'] = 'serialized initialization'; return
        mem = {k: int(v.split()[0])*1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        now = time.time(); by = {r['index']: r for r in rows}
        for key, g in sorted(self.state['groups'].items(), key=lambda item: item[1].get('last_launch', 0)):
            if g['status'] not in ('preflight', 'running', 'evaluating'):
                continue
            action = {'probes': 'probe', 'formal': 'train', 'evaluation': 'evaluate'}[g['phase']]
            pending = [j for j in g['jobs'].values() if j['status'] == 'pending' and j['action'] == action]
            if not pending or mem['MemAvailable']/2**30 < 24 or shutil.disk_usage(self.project).free/2**30 < 10+g['expected_gib']:
                continue
            j = pending[0]
            needed = 16384 if action == 'probe' else max(8192, math.ceil(g['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
            for gpu in self.idle.candidates(rows, now):
                row = by[gpu]
                if any(x['gpu'] == gpu for x in running) or row['processes'] or row['free_mib'] < needed:
                    continue
                if gpu not in self.locks:
                    path = self.project/'work/tracehr_fullstream_20261003/locks'/f'gpu{gpu}.lock'
                    path.parent.mkdir(parents=True, exist_ok=True); handle = path.open('a+')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close(); continue
                    self.locks[gpu] = handle
                current = next(x for x in host.inventory(self.project) if x['index'] == gpu)
                if current['processes'] or current['free_mib'] < needed:
                    continue
                if gpu not in self.idle.preferred and gpu not in self.idle.leases and not current['fully_idle']:
                    continue
                evidence = self.idle.acquire(gpu, current, now)
                evidence.update(required_free_mib=needed, one_owned_worker_per_gpu=True)
                self.launch(key, g, j, gpu, 0, evidence)
                g['last_launch'] = time.time(); self.state['admission_wait'] = None
                return
        self.state['admission_wait'] = 'waiting for free authorized or continuously idle leased GPU'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--host', choices=('mimic', 'eicu'), required=True)
    Queue(p.parse_args()).main()
