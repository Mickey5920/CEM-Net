"""Reuse the existing host scheduler and GPU policy for 51 additional fits."""
import argparse
from pathlib import Path
from scripts import cem_evidence_host_queue as host
from scripts.cem_icde72_queue import Queue as Previous
from scripts.cem_evidence_host_queue import read, sha

host.RUN = 'cem_targeted_evidence_20261106'


class Queue(Previous):
    def __init__(self, args):
        super().__init__(args)
        self.state.update(expected_groups=1 if args.host == 'eicu' else 2,
                          authorized_formal_fits=17 if args.host == 'eicu' else 34,
                          max_concurrent_workers=4)

    def discover(self):
        for file in sorted((self.out/'ready').glob('*.json')):
            key = file.stem
            if key in self.state['groups']:
                assert sha(file) == self.state['groups'][key]['ready_sha256']
                continue
            d = read(file)
            source, out = Path(d['source']), Path(d['output'])
            assert source.resolve().is_relative_to(self.project) and out.resolve().is_relative_to(self.project)
            assert sha(source/'source_manifest.json') == d['source_manifest_sha256']
            for name, digest in read(source/'source_manifest.json').items():
                assert sha(source/name) == digest, name
            verification = read(out.parent/'verification.json')
            assert verification['complete'] and verification['source_manifest_sha256'] == d['source_manifest_sha256']
            assert verification['formal_fits'] == len(d['matrix']) == 17 and not (out/'state.json').exists()
            pilots = {(v, e):min(c['seed'] for c in d['matrix'] if c['variant'] == v and c['endpoint'] == e)
                      for v, e in {(c['variant'], c['endpoint']) for c in d['matrix']}}
            group = dict(**d, ready_sha256=sha(file), phase='probes', status='preflight', jobs={})
            for cell in d['matrix']:
                variant, seed, endpoint = cell['variant'], cell['seed'], cell['endpoint']
                name = f'{variant}_t{endpoint}_s{seed}'
                actions = [('train', '', 'jobs'), ('evaluate', 'test_', 'test')]
                if seed == pilots[variant, endpoint]:
                    actions.insert(0, ('probe', 'probe_', 'pilots'))
                for action, prefix, folder in actions:
                    jid = prefix+name
                    group['jobs'][jid] = dict(id=jid, **cell, action=action, status='pending',
                                             output=str(out/folder/jid), device='gpu', fit_id=name,
                                             pilot_id=f'probe_{variant}_t{endpoint}_s{pilots[variant, endpoint]}')
            self.state['groups'][key] = group

    def admit(self, rows):
        active = [j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status'] == 'running']
        if len(active) >= self.state['max_concurrent_workers']:
            self.state['admission_wait'] = 'four owned workers already active'
            return
        super().admit(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--host', choices=('mimic', 'eicu'), required=True)
    Queue(p.parse_args()).main()
