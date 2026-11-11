"""Existing host queue with explicit missing-cell matrix and seed-51 probes."""
import argparse, fcntl, os, time
from pathlib import Path
from scripts import core_seed_host_queue as host
from scripts.core_seed_icde_queue import Queue as Previous
from scripts.core_seed_host_queue import read, sha, save

host.RUN = 'cem_core_five_seed_20261107'

class Queue(Previous):
    def __init__(self, args):
        if not args.recover:
            super().__init__(args)
        else:
            self.a=args;self.project=args.project.resolve(strict=True)
            self.out=self.project/'runs'/host.RUN/('host_'+args.host)
            self.owner=(self.out/'owner.lock').open('a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state=read(self.out/'state.json');self.handles={};self.locks={}
            self.idle=host.IdlePolicy(self.state['preferred_gpus'])
            self.idle.leases={int(k):v for k,v in self.state.get('idle_leases',{}).items()}
            for file in (self.out/'replacements').glob('*.json'):
                key=file.stem;old=self.state['groups'][key]
                assert all(j['status']=='pending' for j in old['jobs'].values() if j['action']=='train')
                assert not any(j['status']=='running' for j in old['jobs'].values())
                save(self.out/'superseded'/(key+'_'+str(int(time.time()))+'.json'),old)
                save(self.out/'ready'/file.name,read(file));del self.state['groups'][key]
                file.rename(file.with_suffix('.applied'))
            for key,g in self.state['groups'].items():
                estimate=g.get('projected_artifact_gib',0)
                if g['stage']=='controls':
                    required=100 if g['database']=='eicu' else 65
                    assert required>estimate
                    g['storage_budget_extension']=dict(previous_gib=g['expected_gib'],new_gib=required,
                        basis='measured original-path checkpoint sizes; unchanged model and retention')
                    g['expected_gib']=required
            self.state.update(pid=os.getpid(),resumed_at=time.time(),status='running')
        self.state.update(expected_groups=7 if args.host=='eicu' else 12,
                          max_concurrent_workers=4, max_cpu_workers=1)

    def discover(self):
        for file in sorted((self.out/'ready').glob('*.json')):
            key = file.stem
            if key in self.state['groups']:
                assert sha(file) == self.state['groups'][key]['ready_sha256']; continue
            d = read(file); source, out = Path(d['source']), Path(d['output'])
            assert source.resolve().is_relative_to(self.project) and out.resolve().is_relative_to(self.project)
            assert sha(source/'source_manifest.json') == d['source_manifest_sha256']
            for name, digest in read(source/'source_manifest.json').items(): assert sha(source/name) == digest
            v = read(out.parent/'verification.json')
            assert v['complete'] and v['formal_fits'] == len(d['matrix']) and v['source_manifest_sha256'] == d['source_manifest_sha256']
            assert not (out/'state.json').exists()
            group = dict(**d, ready_sha256=sha(file), phase='probes', status='preflight', jobs={})
            for cell in d['matrix']:
                variant, seed, endpoint = cell['variant'], cell['seed'], cell['endpoint']
                assert seed in (51,52)
                name = f'{variant}_t{endpoint}_s{seed}'
                actions = [('train','','jobs'), ('evaluate','test_','test')]
                if seed == 51: actions.insert(0, ('probe','probe_','pilots'))
                for action,prefix,folder in actions:
                    jid = prefix+name
                    group['jobs'][jid] = dict(id=jid, **cell, action=action, status='pending', output=str(out/folder/jid),
                        device='cpu' if variant.startswith(('LightGBM','XGBoost')) else 'gpu', fit_id=name,
                        pilot_id=f'probe_{variant}_t{endpoint}_s51')
            self.state['groups'][key] = group

    def env(self, group, gpu):
        env = super().env(group, gpu)
        env['CEM_CORE_VERIFICATION'] = str(Path(group['output']).parent/'verification.json')
        return env

    def phases(self):
        super().phases()
        # Automated application of the same factual release checks used by
        # the existing launchers; user already authorized these seed-only fits.
        for g in self.state['groups'].values():
            if g['status'] != 'awaiting_path_review': continue
            audits = {}
            for job in g['jobs'].values():
                if job['action'] != 'probe': continue
                path = Path(job['output'])/'path_audit.json'; a = read(path)
                assert a['complete'] and not job['result']['test_loaded']
                if job['device']=='cpu':
                    assert a['continuous_tree_updates']==20 and a['reload_prediction_exact']
                else:
                    assert a['continuous_updates']==20
                    assert a.get('resume_raw_optimizer_ema_rng_exact',a.get('resume_exact',False))
                audits[job['id']] = sha(path)
            assert g['projected_artifact_gib'] < g['expected_gib']
            save(Path(g['output'])/'formal_release.json',dict(complete=True,
                source_manifest_sha256=g['source_manifest_sha256'],audit_sha256=audits,
                projected_artifact_gib=g['projected_artifact_gib'],basis='unchanged verified path; seeds 51/52 authorized'))

    def admit(self, rows):
        active = [j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status']=='running']
        if len(active) >= self.state['max_concurrent_workers']: return
        super().admit(rows)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--host',choices=('mimic','eicu'),required=True);p.add_argument('--recover',action='store_true')
    Queue(p.parse_args()).main()
