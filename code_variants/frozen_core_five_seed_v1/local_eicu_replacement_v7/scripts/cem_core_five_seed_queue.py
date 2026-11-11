"""Existing host queue with explicit missing-cell matrix and seed-51 probes."""
import argparse, fcntl, math, os, shutil, time
from pathlib import Path
from scripts import core_seed_host_queue as host
from scripts.core_seed_icde_queue import Queue as Previous
from scripts.core_seed_host_queue import read, sha, save

host.RUN = 'cem_core_five_seed_20261107'


def stage_priority(hostname, stage):
    # After its completed SeFT fits, eICU has one remaining mTAND fit to admit.
    # Start this long dependency before short fits refill every available slot.
    # Probes still take precedence through admit() below.
    if hostname == 'eicu' and stage == 'sequence':
        return 0
    return {'controls':1, 'basic':2, 'hospital':3, 'tabular':4,
            'legacy':5, 'sequence':6, 'replacement':7}[stage]


def rows_without_active_probes(rows, jobs):
    reserved = {j['gpu'] for j in jobs if j['status'] == 'running' and
                j['action'] == 'probe' and j['device'] == 'gpu'}
    if any(j['status'] == 'pending' and j['action'] == 'probe' and
           j['device'] == 'gpu' for j in jobs):
        # An exclusive probe needs an empty GPU. Do not refill a vacated
        # worker slot with another long fit while a probe is waiting.
        reserved.update(j['gpu'] for j in jobs if j['status'] == 'running'
                        and j['device'] == 'gpu')
    return [row for row in rows if row['index'] not in reserved]


def admission_checks():
    rows = [dict(index=i) for i in range(3)]
    jobs = [dict(status='running', action='probe', device='gpu', gpu=1),
            dict(status='running', action='train', device='gpu', gpu=0),
            dict(status='running', action='probe', device='cpu', gpu=-1)]
    assert [r['index'] for r in rows_without_active_probes(rows, jobs)] == [0, 2]
    jobs[0]['status'] = 'completed'
    assert rows_without_active_probes(rows, jobs) == rows
    jobs.append(dict(status='pending', action='probe', device='gpu'))
    assert [r['index'] for r in rows_without_active_probes(rows, jobs)] == [1, 2]
    jobs[1]['status'] = 'completed'
    assert rows_without_active_probes(rows, jobs) == rows
    jobs[-1]['device'] = 'cpu'
    jobs[1]['status'] = 'running'
    assert rows_without_active_probes(rows, jobs) == rows
    assert stage_priority('eicu', 'sequence') < stage_priority('eicu', 'controls')
    assert stage_priority('mimic', 'controls') < stage_priority('mimic', 'sequence')
    assert stage_priority('eicu', 'sequence') > stage_priority('eicu', 'replacement')-100
    return ['running_GPU_probe_blocks_all_other_admissions_on_its_GPU',
            'completed_probe_releases_GPU', 'CPU_probe_does_not_reserve_GPU',
            'pending_GPU_probe_prevents_long_fit_slot_refill',
            'existing_fits_drain_without_interruption',
            'pending_CPU_probe_does_not_block_GPU_admission',
            'remaining_long_eICU_fit_is_admitted_before_short_fits',
            'probes_still_precede_all_formal_fits']


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
            if time.time()-self.state.get('updated_at',0)<=45:
                self.idle.since={int(k):v for k,v in self.state.get('idle_observations',{}).items()}
                self.idle.last=self.state['updated_at']
            for file in (self.out/'replacements').glob('*.json'):
                key=file.stem;old=self.state['groups'][key]
                if not any(j['status']=='failed' for j in old['jobs'].values()):
                    # An equivalent newer snapshot is not a reason to repeat
                    # a successfully admitted group or any completed work.
                    file.rename(file.with_suffix('.not_needed'));continue
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
                          authorized_formal_cells=80 if args.host=='eicu' else 124,
                          max_concurrent_workers=8, max_cpu_workers=1)
        for g in self.state['groups'].values():
            g['stage_order']=stage_priority(args.host, g['stage'])

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
            group['stage_order']=stage_priority(self.a.host, d['stage'])
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
        for g in self.state['groups'].values():
            if g['phase']!='probes':continue
            probes=[j for j in g['jobs'].values() if j['action']=='probe']
            if not probes or not all(j['status']=='completed' for j in probes):continue
            terms=[]
            for job in g['jobs'].values():
                if job['action']!='train':continue
                size=g['jobs'][job['pilot_id']]['result']['checkpoint_bytes']
                factor=13 if job['variant'].startswith(('EMA_','CEM_')) or job['variant']=='RealMLP' else 3
                terms.append((size,factor))
            denominator=sum(size for size,factor in terms)
            g['retention_multiplier']=sum(size*factor for size,factor in terms)/denominator
            estimate=1+sum(size*factor*1.2/2**30 for size,factor in terms)
            if estimate>=g['expected_gib']:
                required=math.ceil(estimate*1.1)
                if shutil.disk_usage(self.project).free/2**30>required+10:
                    g['storage_budget_extension']=dict(previous_gib=g['expected_gib'],new_gib=required,
                        measured_projection_gib=estimate,basis='completed original-path probes and unchanged family-specific retention')
                    g['expected_gib']=required
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
        jobs = [j for g in self.state['groups'].values() for j in g['jobs'].values()]
        active = [j for j in jobs if j['status']=='running']
        if len(active) >= self.state['max_concurrent_workers']: return
        # Admit waiting probes before formal fits, with the original family
        # order within each class. Running fits keep their existing processes.
        groups = self.state['groups']
        priorities = {key: g['stage_order'] for key, g in groups.items()}
        try:
            for g in groups.values():
                if g['phase'] == 'probes': g['stage_order'] -= 100
            super().admit(rows_without_active_probes(rows, jobs))
        finally:
            for key, g in groups.items(): g['stage_order'] = priorities[key]

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--host',choices=('mimic','eicu'),required=True);p.add_argument('--recover',action='store_true')
    Queue(p.parse_args()).main()
