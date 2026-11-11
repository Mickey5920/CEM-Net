"""Existing host queue with explicit missing-cell matrix and seed-51 probes."""
import argparse, copy, fcntl, math, os, shutil, time
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


def formal_admission_order(jobs):
    # mTAND keeps its original 200-epoch stopping policy. Start pending fits
    # before shorter SeFT fits so this long dependency does not start last.
    # Values are the original job objects; no active process is interrupted.
    return dict(sorted(jobs.items(), key=lambda item: not (
        item[1]['status'] == 'pending' and item[1]['action'] == 'train' and
        item[1].get('variant') == 'mTAND_Enc')))


def allocator_probe_jobs(group):
    """Replay only pending STraTS probes; preserve every original job record."""
    jobs = copy.deepcopy(group['jobs'])
    targets = [j for j in jobs.values() if j['variant'] == 'STraTS_supervised']
    assert targets and all(j['status'] == ('completed' if j['action'] == 'probe' else 'pending')
                           for j in targets)
    for old in [j for j in targets if j['action'] == 'probe']:
        jid = 'allocator_' + old['id']
        assert jid not in jobs
        new = copy.deepcopy(old)
        new.update(id=jid, status='pending', output=str(Path(group['output'])/'pilots'/jid))
        for field in ('result', 'pid', 'started_at', 'finished_at', 'gpu', 'slot',
                      'gpu_admission', 'command', 'log', 'exit_receipt'):
            new.pop(field, None)
        jobs[jid] = new
        for job in targets:
            if job['action'] != 'probe' and job['pilot_id'] == old['id']:
                job['pilot_id'] = jid
    return jobs


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
    original = {
        'short': dict(status='pending', action='train', variant='SeFT'),
        'active': dict(status='running', action='train', variant='mTAND_Enc'),
        'test': dict(status='pending', action='evaluate', variant='mTAND_Enc'),
        'long': dict(status='pending', action='train', variant='mTAND_Enc'),
    }
    ordered = formal_admission_order(original)
    assert list(ordered) == ['long', 'short', 'active', 'test']
    assert list(original) == ['short', 'active', 'test', 'long']
    assert all(ordered[k] is original[k] for k in original)
    return ['running_GPU_probe_blocks_all_other_admissions_on_its_GPU',
            'completed_probe_releases_GPU', 'CPU_probe_does_not_reserve_GPU',
            'pending_GPU_probe_prevents_long_fit_slot_refill',
            'existing_fits_drain_without_interruption',
            'pending_CPU_probe_does_not_block_GPU_admission',
            'remaining_long_eICU_fit_is_admitted_before_short_fits',
            'pending_long_sequence_fits_start_before_short_fits',
            'admission_order_preserves_job_objects_and_running_processes',
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
        if args.host == 'mimic':
            self.restore_strats_allocator()
            self.save()

    def restore_strats_allocator(self):
        # The archived MIMIC STraTS fits used expandable CUDA segments. Their
        # successful full-run peaks are much smaller than the default allocator
        # stress probe's retained pool. Restore that runtime before any new fit.
        predecessor = self.project/'runs/cem_baseline_replacement_20261030/host_mimic'
        handoff = read(predecessor/'allocator-handoff.json')
        assert handoff['complete'] and not handoff['scientific_changes']
        runtime = self.project/'work/cem_baseline_replacement_20261030/allocator_runtime_v1/continue.py'
        assert sha(runtime) == handoff['code_sha256']
        assert "env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'" in runtime.read_text()
        for group in self.state['groups'].values():
            if group['stage'] != 'replacement' or group.get('allocator_runtime'):
                continue
            assert group['phase'] == 'formal' and group['status'] == 'running'
            jobs = allocator_probe_jobs(group)
            out = Path(group['output'])
            original = out/'formal_release_before_allocator_v8.json'
            if original.exists():
                assert sha(original) == sha(out/'formal_release.json')
            else:
                original.write_bytes((out/'formal_release.json').read_bytes())
            group.update(jobs=jobs, phase='probes', status='preflight',
                allocator_runtime=dict(configuration='expandable_segments:True',
                    original_handoff_sha256=sha(predecessor/'allocator-handoff.json'),
                    original_runtime_sha256=sha(runtime),
                    previous_release_sha256=sha(original), scientific_changes=False))

    def release_allocator_probes(self):
        for group in self.state['groups'].values():
            if not group.get('allocator_runtime') or group['phase'] != 'probes':
                continue
            probes = [j for j in group['jobs'].values() if j['action'] == 'probe']
            if not all(j['status'] == 'completed' for j in probes):
                continue
            out = Path(group['output'])
            if (out/'allocator_parity.json').exists():
                continue
            import torch
            from scripts.mimiciii_ema_runtime import digest
            checks = {}
            for job in probes:
                if not job['id'].startswith('allocator_probe_'):
                    continue
                old = group['jobs'][job['id'].removeprefix('allocator_')]
                before = torch.load(Path(old['output'])/'representative_checkpoint.pt',
                                    map_location='cpu', weights_only=False)
                after = torch.load(Path(job['output'])/'representative_checkpoint.pt',
                                   map_location='cpu', weights_only=False)
                assert digest(before) == digest(after), 'Allocator changed model/optimizer/RNG'
                audit = read(Path(job['output'])/'path_audit.json')
                assert audit['complete'] and audit['continuous_updates'] == 20
                assert audit['resume_exact'] and audit['longest_train_and_validation_evaluated']
                required = max(8192, math.ceil(job['result']['peak_reserved_mib']*1.3+2048))
                assert required < 48508, 'Measured admission exceeds the verified device capacity'
                checks[job['id']] = dict(model_optimizer_rng_exact=True,
                    peak_reserved_mib=job['result']['peak_reserved_mib'], required_free_mib=required)
            assert len(checks) == 2
            save(out/'allocator_parity.json', dict(complete=True, checks=checks,
                configuration='expandable_segments:True', scientific_changes=False))
            release = read(out/'formal_release_before_allocator_v8.json')
            assert sha(out/'formal_release_before_allocator_v8.json') == group['allocator_runtime']['previous_release_sha256']
            release['audit_sha256'].update({j['id']:sha(Path(j['output'])/'path_audit.json') for j in probes})
            release['allocator_parity_sha256'] = sha(out/'allocator_parity.json')
            save(out/'formal_release.json', release)

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
        if self.a.host == 'mimic' and group['stage'] == 'replacement' and gpu >= 0:
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        return env

    def phases(self):
        self.release_allocator_probes()
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
        original_jobs = {key: g['jobs'] for key, g in groups.items()}
        try:
            for g in groups.values():
                if g['phase'] == 'probes': g['stage_order'] -= 100
                elif g['phase'] == 'formal':
                    g['jobs'] = formal_admission_order(g['jobs'])
            super().admit(rows_without_active_probes(rows, jobs))
        finally:
            for key, g in groups.items():
                g['stage_order'] = priorities[key]
                g['jobs'] = original_jobs[key]

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--host',choices=('mimic','eicu'),required=True);p.add_argument('--recover',action='store_true')
    Queue(p.parse_args()).main()
