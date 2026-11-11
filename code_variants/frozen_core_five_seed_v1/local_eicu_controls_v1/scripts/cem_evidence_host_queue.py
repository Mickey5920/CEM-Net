"""One host owner for the approved CEM-Net completion groups and GPU leases."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

RUN = 'cem_evidence_completion_20261025'
STATUSES = ('pending', 'running', 'completed', 'failed')


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name+'.writing'); q.write_text(json.dumps(x, indent=2)); os.replace(q, p)


def inventory(project):
    xml = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=20))
    rows = []
    for index, g in enumerate(xml.findall('gpu')):
        processes = []
        for p in g.findall('./processes/process_info'):
            pid = int(p.findtext('pid'))
            try: cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError: cmd = ''
            processes.append(dict(pid=pid, project_worker=str(project) in cmd, identity_known=bool(cmd)))
        used = int(g.findtext('./fb_memory_usage/used').split()[0])
        free = int(g.findtext('./fb_memory_usage/free').split()[0])
        util = int(g.findtext('./utilization/gpu_util').split()[0])
        rows.append(dict(index=index, used_mib=used, free_mib=free, utilization=util, processes=processes,
                         fully_idle=not processes and used <= 256 and util == 0))
    return rows


class IdlePolicy:
    def __init__(self, preferred):
        self.preferred = preferred; self.since = {}; self.leases = {}; self.last = None

    def observe(self, rows, now):
        # Missing observations cannot establish a continuous fifteen-minute interval.
        if self.last is None or now-self.last > 45: self.since.clear()
        self.last = now
        for r in rows:
            g = r['index']
            if r['fully_idle']: self.since.setdefault(g, now)
            else: self.since.pop(g, None)
            if g in self.leases and any(not p['project_worker'] for p in r['processes']):
                self.leases.pop(g)

    def candidates(self, rows, now):
        by = {r['index']: r for r in rows}
        preferred = [g for g in self.preferred if g in by]
        others = [r['index'] for r in rows if r['index'] not in preferred and
                  (r['index'] in self.leases or now-self.since.get(r['index'], now) >= 900)]
        return preferred + others

    def acquire(self, gpu, row, now):
        if gpu not in self.preferred and gpu not in self.leases:
            assert row['fully_idle'] and now-self.since[gpu] >= 900
            self.leases[gpu] = dict(acquired_at=now, idle_since=self.since[gpu], idle_seconds=now-self.since[gpu])
        return dict(preferred=gpu in self.preferred, lease=self.leases.get(gpu), observed=row, time=now)


def policy_checks():
    p = IdlePolicy([2, 3]); r = dict(index=5, fully_idle=True, processes=[])
    for t in range(0, 900, 15):
        p.observe([r], t); assert 5 not in p.candidates([r], t)
    p.observe([r], 900); assert 5 in p.candidates([r], 900)
    p.acquire(5, r, 900)
    p.observe([dict(r, fully_idle=False, processes=[dict(project_worker=True)])], 915)
    assert 5 in p.candidates([r], 915)
    p.observe([dict(r, fully_idle=False, processes=[dict(project_worker=False)])], 930)
    assert 5 not in p.candidates([r], 930)
    p.observe([r], 945); p.observe([r], 2000); assert 5 not in p.candidates([r], 2000)
    assert p.candidates([dict(r, index=2)], 2000) == [2]
    return ['other_gpu_requires_900_continuous_seconds', 'external_activity_resets_lease',
            'observation_gap_resets_timer', 'preferred_pool_not_blocked_by_waiting_idle_timer']


class Queue:
    def __init__(self, a):
        self.a = a; self.project = a.project.resolve(strict=True)
        self.out = self.project/'runs'/RUN/('host_'+a.host)
        self.out.mkdir(parents=True, exist_ok=True)
        self.owner = (self.out/'owner.lock').open('a+')
        fcntl.flock(self.owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (self.out/'state.json').exists(), 'Existing queue requires audited recovery, never duplicate'
        self.handles = {}; self.locks = {}; self.idle = IdlePolicy([2, 3] if a.host == 'eicu' else [0, 1, 2, 3])
        self.state = dict(pid=os.getpid(), host=a.host, started_at=time.time(), status='waiting_for_verified_groups',
                          preferred_gpus=self.idle.preferred, other_gpu_idle_seconds=900, max_workers_per_gpu=2,
                          expected_groups=3 if a.host == 'eicu' else 6, groups={}, policy_checks=policy_checks())

    def save(self):
        self.state['updated_at'] = time.time()
        self.state['idle_observations'] = self.idle.since
        self.state['idle_leases'] = self.idle.leases
        for g in self.state['groups'].values():
            for action, key in [('train','counts'),('probe','probe_counts'),('evaluate','evaluation_counts')]:
                g[key] = {s: sum(j['action']==action and j['status']==s for j in g['jobs'].values()) for s in STATUSES}
            save(Path(g['output'])/'state.json', g)
        save(self.out/'state.json', self.state)

    def discover(self):
        for file in sorted((self.out/'ready').glob('*.json')):
            key = file.stem
            if key in self.state['groups']:
                assert sha(file) == self.state['groups'][key]['ready_sha256']; continue
            d = read(file); source = Path(d['source']); out = Path(d['output'])
            assert source.resolve().is_relative_to(self.project) and out.resolve().is_relative_to(self.project)
            assert sha(source/'source_manifest.json') == d['source_manifest_sha256']
            for n, h in read(source/'source_manifest.json').items(): assert sha(source/n) == h, n
            verification = read(out.parent/'verification.json')
            assert verification['complete'] and verification['source_manifest_sha256'] == d['source_manifest_sha256']
            assert verification['formal_fits'] == len(d['variants'])*6
            assert not (out/'state.json').exists()
            g = dict(**d, ready_sha256=sha(file), phase='probes', status='preflight', jobs={})
            for seed in (48,49,50):
                for v in d['variants']:
                    for e in (0,1):
                        name = f'{v}_t{e}_s{seed}'
                        actions = [('train','','jobs'),('evaluate','test_','test')]
                        if seed == 48: actions.insert(0, ('probe','probe_','pilots'))
                        for action,prefix,folder in actions:
                            jid = prefix+name
                            g['jobs'][jid] = dict(id=jid, variant=v, seed=seed, endpoint=e, action=action,
                                status='pending', output=str(out/folder/jid), pilot_id=f'probe_{v}_t{e}_s48', fit_id=name)
            self.state['groups'][key] = g

    def reap(self):
        for key, g in self.state['groups'].items():
            for j in g['jobs'].values():
                if j['status'] != 'running': continue
                receipt = Path(j['exit_receipt'])
                if not receipt.exists():
                    handle = self.handles.get((key,j['id']))
                    if handle is not None and handle.poll() is not None:
                        j.update(status='failed', error='Wrapper exited without child receipt')
                    continue
                rc = read(receipt)
                try:
                    assert rc['returncode'] == 0, f"Child exit {rc['returncode']}"
                    out = Path(j['output']); r = read(out/'completion.json'); identity = r.get('identity', r)
                    assert r['complete'] and identity['source_manifest_sha256'] == g['source_manifest_sha256']
                    for name in ('variant','seed','endpoint'): assert identity[name] == j[name]
                    if j['action'] == 'probe':
                        audit = read(out/'path_audit.json'); assert audit['complete'] and audit['continuous_updates'] == 20
                        assert audit.get('resume_raw_optimizer_ema_rng_exact', audit.get('resume_exact', False))
                        assert not r['test_loaded'] and r['from_scratch']
                    elif j['action'] == 'train':
                        assert not r['pilot'] and r['from_scratch'] and not r['test_loaded']
                        assert sha(out/'best.pt') == r['checkpoint_sha256']
                        assert sha(out/'predictions_dev.npz') == r['predictions_sha256']
                        assert r['checkpoint_replay']['max_probability_drift'] <= 1e-5
                    else:
                        assert r['optimizer_updates'] == 0 and not r['parameters_updated']
                        assert r['dev_replay_max_drift'] <= 1e-5
                        assert r['model_sha256'] == g['jobs'][j['fit_id']]['result']['checkpoint_sha256']
                        assert sha(out/'predictions_test.npz') == r['predictions_sha256']
                    j.update(status='completed', result=r, finished_at=rc['finished_at'])
                except Exception as exc:
                    j.update(status='failed', error=repr(exc), finished_at=time.time())
                handle = self.handles.pop((key,j['id']), None)
                if handle is not None: handle.poll()

    def phases(self):
        for g in self.state['groups'].values():
            jobs = list(g['jobs'].values()); out = Path(g['output'])
            if any(j['status']=='failed' for j in jobs): g['status']='needs_attention'; continue
            if g['phase']=='probes' and all(j['status']=='completed' for j in jobs if j['action']=='probe'):
                estimate = 1+sum(g['jobs'][j['pilot_id']]['result']['checkpoint_bytes']*g['retention_multiplier']*1.2/2**30
                                 for j in jobs if j['action']=='train')
                g['projected_artifact_gib'] = estimate
                if estimate >= g['expected_gib']:
                    g['status']='storage_review_required'; continue
                release = out/'formal_release.json'
                if not release.exists(): g['status']='awaiting_path_review'; continue
                r = read(release); assert r['complete'] and r['source_manifest_sha256']==g['source_manifest_sha256']
                for j in jobs:
                    if j['action']=='probe': assert sha(Path(j['output'])/'path_audit.json') == r['audit_sha256'][j['id']]
                g.update(phase='formal',status='running')
            if g['phase']=='formal' and all(j['status']=='completed' for j in jobs if j['action']=='train'):
                save(out/'fixed-checkpoint-lock.json', dict(complete=True,test_used_for_selection=False,created_at=time.time(),
                    selections={j['id']:dict(checkpoint_sha256=j['result']['checkpoint_sha256'],best_epoch=j['result']['best_epoch'],
                        dev_ap=j['result']['dev_ap']) for j in jobs if j['action']=='train'}))
                g.update(phase='evaluation',status='evaluating')
            if g['phase']=='evaluation' and all(j['status']=='completed' for j in jobs if j['action']=='evaluate'):
                g.update(status='complete',finished_at=time.time())

    def progress(self, j):
        p = Path(j['output'])/'progress.json'
        try: return read(p)
        except (OSError, ValueError): return {}

    def env(self, g, gpu):
        source = Path(g['source']); deps = self.project/'work/tracehr_tabular_20261004/deps'
        cache = self.project/'cache'/RUN; temp = self.project/'work'/RUN/'tmp'
        for p in (cache,temp): p.mkdir(parents=True, exist_ok=True)
        return dict(os.environ, TRACEHR_PROJECT=str(self.project), TRACEHR_DATABASE=g['database'],
            CEM_COMPLETION_STAGE=g['stage'], TRACEHR_TABULAR_DEPS=str(deps),
            PYTHONPATH=os.pathsep.join(map(str,[source,deps,*deps.glob('*.whl')])), PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', NUMPY_MADVISE_HUGEPAGE='0', TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),
            XDG_CACHE_HOME=str(cache/'runtime'), TORCH_HOME=str(cache/'torch'), MPLCONFIGDIR=str(cache/'matplotlib'),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            CUDA_VISIBLE_DEVICES=str(gpu))

    def launch(self, key, g, j, gpu, slot, evidence):
        source=Path(g['source']); out=Path(g['output']); assert not Path(j['output']).exists()
        cmd=[sys.executable,'-B',str(source/'work/tracehr_group_v1_validation_20261023/memory_entry.py'),
             str(source/g['entry']),'--project',str(self.project),'--database',g['database'],'--output',j['output'],
             '--variant',j['variant'],'--seed',str(j['seed']),'--endpoint',str(j['endpoint']),
             '--action',j['action'],'--expected-gib',str(g['expected_gib'])]
        if j['action']=='train': cmd+=['--pilot-record',g['jobs'][j['pilot_id']]['output']]
        if j['action']=='evaluate': cmd+=['--fit-output',g['jobs'][j['fit_id']]['output']]
        spec=out/(j['id']+'_attempt.json'); receipt=out/(j['id']+'_exit.json'); log=out/(j['id']+'.log')
        save(spec,dict(command=cmd,cwd=str(source),receipt=str(receipt),job_id=j['id'],source_manifest_sha256=g['source_manifest_sha256']))
        with log.open('x') as f:
            proc=subprocess.Popen([sys.executable,'-B',str(source/'scripts/five_seed_attempt.py'),'--spec',str(spec)],
                cwd=source,env=self.env(g,gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.handles[key,j['id']]=proc
        j.update(status='running',gpu=gpu,slot=slot,pid=proc.pid,started_at=time.time(),gpu_admission=evidence,
                 command=cmd,log=str(log),exit_receipt=str(receipt))

    def admit(self, rows):
        running=[j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status']=='running']
        if any(self.progress(j).get('status') in (None,'loading_verified_inputs','randomly_initializing_native','preparing_inputs') for j in running):
            self.state['admission_wait']='serialized initialization'; return
        mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
        by={r['index']:r for r in rows}; now=time.time()
        groups=sorted(self.state['groups'].items(),key=lambda item:(item[1]['stage_order'],item[1].get('last_launch',0)))
        for key,g in groups:
            if g['status'] not in ('preflight','running','evaluating'): continue
            action={'probes':'probe','formal':'train','evaluation':'evaluate'}[g['phase']]
            pending=[j for j in g['jobs'].values() if j['status']=='pending' and j['action']==action]
            if not pending: continue
            j=pending[0]
            needed=24576 if action=='probe' else max(8192,math.ceil(g['jobs'][j['pilot_id']]['result']['peak_reserved_mib']*1.3+2048))
            if mem['MemAvailable']/2**30 < g.get('min_available_ram_gib',40): continue
            if shutil.disk_usage(self.project).free/2**30 < 10+g['expected_gib']: continue
            for gpu in self.idle.candidates(rows,now):
                row=by[gpu]; active=[x for x in running if x['gpu']==gpu]
                if len(active)>=2 or sum(p['project_worker'] for p in row['processes'])>=2 or row['free_mib']<needed: continue
                # Probe resource measurement is isolated from other workers in this batch.
                if action=='probe' and active: continue
                if gpu not in self.locks:
                    lockpath=self.project/'work/tracehr_fullstream_20261003/locks'/f'gpu{gpu}.lock'
                    lockpath.parent.mkdir(parents=True,exist_ok=True); handle=lockpath.open('a+')
                    try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: handle.close(); continue
                    self.locks[gpu]=handle
                current=next(x for x in inventory(self.project) if x['index']==gpu)
                if current['free_mib']<needed or sum(p['project_worker'] for p in current['processes'])>=2: continue
                if gpu not in self.idle.preferred and gpu not in self.idle.leases and not current['fully_idle']: continue
                evidence=self.idle.acquire(gpu,current,now)
                slot=next(s for s in (0,1) if not any(x['slot']==s for x in active))
                self.launch(key,g,j,gpu,slot,evidence);g['last_launch']=time.time()
                self.state['admission_wait']=None;return
        self.state['admission_wait']='waiting for verified tasks or measured resources'

    def main(self):
        while True:
            rows=inventory(self.project);self.idle.observe(rows,time.time());self.state['gpu_inventory']=rows
            self.discover();self.reap();self.phases();self.admit(rows)
            groups=list(self.state['groups'].values())
            if len(groups)==self.state['expected_groups'] and all(g['status']=='complete' for g in groups):
                self.state.update(status='complete',finished_at=time.time());self.save();return
            self.state['status']='needs_attention' if any(g['status']=='needs_attention' for g in groups) else 'running'
            self.save();time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--host',choices=['mimic','eicu'],required=True)
    Queue(p.parse_args()).main()
