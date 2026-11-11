"""Six prechecked, fresh fits on the explicitly authorized GPU pool."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tracehr_recovery_queue import Queue as BaseQueue, atomic, read, sha, process_identity
from scripts.mimiciii_fixed_common import RUN, VARIANT, SEEDS, expected, protocol, stopping


class Queue(BaseQueue):
    def __init__(self, a):
        import fcntl
        self.a, self.fcntl = a, fcntl
        self.project = a.project.resolve(strict=True)
        self.source = a.source.resolve(strict=True)
        self.out = a.output.resolve()
        self.cache = self.project / 'cache' / RUN
        self.temp = self.project / 'work' / RUN / 'tmp'
        self.preparation = self.project / expected()['preparation']
        for path in (self.source, self.out, self.cache, self.temp, self.preparation):
            assert path.resolve().is_relative_to(self.project) and path != self.project
        for path in (self.out, self.cache, self.temp):
            path.mkdir(parents=True, exist_ok=True)
        self.owner = (self.out / 'owner.lock').open('a+')
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (self.out / 'state.json').exists(), 'Existing queue must not be restarted'
        self.handles, self.locks = {}, {}
        for name, digest in read(self.source / 'source_manifest.json').items():
            assert sha(self.source / name) == digest, name
        verification = read(self.out.parent / 'verification.json')
        assert verification['complete'] and not verification['test_uploaded']
        assert verification['source_manifest_sha256'] == sha(self.source / 'source_manifest.json')
        self.state = dict(database='mimiciii', pid=os.getpid(), process_identity=process_identity(os.getpid()),
            status='preflight', phase='probes', jobs={}, started_at=time.time(),
            source_manifest_sha256=sha(self.source / 'source_manifest.json'), expected_gib=a.expected_gib,
            gpus=[0, 1, 4], slots_per_gpu=2, max_workers_per_gpu=2, formal_job_limit=6,
            from_scratch=True, test_loaded=False, hyperparameter_search=False, min_free_gpu_mib=16384,
            input_contract_sha256=sha(self.source / 'work' / RUN / 'expected.json'))
        for seed in SEEDS:
            for endpoint in (0, 1):
                name = f'{VARIANT}_t{endpoint}_s{seed}'
                for pilot in (True, False):
                    key = ('probe_' if pilot else '') + name
                    self.state['jobs'][key] = dict(id=key, variant=VARIANT, endpoint=endpoint, seed=seed,
                        pilot=pilot, pilot_id=None if pilot else 'probe_' + name,
                        action='probe' if pilot else 'train', status='pending',
                        output=str(self.out / ('pilots' if pilot else 'jobs') / key))
        self.save()

    def save(self):
        formal = [j for j in self.state['jobs'].values() if not j['pilot']]
        assert {(j['seed'], j['endpoint']) for j in formal} == {(s, e) for s in SEEDS for e in (0, 1)}
        self.state.update(updated_at=time.time(),
            counts={s:sum(j['status'] == s for j in formal) for s in ('pending', 'running', 'completed', 'failed')},
            probe_counts={s:sum(j['status'] == s for j in self.state['jobs'].values() if j['pilot'])
                          for s in ('pending', 'running', 'completed', 'failed')})
        atomic(self.out / 'state.json', self.state)

    def launch(self, j, gpu, slot):
        assert gpu in (0, 1, 4) and slot in (0, 1)
        out = Path(j['output'])
        assert not out.exists() and out.resolve().is_relative_to(self.out)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
            CUDA_VISIBLE_DEVICES=str(gpu), TRACEHR_PROJECT=str(self.project), TRACEHR_DATABASE='mimiciii',
            NUMPY_MADVISE_HUGEPAGE='0', TRACEHR_TABULAR_DEPS=str(self.project / 'work/tracehr_tabular_20261004/deps'),
            TEMP=str(self.temp), TMP=str(self.temp), TMPDIR=str(self.temp), XDG_CACHE_HOME=str(self.cache / 'runtime'),
            TORCH_HOME=str(self.cache / 'runtime/torch'), MPLCONFIGDIR=str(self.cache / 'runtime/matplotlib'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        bootstrap = self.source / 'work/tracehr_group_v1_validation_20261023/memory_entry.py'
        cmd = [sys.executable, '-B', str(bootstrap), str(self.source / 'scripts/train_mimiciii_fixed.py'),
            '--project', str(self.project), '--preparation', str(self.preparation), '--output', str(out),
            '--seed', str(j['seed']), '--endpoint', str(j['endpoint']), '--action', j['action'],
            '--expected-gib', str(self.a.expected_gib)]
        if not j['pilot']:
            cmd += ['--pilot-record', self.state['jobs'][j['pilot_id']]['output']]
        spec, receipt, log = [self.out / (j['id'] + suffix) for suffix in ('_attempt.json', '_exit.json', '.log')]
        atomic(spec, dict(command=cmd, cwd=str(self.source), receipt=str(receipt), job_id=j['id'],
                         source_manifest_sha256=self.state['source_manifest_sha256']))
        j.update(status='launching', gpu=gpu, slot=slot, command=cmd, log=str(log), exit_receipt=str(receipt))
        self.save()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-B', str(self.source / 'scripts/tracehr_recovery_attempt.py'),
                '--spec', str(spec)], cwd=self.source, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.handles[j['id']] = child
        j.update(status='running', pid=child.pid, process_identity=process_identity(child.pid), started_at=time.time())
        self.save()

    def validate(self, j):
        out = Path(j['output'])
        r = read(out / 'completion.json')
        identity = r.get('identity', r)
        assert r['complete'] and r['pilot'] == j['pilot'] and r['from_scratch']
        assert not r['test_loaded'] and r['frozen_learnable_parameters'] == 0
        for key, value in dict(database='mimiciii', variant=VARIANT, endpoint=j['endpoint'], seed=j['seed'],
            source_manifest_sha256=self.state['source_manifest_sha256'], parameters=13533083, members=8).items():
            assert identity[key] == value, key
        spec = protocol(VARIANT, 'mimiciii', j['seed'])
        assert identity['protocol'] == spec
        edge = read(out / 'connection_audit.json')
        assert edge['complete'] and not edge['learned_checkpoint_loaded'] and not edge['stop_gradient']
        assert edge['retained_CEM_tensors_exact'] and edge['reference_seed'] == j['seed']
        assert not edge['auxiliary_enabled'] and edge['main_residual_enabled']
        assert edge['main_loss_coefficient'] == .5 and edge['auxiliary_loss_coefficient'] == 0.
        equivalence = read(out / 'G0_equivalence.json')
        assert equivalence['complete'] and equivalence['disabled_fusion_max_logit_drift'] == 0
        if j['pilot']:
            assert r['selected_microbatch'] == 128 and r['evaluation_microbatch'] == 256
            assert r['effective_batch'] == 256 and r['optimizer_resume_max_parameter_drift'] <= 1e-6
            assert r['rate_check']['complete'] and len(r['rate_check']['schedules']) == 8
            assert r['longest_evidence_stress']['complete']
            assert r['longest_evidence_stress']['all_retained_tensors_have_main_gradients']
            assert all(g['main_connected'] == g['tensors'] for g in r['gradient_checks']['128']['loss_paths']['groups'].values())
        else:
            assert r['checkpoint_sha256'] == sha(out / 'best.pt')
            assert r['predictions_sha256'] == sha(out / 'predictions_dev.npz')
            assert r['checkpoint_replay']['complete'] and r['checkpoint_replay']['max_probability_drift'] <= 1e-5
            assert r['main_view'] == r['checkpoint_selection_view'] == r['stopping_view'] == 'z0'
            assert not r['auxiliary_enabled'] and r['auxiliary_metrics_at_selected'] is None
            pilot = read(Path(self.state['jobs'][j['pilot_id']]['output']) / 'completion.json')
            assert pilot['identity'] == {k:r[k] for k in pilot['identity']}
            history = read(out / 'history.json')
            assert len(history) == r['epochs'] and r['dev_ap'] == max(h['dev_ap'] for h in history)
            for i in range(1, len(history)):
                assert not stopping(history[:i], spec)['stop'], ('past first stopping boundary', i)
            assert stopping(history, spec) == r['stopping_status']
            assert r['stopping_status']['stop'] or r['epochs'] == 256
            for epoch in (1, 5, 8, 10, 12, 15, 18, 20):
                record = read(out / f'epoch_{epoch:03d}_diagnostics.json')
                assert record['checkpoint_sha256'] == sha(out / f'epoch_{epoch:03d}_model.pt')
        return r

    def phase(self):
        jobs = list(self.state['jobs'].values())
        if any(j['status'] == 'failed' for j in jobs):
            self.state['status'] = 'needs_attention'
            return
        if self.state['phase'] == 'probes' and all(j['status'] == 'completed' for j in jobs if j['pilot']):
            probes = [j['result'] for j in jobs if j['pilot']]
            input_gib = sum(v['bytes'] for v in expected()['inputs'].values()) / 2**30
            estimate = input_gib + .75 + sum(p['checkpoint_bytes'] * 12 * 1.15 for p in probes) / 2**30
            assert estimate < self.a.expected_gib, ('storage estimate exceeds declaration', estimate)
            peak = max(p['peak_reserved_mib'] for p in probes)
            self.state.update(phase='formal', status='running', projected_artifact_gib=estimate,
                min_free_gpu_mib=max(8192, math.ceil(peak * 1.5 + 2048)),
                formal_released_at=time.time(), all_six_same_seed_probes_passed=True)
        if all(j['status'] == 'completed' for j in jobs if not j['pilot']):
            self.state.update(status='complete', finished_at=time.time())

    def admit(self):
        if self.state['status'] in ('complete', 'needs_attention'):
            return
        running = [j for j in self.state['jobs'].values() if j['status'] == 'running']
        # Serialize initialization and the bounded worst-input stress check.
        if any(self.progress(j).get('status') in (None, 'loading_verified_inputs', 'randomly_initializing_native') for j in running):
            self.state['admission_wait'] = 'serialized initialization'
            return
        probes = self.state['phase'] == 'probes'
        pending = [j for j in self.state['jobs'].values() if j['status'] == 'pending' and j['pilot'] == probes]
        if not pending:
            return
        slots = [(g, s) for s in range(1 if probes else 2) for g in (0, 1, 4)]
        for gpu, slot in slots:
            if any(j['gpu'] == gpu and j['slot'] == slot for j in running):
                continue
            if self.resources(gpu):
                self.launch(pending[0], gpu, slot)
                return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('project', 'source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-gib', type=float, required=True)
    args = p.parse_args()
    args.database = 'mimiciii'
    q = Queue(args)
    q.main()
    if q.state['status'] != 'complete':
        raise SystemExit(2)
