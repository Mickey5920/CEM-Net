"""Adopt existing fits; lock and evaluate each complete ten-fit method separately."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

RUN = 'cem_sequence_comparators_20261102'
CNN = ('TCN', 'ModernTCN', 'LITEMV')
STAGES = {'cnn': CNN, 'ft': ('FTTransformer',)}
SEEDS = (48, 49, 50, 51, 52)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def method_ids(variant):
    return {f'{variant}_t{e}_s{s}' for s in SEEDS for e in (0, 1)}


def selections(jobs):
    return {j['id']: {k: j['result'][k] for k in ('checkpoint_sha256', 'best_epoch', 'dev_ap')}
            for j in jobs}


def evaluation_source(path):
    """Change only the completed-scope lock, retaining the frozen evaluator body."""
    original = Path(path).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(original).body
                if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    before = ast.get_source_segment(original, node)
    assert before.count("fit.parent.parent/'fixed-checkpoint-lock.json'") == 1
    assert before.count("expected()['formal_fits']") == 1
    after = before.replace("fit.parent.parent/'fixed-checkpoint-lock.json'",
                           "fit.parent.parent/'partition_locks'/f'{a.variant}.json'")
    after = after.replace("expected()['formal_fits']", '10')
    a, b = ast.parse(before).body[0], ast.parse(after).body[0]
    assert ast.dump(ast.Module(body=a.body[5:], type_ignores=[]), include_attributes=False) == \
           ast.dump(ast.Module(body=b.body[5:], type_ignores=[]), include_attributes=False)
    return after


def evaluate(project, database):
    source = project / 'work' / RUN / database / 'v3'
    sys.path.insert(0, str(source))
    from scripts import train_cem_sequence_comparator as frozen
    assert '--action' in sys.argv and sys.argv[sys.argv.index('--action') + 1] == 'evaluate'
    host = project / 'runs' / RUN / ('host_eicu_v3' if database == 'eicu' else 'host_mimic_v3')
    proof = read(host / 'partition_v1/handoff.json')
    assert proof['complete'] and proof['workers_stopped'] == 0
    assert sha(__file__) == proof['runtime_sha256']
    manifest = read(source / 'source_manifest.json')
    for name in ('scripts/train_cem_sequence_comparator.py', 'scripts/cem_sequence_comparator_common.py',
                 'models/cem_sequence_comparators.py'):
        assert sha(source / name) == manifest[name]
    exec(compile(evaluation_source(source / 'scripts/train_cem_sequence_comparator.py'),
                 str(__file__) + ':frozen_evaluator', 'exec'), frozen.__dict__)
    run = frozen.evaluate

    def checked(a, out, data):
        lockfile = a.fit_output.parent.parent / 'partition_locks' / (a.variant + '.json')
        lock = read(lockfile)
        assert lock['variant'] == a.variant and set(lock['selections']) == method_ids(a.variant)
        assert lock['source_manifest_sha256'] == sha(source / 'source_manifest.json')
        assert lock['handoff_sha256'] == sha(host / 'partition_v1/handoff.json')
        assert not lock['test_used_for_selection']
        digest = sha(lockfile)
        run(a, out, data)
        assert sha(lockfile) == digest
        result = read(out / 'completion.json')
        result.update(evaluation_runtime_sha256=sha(__file__), checkpoint_lock_sha256=digest,
                      checkpoint_lock_path=str(lockfile), partition_handoff_sha256=sha(host / 'partition_v1/handoff.json'))
        save(out / 'completion.json', result)

    frozen.evaluate = checked
    frozen.main()


def queue_class(project, host):
    import fcntl
    import math
    import shutil
    source = project / 'work' / RUN / ('eicu' if host == 'eicu' else 'mimiciii') / 'v3'
    sys.path.insert(0, str(source))
    from scripts.cem_sequence_comparator_queue import Queue
    from scripts.cem_evidence_host_queue import IdlePolicy, inventory
    from scripts.tracehr_recovery_queue import same_process

    class Policy(IdlePolicy):
        def candidates(self, rows, now):
            return [g for g in super().candidates(rows, now) if host != 'eicu' or g != 3]

    class PartitionQueue(Queue):
        def __init__(self, a):
            self.a = a
            self.project = project
            self.out = project / 'runs' / RUN / ('host_' + host + '_v3')
            self.change = self.out / 'partition_v1'
            proof = read(self.change / 'handoff.json')
            assert proof['complete'] and proof['workers_stopped'] == 0
            assert sha(__file__) == proof['runtime_sha256']
            assert sha(self.change / 'predecessor-state.json') == proof['state_sha256']
            assert read(proof['predecessor_exit_path'])['returncode'] == -15
            assert sha(proof['predecessor_exit_path']) == proof['predecessor_exit_sha256']
            self.owner = (self.out / 'owner.lock').open('a+')
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = read(self.change / 'predecessor-state.json')
            for path, digest in proof['protected'].items():
                assert sha(path) == digest, path
            self.handles, self.locks = {}, {}
            cards = set()
            for g in self.state['groups'].values():
                for j in g['jobs'].values():
                    if j['status'] != 'running':
                        continue
                    f = Path(j['exit_receipt'])
                    owner = read(f.with_name(f.stem + '_owner.json'))
                    assert owner == proof['workers'][g['database'] + ':' + j['id']]['owner']
                    assert (f.exists() and read(f)['returncode'] == 0) or same_process(owner['child_pid'], owner['child_identity'])
                    cards.add(j['gpu'])
            for gpu in cards:
                f = project / 'work/tracehr_fullstream_20261003/locks' / f'gpu{gpu}.lock'
                handle = f.open('a+')
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locks[gpu] = handle
            self.idle = Policy([2, 4, 6] if host == 'eicu' else [0, 1, 2, 3])
            self.idle.since = {int(k): v for k, v in self.state.get('idle_observations', {}).items()}
            self.idle.leases = {int(k): v for k, v in self.state.get('idle_leases', {}).items() if host != 'eicu' or int(k) != 3}
            self.idle.last = self.state['updated_at']
            self.state.update(pid=os.getpid(), active_scheduler_receipt=str(self.change / 'exit.json'),
                              partition_handoff_sha256=sha(self.change / 'handoff.json'),
                              partition_runtime_sha256=sha(__file__), scheduler_adopted_at=time.time(),
                              predecessor_scheduler_pid=proof['predecessor_pid'],
                              evaluation_gate='all ten fixed fits of each method; CNN tests prioritized',
                              statistics_gate='immutable completed cnn/ft partitions; host may remain active')
            self.save()

        def phases(self):
            for g in self.state['groups'].values():
                jobs = list(g['jobs'].values())
                out = Path(g['output'])
                if any(j['status'] == 'failed' for j in jobs):
                    g['status'] = 'needs_attention'
                    continue
                g.update(phase='partitioned', status='running')
                for variant in (*CNN, 'FTTransformer'):
                    fits = [j for j in jobs if j['action'] == 'train' and j['variant'] == variant]
                    assert {j['id'] for j in fits} == method_ids(variant)
                    if not all(j['status'] == 'completed' for j in fits):
                        continue
                    selected = selections(fits)
                    path = out / 'partition_locks' / (variant + '.json')
                    if path.exists():
                        assert read(path)['selections'] == selected
                    else:
                        for j in fits:
                            assert read(j['exit_receipt'])['returncode'] == 0
                            assert sha(Path(j['output']) / 'best.pt') == selected[j['id']]['checkpoint_sha256']
                        save(path, dict(complete=True, variant=variant, test_used_for_selection=False,
                                        source_manifest_sha256=g['source_manifest_sha256'],
                                        handoff_sha256=self.state['partition_handoff_sha256'],
                                        selections=selected, created_at=time.time()))
                fits = [j for j in jobs if j['action'] == 'train']
                if all(j['status'] == 'completed' for j in fits):
                    path = out / 'fixed-checkpoint-lock.json'
                    if path.exists():
                        assert read(path)['selections'] == selections(fits)
                    else:
                        save(path, dict(complete=True, test_used_for_selection=False, created_at=time.time(),
                                        selections=selections(fits)))
                for stage, variants in STAGES.items():
                    subset = {j['id']: j for j in jobs if j['variant'] in variants}
                    path = out / 'partition_receipts' / (stage + '.json')
                    if path.exists() or not all(j['status'] == 'completed' for j in subset.values()):
                        continue
                    for j in subset.values():
                        assert read(j['exit_receipt'])['returncode'] == 0
                    snapshot = copy.deepcopy(g)
                    snapshot.update(jobs=copy.deepcopy(subset), status='partition_complete', partition=stage)
                    save(path, dict(complete=True, partition=stage, variants=list(variants),
                                    formal_fits=10 * len(variants), fixed_tests=10 * len(variants),
                                    host_queue_exit_required=False, host_queue_pid=self.state['pid'],
                                    handoff_path=str(self.change / 'handoff.json'),
                                    handoff_sha256=self.state['partition_handoff_sha256'],
                                    runtime_sha256=sha(__file__), group=snapshot, created_at=time.time()))
                if all(j['status'] == 'completed' for j in jobs):
                    g.update(status='complete', phase='complete', finished_at=time.time())

        def launch(self, key, g, j, gpu, slot, evidence):
            if j['action'] == 'evaluate':
                g = dict(g, entries=dict(g['entries'], tabular=str(Path(__file__).resolve())))
            super().launch(key, g, j, gpu, slot, evidence)

        def admit(self, rows):
            running = [j for g in self.state['groups'].values() for j in g['jobs'].values() if j['status'] == 'running']
            if any(self.progress(j).get('status') in (None, 'loading_verified_inputs', 'randomly_initializing_native', 'preparing_inputs') for j in running):
                self.state['admission_wait'] = 'serialized initialization'
                return
            mem = {k: int(v.split()[0]) * 1024 for k, v in
                   (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
            candidates = []
            for key, g in self.state['groups'].items():
                if g['status'] != 'running':
                    continue
                for order, j in enumerate(g['jobs'].values()):
                    if j['status'] != 'pending':
                        continue
                    if j['action'] == 'evaluate':
                        if not (Path(g['output']) / 'partition_locks' / (j['variant'] + '.json')).exists():
                            continue
                        priority = 0 if j['variant'] in CNN else 3
                    elif j['action'] == 'train':
                        priority = 1 if j['variant'] in CNN else 2
                    else:
                        raise AssertionError('Preflights must already be complete')
                    candidates.append((priority, g.get('last_launch', 0), order, key, g, j))
            by = {r['index']: r for r in rows}
            now = time.time()
            for _, _, _, key, g, j in sorted(candidates, key=lambda x: x[:4]):
                if shutil.disk_usage(project).free / 2**30 < 10 + g['expected_gib']:
                    continue
                if mem['MemAvailable'] / 2**30 < g.get('min_available_ram_gib', 24):
                    continue
                assert j['device'] == 'gpu'
                needed = max(8192, math.ceil(g['jobs'][j['pilot_id']]['result']['peak_reserved_mib'] * 1.3 + 2048))
                for gpu in self.idle.candidates(rows, now):
                    row = by[gpu]
                    active = [x for x in running if x['gpu'] == gpu]
                    if len(active) >= 2 or sum(p['project_worker'] for p in row['processes']) >= 2 or row['free_mib'] < needed:
                        continue
                    if gpu not in self.locks:
                        f = project / 'work/tracehr_fullstream_20261003/locks' / f'gpu{gpu}.lock'
                        handle = f.open('a+')
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            handle.close()
                            continue
                        self.locks[gpu] = handle
                    current = next(x for x in inventory(project) if x['index'] == gpu)
                    if current['free_mib'] < needed or sum(p['project_worker'] for p in current['processes']) >= 2:
                        continue
                    if gpu not in self.idle.preferred and gpu not in self.idle.leases and not current['fully_idle']:
                        continue
                    evidence = self.idle.acquire(gpu, current, now)
                    evidence.update(required_free_mib=needed, probe_requirement_basis=g.get('probe_requirement_basis'))
                    slot = next(s for s in (0, 1) if not any(x['slot'] == s for x in active))
                    self.launch(key, g, j, gpu, slot, evidence)
                    g['last_launch'] = time.time()
                    self.state['admission_wait'] = None
                    return
            self.state['admission_wait'] = 'waiting for verified tasks or measured resources'

    return PartitionQueue


def main():
    if '--action' in sys.argv:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument('--project', type=Path, required=True)
        p.add_argument('--database', required=True)
        a, _ = p.parse_known_args()
        evaluate(a.project.resolve(strict=True), a.database)
        return
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--host', choices=('mimic', 'eicu'), required=True)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    project = a.project.resolve(strict=True)
    if a.self_test:
        source = project / 'work' / RUN / ('eicu' if a.host == 'eicu' else 'mimiciii') / 'v3'
        evaluation_source(source / 'scripts/train_cem_sequence_comparator.py')
        assert len(method_ids('TCN')) == 10 and len(set().union(*(method_ids(x) for x in CNN))) == 30
        queue_class(project, a.host)
        print(json.dumps(dict(complete=True, frozen_evaluation_computation_unchanged=True, training_started=False)))
    else:
        queue_class(project, a.host)(a).main()


if __name__ == '__main__':
    main()
