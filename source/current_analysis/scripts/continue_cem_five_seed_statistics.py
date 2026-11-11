"""Automatically continue the authorized five-seed run after both host barriers.

This process only observes training and invokes the existing statistics collector.
It exposes no training command and never retries a statistical writer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_cem_five_seed as five

RUN = five.RUN
OUTPUT = ROOT / 'runs' / RUN / 'auto_statistics_v1'
GUARD = Path('D:/Codex/storage-tools/storage_guard.py')
EXPECTED = {'mimic': {'mimiciii': 12, 'mimic': 8}, 'eicu': {'eicu': 10}}
SOURCES = (
    'scripts/continue_cem_five_seed_statistics.py',
    'scripts/collect_cem_five_seed.py',
    'scripts/cem_five_seed_statistics.py',
    'scripts/collect_cem_replacements.py',
    'scripts/cem_replacement_statistics.py',
    'scripts/cem_icde72_statistics.py',
    'scripts/cem_icde72_significance.py',
    'scripts/launch_ema_cross.py',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def training_status(host):
    code = r'''
import hashlib,json,os,time
from pathlib import Path
d=PAYLOAD;p=Path(d['project']);folder=p/'runs'/d['run']/('host_'+d['host'])
state=json.loads((folder/'state.json').read_text());exitfile=folder/'start_exit.json'
receipt=json.loads(exitfile.read_text()) if exitfile.exists() else None
groups={}
for g in state['groups'].values():
 counts={action:{s:sum(j['action']==action and j['status']==s for j in g['jobs'].values()) for s in ('completed','running','pending','failed')} for action in ('train','evaluate','probe')}
 groups[g['database']]=dict(status=g['status'],**counts)
pid=state['pid'];proc=Path('/proc')/str(pid);alive=proc.exists();identity=None
if alive:
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 identity=dict(pid=pid,start_ticks=stat[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 alive=stat[0] not in ('Z','X')
 if alive:assert d['run'] in cmd and 'cem_five_seed_queue.py' in cmd, 'Scheduler PID identity changed'
print(json.dumps(dict(host=d['host'],status=state['status'],groups=groups,alive=alive,identity=identity,
 exit_path=str(exitfile),exit_sha256=hashlib.sha256(exitfile.read_bytes()).hexdigest() if receipt else None,
 returncode=receipt['returncode'] if receipt else None,observed_at=time.time())))
'''
    payload = dict(project=five.HOSTS[host][1], run=RUN, host=host)
    return five.ssh(host, code.replace('PAYLOAD', repr(payload)), 120)


def training_ready(status):
    expected = EXPECTED[status['host']]
    assert set(status['groups']) == set(expected), 'Unexpected training matrix'
    if status['returncode'] is not None:
        assert status['returncode'] == 0, 'Training host has a nonzero exit'
    for db, group in status['groups'].items():
        for action in ('train', 'evaluate'):
            assert sum(group[action].values()) == expected[db]
        assert not any(group[a]['failed'] for a in ('train', 'evaluate', 'probe')), 'A training/test/probe failed'
        assert group['status'] not in ('failed', 'blocked'), 'Training group failed'
    completed = all(g['status'] == 'complete' and g['train']['completed'] == g['evaluate']['completed'] == expected[db]
                    for db, g in status['groups'].items())
    if status['returncode'] is not None:
        assert completed, 'Host exited without all authorized outputs'
    # A small interval between child exit and the owning wrapper's exit receipt
    # is normal. Longer missing-exit situations are bounded in run().
    return completed and status['returncode'] == 0 and not status['alive']


def statistics_status(db):
    host = 'eicu' if db == 'eicu' else 'mimic'
    payload = dict(project=five.HOSTS[host][1], run=RUN, db=db, host=host, mode='status')
    return five.ssh(host, five.collector.REMOTE.replace('PAYLOAD', repr(payload)), 120)


def phase(status):
    if 'exit.json' in status:
        assert status['exit.json']['returncode'] == 0, 'Statistical writer failed; inspect without restarting'
        assert status.get('completion.json', {}).get('complete'), 'Successful exit lacks verified output'
        return 'complete'
    if status.get('attempt_exists'):
        return 'running'
    assert not status.get('completion.json') and not status.get('launch.json'), 'Inconsistent statistical attempt'
    return 'pending'


def next_starts(statuses):
    states = {db: phase(s) for db, s in statuses.items()}
    result = []
    if states['eicu'] == 'pending':
        result.append('eicu')
    if states['mimiciii'] == 'pending':
        assert states['mimic'] != 'running', 'An unexpected MIMIC-IV writer is already running'
        result.append('mimiciii')
    elif states['mimiciii'] == 'complete' and states['mimic'] == 'pending':
        result.append('mimic')
    return result


def invoke(mode, db=None):
    assert mode in ('start', 'results', 'aggregate')
    out = ROOT / 'runs' / RUN / ('paired_results_v1' if mode == 'aggregate' else db + '_results_v1')
    command = [sys.executable, '-B', str(GUARD), '--project', str(ROOT),
               '--run', 'five_seed_auto_' + (db or 'all') + '_' + mode,
               '--expected-gib', '0.5', '--output', str(out),
               '--cache', str(ROOT / 'cache' / RUN), '--temp', str(ROOT / 'work' / RUN / 'tmp_local'),
               '--', sys.executable, '-X', 'utf8', '-B', str(ROOT / 'scripts/collect_cem_five_seed.py'), '--mode', mode]
    if db is not None:
        command += ['--database', db]
    logfile = OUTPUT / ((db or 'all') + '_' + mode + '.log')
    # Exclusive log creation makes ambiguous launch outcomes stop for inspection.
    with logfile.open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, f'{mode} failed: {logfile}; no automatic retry'


def run(interval, max_hours):
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve() == OUTPUT.resolve()
    assert 30 <= interval <= 60 and 1 <= max_hours <= 72
    attempt = dict(pid=os.getpid(), parent_pid=os.getppid(), started_at=time.time(),
                   interval_seconds=interval, max_hours=max_hours, source_hashes=source_hashes(),
                   training_commands_exposed=False, repeated_statistical_launches_allowed=False)
    with (OUTPUT / 'attempt.json').open('x', encoding='utf-8') as stream:
        json.dump(attempt, stream, indent=2)
    rc = 1
    try:
        identities = {}; missing_since = {}; transient_errors = 0
        while True:
            assert time.time() - attempt['started_at'] < max_hours * 3600, 'Continuation deadline reached'
            assert source_hashes() == attempt['source_hashes'], 'Continuation or collector source changed'
            try:
                hosts = {host: training_status(host) for host in EXPECTED}
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                transient_errors += 1
                save(OUTPUT / 'progress.json', dict(stage='transport_wait',consecutive_errors=transient_errors,error=str(exc),time=time.time()))
                if transient_errors >= 3:
                    raise
                time.sleep(interval)
                continue
            transient_errors = 0
            readiness = {host: training_ready(status) for host, status in hosts.items()}
            for host, status in hosts.items():
                if status['alive']:
                    identity = status['identity']
                    assert identities.setdefault(host, identity) == identity, 'Scheduler process identity changed'
                if not status['alive'] and status['returncode'] is None:
                    first = missing_since.setdefault(host, time.time())
                    assert time.time() - first < 180, 'Scheduler exited without a recorded exit'
            if not all(readiness.values()):
                save(OUTPUT / 'progress.json', dict(stage='waiting_for_30_fits_and_30_tests',hosts=hosts,time=time.time()))
                time.sleep(interval)
                continue
            statuses = {db: statistics_status(db) for db in five.analysis.DATABASES}
            starts = next_starts(statuses)
            for db in starts:
                invoke('start', db)
                statuses[db] = statistics_status(db)
            for db, status in statuses.items():
                if phase(status) == 'complete':
                    local = ROOT / 'runs' / RUN / (db + '_results_v1/collection-receipt.json')
                    if not local.exists():
                        invoke('results', db)
            save(OUTPUT / 'progress.json', dict(stage='statistics',databases={db:phase(s) for db,s in statuses.items()},time=time.time()))
            if all(phase(s) == 'complete' for s in statuses.values()):
                invoke('aggregate')
                combined = ROOT / 'runs' / RUN / 'paired_results_v1'
                receipt = read(combined / 'completion.json')
                assert receipt['complete']
                for name, digest in receipt['files'].items():
                    assert sha(combined / name) == digest
                result = read(combined / 'verified-results.json')
                assert result['verified_new_fits'] == result['verified_fixed_tests'] == 30
                save(OUTPUT / 'completion.json', dict(complete=True,statistics_complete=True,paper_update_pending=True,
                     aggregate_sha256=sha(combined/'verified-results.json'),finished_at=time.time()))
                rc = 0
                return
            time.sleep(interval)
    except BaseException as exc:
        save(OUTPUT / 'failure.json', dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        save(OUTPUT / 'exit.json', dict(pid=os.getpid(),started_at=attempt['started_at'],finished_at=time.time(),returncode=rc))


def self_test():
    import copy
    pending = dict(attempt_exists=False)
    active = dict(attempt_exists=True)
    done = dict(attempt_exists=True, **{'exit.json':dict(returncode=0),'completion.json':dict(complete=True)})
    assert next_starts(dict(mimiciii=pending,mimic=pending,eicu=pending)) == ['eicu','mimiciii']
    assert next_starts(dict(mimiciii=active,mimic=pending,eicu=active)) == []
    assert next_starts(dict(mimiciii=done,mimic=pending,eicu=done)) == ['mimic']
    assert next_starts(dict(mimiciii=done,mimic=done,eicu=done)) == []
    bad = copy.deepcopy(done);bad['exit.json']['returncode'] = 1
    try: phase(bad)
    except AssertionError: pass
    else: raise AssertionError('Failed statistics accepted')
    good = dict(host='eicu',alive=False,returncode=0,groups={'eicu':dict(status='complete',
                train=dict(completed=10,running=0,pending=0,failed=0),
                evaluate=dict(completed=10,running=0,pending=0,failed=0),
                probe=dict(completed=5,running=0,pending=0,failed=0))})
    assert training_ready(good)
    waiting=copy.deepcopy(good);waiting.update(alive=True,returncode=None)
    waiting['groups']['eicu'].update(status='running',evaluate=dict(completed=9,running=1,pending=0,failed=0))
    assert not training_ready(waiting)
    for action in ('train','evaluate','probe'):
        bad=copy.deepcopy(good);bad['groups']['eicu'][action]['completed']-=1;bad['groups']['eicu'][action]['failed']+=1
        try: training_ready(bad)
        except AssertionError: pass
        else: raise AssertionError('Failed producer accepted')
    return dict(complete=True,checks=10,training_or_sampling_started=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('check','run','status'),required=True)
    parser.add_argument('--interval',type=int,default=60);parser.add_argument('--max-hours',type=int,default=48)
    args=parser.parse_args()
    if args.mode == 'check':
        result=self_test();save(Path(os.environ['PROJECT_RUN_DIR'])/'continuation-check.json',result);print(json.dumps(result))
    elif args.mode == 'status':
        result={name:read(OUTPUT/name) for name in ('attempt.json','progress.json','completion.json','failure.json','exit.json') if (OUTPUT/name).exists()}
        save(Path(os.environ['PROJECT_RUN_DIR'])/'continuation-status.json',result);print(json.dumps(result))
    else: run(args.interval,args.max_hours)
