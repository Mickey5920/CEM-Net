"""Authorized eICU-only early start using the frozen Medformer analysis.

The existing scheduled monitor keeps ownership of MIMIC and the six-comparison
aggregate. This entry changes only the cross-host start prerequisite; a remote
exclusive attempt and the original writer lock still prevent duplicate work.
"""
import argparse
import ast
import base64
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HOOK = ROOT / 'work/cem_sequence_comparators_20261102/monitor_quiet_v1/sitecustomize.py'
import hashlib
assert hashlib.sha256(HOOK.read_bytes()).hexdigest() == '8399be34c6f0cd563f8e07bed0d1a8c483a7cb946cd0382b7f9d51be9dea5d49'
runpy.run_path(str(HOOK))
from scripts import collect_cem_medformer as collection

RUN = collection.RUN
OUT = ROOT / 'runs' / RUN / 'eicu_results_v1'
JOURNAL = ROOT / 'runs' / RUN / 'auto_statistics_quiet_v1'
sha = collection.prior.sha
read = collection.prior.read
save = collection.prior.save
OLD_TEST = "all(r['ready'] for r in d['global_readiness'].values()) and d['global_readiness'][d['host']]['statistics_slot_ready']"
NEW_TEST = "d['db'] == 'eicu' and d['host'] == 'eicu' and d['global_readiness']['eicu']['ready'] and d['global_readiness']['eicu']['statistics_slot_ready']"


def frozen_hashes():
    expected = read(JOURNAL / 'attempt.json')['source_hashes']
    actual = {name: sha(ROOT / name) for name in expected}
    assert actual == expected, 'Frozen Medformer source changed'
    assert collection.analysis.B == 20000 and collection.analysis.FAMILY == 6
    assert collection.analysis.SEEDS == (48, 49, 50, 51, 52)
    return actual


def remote_entry():
    original = ast.parse(collection.collector.REMOTE)
    changed = ast.parse(collection.collector.REMOTE)
    expected = ast.dump(ast.parse(OLD_TEST, mode='eval').body)
    matches = [node for node in ast.walk(changed)
               if isinstance(node, ast.Assert) and ast.dump(node.test) == expected]
    assert len(matches) == 1, 'Start prerequisite changed unexpectedly'
    matches[0].test = ast.parse(NEW_TEST, mode='eval').body
    entry = ast.unparse(changed)
    matches[0].test = ast.parse(OLD_TEST, mode='eval').body
    assert ast.dump(changed) == ast.dump(original), 'Non-gate remote code changed'
    return entry


def self_test():
    frozen_hashes()
    remote_entry()
    gate = compile(NEW_TEST, '<eicu-only-start>', 'eval')
    def accepted(db, host, eicu_ready, slot, mimic_ready=False):
        d = dict(db=db, host=host, global_readiness=dict(
            eicu=dict(ready=eicu_ready, statistics_slot_ready=slot),
            mimic=dict(ready=mimic_ready, statistics_slot_ready=False)))
        return eval(gate, {}, dict(d=d))
    assert accepted('eicu', 'eicu', True, True)
    assert not accepted('eicu', 'eicu', False, True)
    assert not accepted('eicu', 'eicu', True, False)
    assert not accepted('mimiciii', 'mimic', True, True)
    assert not accepted('mimic', 'mimic', True, True)
    assert not accepted('eicu', 'mimic', True, True)
    return dict(complete=True, checks=6, only_cross_host_start_gate_changed=True,
                unchanged_analysis=True, family_count=6, replicates=20000,
                no_training_entry=True, no_console=True, script_sha256=sha(__file__))


def start():
    hashes = frozen_hashes()
    assert read(OUT / 'early-start-check.json')['script_sha256'] == sha(__file__)
    assert read(OUT / 'check-receipt.json')['complete']
    readiness = {host: collection.host_readiness(host) for host in ('mimic', 'eicu')}
    eicu = readiness['eicu']
    assert eicu['ready'] and eicu['returncode'] == 0 and not eicu['alive']
    assert eicu['statistics_slot_ready'], 'Existing eICU CNN statistics must finish first'
    group = eicu['groups']['eicu']
    assert group['train']['completed'] == group['evaluate']['completed'] == 10
    assert all(not group[action]['failed'] for action in ('train', 'evaluate', 'probe'))
    assert not readiness['mimic']['ready'], 'Normal monitor already owns the all-ready phase'
    d = dict(project=collection.HOSTS['eicu'][1], run=RUN, db='eicu', host='eicu', mode='status')
    existing = collection.ssh('eicu', collection.collector.REMOTE.replace('PAYLOAD', repr(d)), 120)
    assert not existing.get('attempt_exists') and not existing.get('completion.json'), 'Existing attempt; status/results only'
    assert not existing.get('launch.json') and not existing.get('exit.json')
    raw, manifest = collection.bundle()
    d.update(mode='start', files={n: base64.b64encode(b).decode() for n, b in raw.items()},
             manifest=manifest, global_readiness=readiness)
    policy = dict(authorized_at=time.time(), database='eicu',
                  scope='Start completed eICU separately; MIMIC and final six-comparison Holm remain with the original monitor',
                  original_readiness=readiness, frozen_source_hashes=hashes,
                  entry_sha256=sha(__file__), family_count=6, replicates=20000,
                  training_restarts=0, statistics_restarts=0)
    with (OUT / 'early-start-authorization.json').open('x', encoding='utf-8') as stream:
        json.dump(policy, stream, indent=2)
    result = collection.ssh('eicu', remote_entry().replace('PAYLOAD', repr(d)), 120)
    assert result['started'] and not result['training_launched']
    save(OUT / 'start-receipt.json', result)
    assert frozen_hashes() == hashes
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'start', 'status', 'results'), required=True)
    args = parser.parse_args()
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve() == OUT.resolve()
    if args.mode == 'check':
        assert not (OUT / 'early-start-check.json').exists()
        result = self_test()
        save(OUT / 'early-start-check.json', result)
        print(json.dumps(result))
    elif args.mode == 'start':
        print(json.dumps(start()))
    else:
        frozen_hashes()
        sys.argv = [__file__, '--database', 'eicu', '--mode', args.mode]
        collection.collector.main()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        if OUT.exists():
            save(OUT / 'early-entry-failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
