"""Authorized local monitor handoff: no consoles, ten-minute polling.

Frozen training/statistics sources and running remote tasks remain unchanged.
"""
import argparse
import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PARENT = ROOT/'work/cem_sequence_comparators_20261102/monitor_recovery_v2/resume.py'
PARENT_SHA = '5e22e1ab15715845f5bea2fb31b726821537619d990cadfa5beb673861c002ec'
PLANS = {
    'cnn': ('cem_sequence_comparators_20261102', 'auto_cnn_scheduled_v1',
            'auto_cnn_quiet_v1', 'continue_cem_cnn_delivery', 2),
    'medformer': ('cem_medformer_20261103', 'auto_statistics_scheduled_v1',
                  'auto_statistics_quiet_v1', 'continue_cem_medformer', 3),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)


assert sha(PARENT) == PARENT_SHA
spec = importlib.util.spec_from_file_location('prior_scheduled_monitor', PARENT)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def no_consoles():
    # The same hook is inherited by collector/release Python children. It is not
    # installed globally or transmitted to remote training processes.
    os.environ['PYTHONPATH'] = str(HERE) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    spec = importlib.util.spec_from_file_location('cem_console_policy', HERE/'sitecustomize.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    assert subprocess.Popen._cem_console_free


def adjusted_run(module, sleep_count):
    tree = ast.parse(inspect.getsource(module.run))
    original = ast.dump(tree, include_attributes=False)
    changed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'time' and node.func.attr == 'sleep' and len(node.args) == 1:
                assert isinstance(node.args[0], ast.Constant) and node.args[0].value == 60
                changed.append(node.args[0])
        if isinstance(node, ast.keyword) and node.arg == 'interval_seconds':
            assert isinstance(node.value, ast.Constant) and node.value.value == 60
            changed.append(node.value)
    assert len(changed) == sleep_count + 1
    for node in changed:
        node.value = 600
    code = compile(tree, str(Path(module.__file__)), 'exec')
    # Reversing precisely the authorized polling constants must restore the AST.
    for node in changed:
        node.value = 60
    assert ast.dump(tree, include_attributes=False) == original
    namespace = dict(module.__dict__)
    exec(code, namespace)
    return namespace['run'], dict(poll_seconds=600,prior_poll_seconds=60,
        changed_poll_sites=sleep_count,attempt_metadata_updated=True,
        all_other_control_flow_unchanged=True)


def prepare(pipeline):
    run_id, old_name, new_name, module_name, sleeps = PLANS[pipeline]
    old = ROOT/'runs'/run_id/old_name
    out = ROOT/'runs'/run_id/new_name
    assert old.resolve().is_relative_to(ROOT) and out.resolve().is_relative_to(ROOT)
    attempt = read(old/'attempt.json')
    module = importlib.import_module('scripts.'+module_name)
    assert module.hashes() == attempt['source_hashes']
    for name, digest in attempt['source_hashes'].items():
        source = ROOT/name if name.startswith('scripts/') else ROOT/'scripts'/name
        assert sha(source) == digest
    module.OUTPUT = out
    run, proof = adjusted_run(module, sleeps)
    return old, out, attempt, run, proof


def check():
    no_consoles()
    checks = {}
    for pipeline in PLANS:
        old, out, attempt, _, proof = prepare(pipeline)
        assert not (out/'attempt.json').exists()
        checks[pipeline] = proof
    console = "import ctypes,json; k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=ctypes.c_void_p; print(json.dumps({'console':bool(k.GetConsoleWindow())}))"
    python = str(Path(sys.executable).with_name('python.exe'))
    direct = subprocess.run([python, '-B', '-c', console], capture_output=True, text=True, check=True)
    assert json.loads(direct.stdout)['console'] is False
    nested = "import subprocess,sys,json; assert subprocess.Popen._cem_console_free; r=subprocess.run([sys.executable,'-B','-c'," + repr(console) + "],capture_output=True,text=True,check=True); print(r.stdout)"
    inherited = subprocess.run([python, '-B', '-c', nested], capture_output=True, text=True, check=True)
    assert json.loads(inherited.stdout)['console'] is False
    from scripts.launch_ema_cross import ssh
    for host in ('mimic', 'eicu'):
        assert ssh(host,"import json; print(json.dumps({'read_only':True}))",60)['read_only']
    report = dict(complete=True,checks=checks,direct_child_console=False,
        nested_child_console=False,hidden_SSH_both_hosts=True,
        frozen_sources_match=True,training_restarts=0,statistics_restarts=0,
        wrapper_sha256=sha(__file__),hook_sha256=sha(HERE/'sitecustomize.py'))
    save_new(Path(os.environ['PROJECT_RUN_DIR'])/'check.json', report)
    print(json.dumps(report))


def run(pipeline):
    no_consoles()
    old, out, previous, runner, policy = prepare(pipeline)
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve() == out.resolve()
    handoff = read(HERE/'handoff.json')
    observed = handoff[pipeline]
    assert observed['pid'] == previous['pid'] and observed['guard_pid'] == previous['parent_pid']
    assert observed['old_attempt_sha256'] == sha(old/'attempt.json')
    assert observed['stopped_local_monitor_only'] and not observed['children_at_stop']
    assert not any((old/n).exists() for n in ('failure.json','completion.json'))
    journals = [ROOT/'runs'/PLANS[pipeline][0]/n for n in prior.PLANS[pipeline][1]] + [old]
    with ExitStack() as locks:
        for journal in journals:
            locks.enter_context(prior.lock_journal(journal))
        for journal in journals:
            identity = read(journal/'attempt.json')
            assert not prior.alive(identity['pid']) and not prior.alive(identity['parent_pid'])
        save_new(out/'runtime_policy.json', dict(policy,pipeline=pipeline,
            wrapper_sha256=sha(__file__),hook_sha256=sha(HERE/'sitecustomize.py'),
            handoff_sha256=sha(HERE/'handoff.json'),frozen_source_hashes=previous['source_hashes'],
            creationflags='CREATE_NO_WINDOW',startupinfo='SW_HIDE',
            inherited_by_local_python_children=True,old_journal_locks_held=True,
            training_restarts=0,statistics_restarts=0))
        runner()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('check','run'),required=True)
    parser.add_argument('--pipeline',choices=tuple(PLANS))
    args=parser.parse_args()
    if args.mode == 'check':
        check()
    else:
        assert args.pipeline
        output=ROOT/'runs'/PLANS[args.pipeline][0]/PLANS[args.pipeline][2]
        assert Path(os.environ['PROJECT_RUN_DIR']).resolve() == output.resolve()
        rc=1
        with (output/'entry.log').open('x',encoding='utf8',buffering=1) as log:
            with redirect_stdout(log),redirect_stderr(log):
                try:
                    run(args.pipeline)
                    rc=0
                except BaseException:
                    traceback.print_exc()
                    raise
                finally:
                    save_new(output/'launcher_exit.json',dict(pid=os.getpid(),returncode=rc,finished_at=time.time()))
