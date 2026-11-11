"""Authorized fresh P/M deletion fits and separately locked EMA tests."""
import argparse, gc, os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path[:0] = [str(deps), *[str(p) for p in deps.glob('*.whl')]]
import torch
from scripts.ema_pm_cross_common import *
from scripts import ema_pm_cross_runtime as runtime
from scripts.train_tracehr_unified import seed_all

def evaluate(a, out, data):
    # Reuse the audited evaluator: constructor audit with gradients enabled,
    # then strict checkpoint replay and a zero-update inference-only boundary.
    from scripts import train_mimiciii_ema_pm as evaluator
    evaluator.fresh_model = fresh_model
    evaluator.expected = expected
    evaluator.runtime = runtime
    started = time.time()
    evaluator.evaluate(a, out, data)
    result = read(out/'completion.json')
    result.update(evaluation_seconds_including_initialization_and_replay=time.time()-started)
    atomic(out/'completion.json', result)

def main():
    p = argparse.ArgumentParser()
    for n in ('project', 'output', 'pilot-record', 'fit-output'):
        p.add_argument('--'+n, type=Path, required=n in ('project', 'output'))
    p.add_argument('--database', choices=('mimic','eicu'), required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--endpoint', type=int, choices=[0,1], required=True)
    p.add_argument('--action', choices=['probe','train','evaluate'], required=True)
    p.add_argument('--expected-gib', type=float, required=True)
    a = p.parse_args(); a.project = a.project.resolve(strict=True)
    assert database() == a.database
    a.preparation = a.project/expected()['input_preparation']
    out = a.output.resolve(); assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free > (a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        value = Path(os.environ[k]).resolve(); assert value.is_relative_to(a.project)
        value.mkdir(parents=True, exist_ok=True)
    atomic(out/'storage.json', dict(project=str(a.project), output=str(out), preparation=str(a.preparation),
        expected_gib=a.expected_gib, code=str(ROOT), cache=os.environ['XDG_CACHE_HOME'], temp=os.environ['TMPDIR']))
    torch.set_num_threads(4); assert torch.cuda.is_available(); seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json', dict(status='loading_verified_inputs'))
    data = Data(a.project, a.database, a.preparation)
    if a.action == 'evaluate': evaluate(a, out, data); return
    atomic(out/'progress.json', dict(status='randomly_initializing_native'))
    model = fresh_model(data, a.preparation, a.seed, a.endpoint, out, a.variant)
    identity = dict(database=a.database, variant=a.variant, endpoint=a.endpoint, seed=a.seed, input_dim=data.input_dim,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'), input_manifest_sha256=sha(a.preparation/'completion.json'),
        protocol=protocol(a.variant,a.seed), ema_rule=EMA_RULE, epoch_budget=budget(a.seed,a.endpoint),
        initial_weights_sha256=parameter_hash(model), parameters=sum(p.numel() for p in model.parameters()),
        members=8, lr_multiplier=1., ablated_component=a.variant[-1])
    atomic(out/'run_manifest.json', identity); model.cuda(); gc.collect()
    from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
    atomic(out/'G0_equivalence.json', initial_equivalence(model,data))
    (runtime.probe if a.action=='probe' else runtime.train)(a,out,data,model,identity,identity['protocol'])

if __name__ == '__main__': main()
