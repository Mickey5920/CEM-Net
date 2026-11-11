"""Use the already validated optimizer, loss, EMA, batches and fixed training loop."""
import argparse
import gc
import os
import shutil
from pathlib import Path
import torch
from scripts import train_cem_matched_controls as runtime
from scripts.cem_targeted_common import Data, construct, configuration, protocol, budget, VARIANTS, SEEDS, ROOT
from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.train_tracehr_unified import seed_all
from scripts.mimiciii_ema_runtime import EMA_RULE
from models.tracehr_unified_next import parameter_hash


def main():
    p = argparse.ArgumentParser()
    for name in ('project', 'output', 'pilot-record', 'fit-output'):
        p.add_argument('--'+name, type=Path, required=name in ('project', 'output'))
    p.add_argument('--database', required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--endpoint', type=int, choices=(0, 1), required=True)
    p.add_argument('--action', choices=('probe', 'train', 'evaluate'), required=True)
    p.add_argument('--expected-gib', type=float, required=True)
    a = p.parse_args()
    project, out = a.project.resolve(strict=True), a.output.resolve()
    c = configuration()
    assert c['database'] == a.database and out.is_relative_to(project) and not out.exists()
    assert shutil.disk_usage(project).free > (a.expected_gib+10)*2**30
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        path = Path(os.environ[name]).resolve()
        assert path.is_relative_to(project)
        path.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json', dict(status='loading_verified_inputs'))
    data = Data(project, a.variant, test=a.action == 'evaluate')
    atomic(out/'progress.json', dict(status='randomly_initializing_native'))
    model = construct(data, a.seed, a.endpoint, a.variant)
    spec = protocol(a.seed, a.endpoint)
    overlay_hash = sha(project/c['overlay']/'completion.json') if a.variant == 'observation_weighted' else None
    identity = dict(database=a.database, variant=a.variant, endpoint=a.endpoint, seed=a.seed,
                    input_dim=data.input_dim, parameters=sum(p.numel() for p in model.parameters()), members=8,
                    protocol=spec, epoch_budget=budget(a.database, a.seed, a.endpoint), ema_rule=EMA_RULE,
                    source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                    input_manifest_sha256=sha(data.preparation/'completion.json'),
                    window_sha256=overlay_hash, aggregation_overlay_sha256=overlay_hash,
                    initial_weights_sha256=parameter_hash(model), initialization='fresh; no learned weights loaded')
    atomic(out/'run_manifest.json', identity)
    atomic(out/'model_details.json', model.base.info)
    model.cuda()
    gc.collect()
    if a.action == 'probe':
        runtime.probe(a, out, data, model, identity, spec)
    elif a.action == 'train':
        runtime.train(a, out, data, model, identity, spec)
    else:
        runtime.evaluate(a, out, data, model, identity)


if __name__ == '__main__':
    main()
