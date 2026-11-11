"""Verify source/inputs, prepare the compact overlay and prove reference parity."""
import argparse
import gc
import json
from pathlib import Path
import torch
from scripts.cem_matched_common import configuration, verify_original_constructor, Data, construct, ROOT
from scripts.cem_matched_windows import prepare, checks
from scripts.tracehr_multiview_common import read, sha, atomic


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    project = a.project.resolve(strict=True); output = a.output.resolve()
    assert output.is_relative_to(project) and not output.exists()
    output.mkdir(parents=True); torch.set_num_threads(4)
    c = configuration()
    for name, digest in read(ROOT/'source_manifest.json').items():
        assert sha(ROOT/name) == digest, name
    prep = project/c['parent']['input_preparation']
    protected = {}
    for name in ('completion.json', 'preprocessing.npz', 'native_configs.json', 'native_summary_scaling.npz'):
        protected[str(prep/name)] = sha(prep/name)
    for split, folder in [('train', Path(c['parent']['raw_inputs'])), ('dev', Path(c['parent']['raw_inputs'])), ('test', project/c['parent']['test_inputs'])]:
        for name in (f'{split}_raw.npy', f'{split}_targets.npz'):
            file = folder/name
            protected[str(file)] = sha(file)
            if str(file) in c['parent']['protected']:
                assert protected[str(file)] == c['parent']['protected'][str(file)]
    atomic(output/'progress.json', dict(stage='reference_constructor_parity'))
    parity = verify_original_constructor(project)
    atomic(output/'progress.json', dict(stage='window_overlay'))
    window = prepare(project, c['database'], c['parent'], project/c['window_preparation'], c['measurements'])
    models = {}
    for variant in ('matched_tabm', 'window_summary'):
        data = Data(project, window=variant == 'window_summary')
        model = construct(data, 48, 0, variant)
        models[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                               details=model.base.info, expanded_width=model.base.input_members.shape[1])
        assert all(p.requires_grad for p in model.parameters())
        if variant == 'window_summary':
            assert models[variant]['parameters'] == c['parent']['full_ema']['48:0']['parameters']
        del model, data; gc.collect()
    result = dict(complete=True, source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  formal_fits=20, old_constructor_parity=parity, original_protected=protected,
                  windows_sha256=sha(project/c['window_preparation']/'completion.json'),
                  windows_bytes=window['total_bytes'], unit_checks=checks(), models=models,
                  test_labels_used_for_fitting=False, optimizer_updates=0)
    atomic(output/'completion.json', result)
    atomic(output.parent/'verification.json', result)
    print(json.dumps(dict(complete=True, models={k: v['parameters'] for k, v in models.items()})), flush=True)


if __name__ == '__main__':
    main()
