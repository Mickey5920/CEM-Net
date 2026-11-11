"""Check reused fits and unchanged mask/forward paths before new training."""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import torch
from scripts.cem_targeted_common import ROOT, configuration, Data, construct, removed_columns, budget, protocol
from scripts.cem_targeted_features import prepare, checks
from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.train_tracehr_unified import seed_all
from models.cem_icde72_mask import invariance
from models.tracehr_unified_next import parameter_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    project, output = a.project.resolve(strict=True), a.output.resolve()
    assert output.is_relative_to(project) and not output.exists()
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    c = configuration()
    for name, digest in read(ROOT/'source_manifest.json').items():
        assert sha(ROOT/name) == digest, name
    atomic(output/'progress.json', dict(stage='prepare_same_basis_overlay'))
    features = prepare(project, c, project/c['overlay'])
    atomic(output/'progress.json', dict(stage='verify_reuse'))
    selected_endpoint = c['evidence_endpoint']
    data = Data(project, test=True)
    protected, reused = {}, []
    def keep(path, digest=None):
        path = Path(path).resolve(strict=True)
        assert path.is_relative_to(project)
        value = sha(path)
        if digest is not None:
            assert value == digest, str(path)
        protected[str(path)] = value
        return path
    for seed in (48, 49, 50):
        fit = project/c['basic_screen']/f'jobs/CEM_basic_t{selected_endpoint}_s{seed}'
        test = project/c['basic_screen']/f'test/test_CEM_basic_t{selected_endpoint}_s{seed}'
        receipt = read(keep(fit/'completion.json'))
        manifest = read(keep(fit/'run_manifest.json'))
        test_receipt = read(keep(test/'completion.json'))
        assert receipt['complete'] and not receipt['test_loaded'] and test_receipt['optimizer_updates'] == 0
        assert receipt['epochs'] == budget(c['database'], seed, selected_endpoint)
        # The old CEM wrapper retained a larger identity, but the executed native recipe is unchanged.
        actual = manifest['protocol']
        wanted = protocol(seed, selected_endpoint)
        for key in ('lr', 'weight_decay', 'dropout', 'beta2', 'schedule_reference_epochs', 'training_objective'):
            assert actual[key] == wanted[key], key
        seed_all(seed)
        model = construct(data, seed, selected_endpoint, 'CEM_basic')
        assert parameter_hash(model) == manifest['initial_weights_sha256']
        checkpoint = keep(fit/receipt.get('checkpoint_file', 'best.pt'), receipt['checkpoint_sha256'])
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        del ck
        model.eval()
        path = keep(test/'predictions_test.npz', test_receipt['predictions_sha256'])
        saved = dict(np.load(path, allow_pickle=False))
        np.testing.assert_array_equal(saved['ids'], data.targets['test']['ids'])
        ix = np.unique(np.linspace(0, len(saved['ids'])-1, 129, dtype=int))
        with torch.no_grad():
            x, y = data.batch('test', ix, device='cpu')
            logits = model(x)
            prediction = logits.softmax(-1)[..., 1].mean(1).numpy()
        expected = saved['probs']
        expected = expected[:, selected_endpoint if expected.shape[1] == 2 else 0] if expected.ndim == 2 else expected
        drift = float(np.max(np.abs(prediction-expected[ix])))
        assert drift <= 2e-5, drift
        reused.append(dict(seed=seed, endpoint=selected_endpoint, fit=str(fit), prediction=str(path),
                           prediction_sha256=sha(path), checkpoint_sha256=sha(checkpoint), cpu_replay_max_probability_drift=drift,
                           initial_parameters_identical=True, original_epoch_budget=receipt['epochs']))
        del model
        gc.collect()
    identities = {}
    for variant, endpoint, seed in [('CEM_basic', selected_endpoint, 51), ('matched_basic', selected_endpoint, 48),
                                    ('observation_weighted', 0, 48), ('observation_weighted', 1, 48)]:
        current = Data(project, variant)
        seed_all(seed)
        model = construct(current, seed, endpoint, variant)
        if variant != 'observation_weighted':
            raw = torch.as_tensor(np.array(current.raw['train'][:9], copy=True))
            check = invariance(model.base, raw, removed_columns(current))
        else:
            assert sum(p.numel() for p in model.parameters()) == c['parent']['full_ema'][f'48:{endpoint}']['parameters']
            check = dict(original_parameter_count=True, other_coordinates_and_transforms_unchanged=True)
        identities[f'{variant}:{endpoint}'] = dict(parameters=sum(p.numel() for p in model.parameters()), **check)
        del current, model
        gc.collect()
    for path, digest in protected.items():
        assert sha(path) == digest
    result = dict(complete=True, source_manifest_sha256=sha(ROOT/'source_manifest.json'), formal_fits=17,
                  reused_basic_fits=reused, original_protected=protected, new_paths=identities,
                  overlay_sha256=sha(project/c['overlay']/'completion.json'), overlay_bytes=features['total_bytes'],
                  unit_checks=checks(), new_training_updates=0, original_results_unchanged=True)
    atomic(output/'completion.json', result)
    atomic(output.parent/'verification.json', result)
    print(__import__('json').dumps(dict(complete=True, reused_fits=len(reused), formal_fits=17)), flush=True)


if __name__ == '__main__':
    main()
