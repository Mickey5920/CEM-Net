"""Same basis/kernel/coverage as duration, using observation-point weights instead."""
import argparse
import gc
import hashlib
import time
from pathlib import Path
import numpy as np
from scripts.tracehr_value_features import TAUS, basis, duration_summary
from scripts.tracehr_multiview_common import read, atomic, sha


def observation_summary(times, values, window=24.):
    t, z = np.asarray(times, float), np.asarray(values, float)
    assert t.ndim == 1 and t.shape == z.shape and np.all(np.diff(t) > 0)
    assert np.isfinite(z).all() and np.all((t >= 0) & (t <= window))
    if not len(t):
        return np.zeros(15, np.float32)
    w = np.exp(-(window-t[:, None])/TAUS)
    coverage = -np.expm1(-(window-t[0])/TAUS)/(-np.expm1(-window/TAUS))
    return (coverage[:, None]*(w.T@basis(z))/w.sum(0)[:, None]).astype(np.float32).ravel()


def discrepancy(times, values):
    t, z = np.asarray(times, float), np.asarray(values, float)
    if len(t) < 2 or t[0] >= 24:
        return np.nan
    elapsed = np.diff(np.r_[t, 24.])
    return float(abs(elapsed@z/elapsed.sum()-z.mean()))


def checks():
    assert not observation_summary([], []).any()
    for t, z in [([0], [2.]), ([2, 8, 19], [1.7]*3), ([24], [3.])]:
        np.testing.assert_allclose(observation_summary(t, z), duration_summary(t, z), rtol=2e-6, atol=2e-6)
    t, z = [0, 4, 20], [-2., 3., 1.]
    a, b = observation_summary(t, z).reshape(3, 5), duration_summary(t, z).reshape(3, 5)
    np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=2e-6, atol=2e-6)
    assert not np.allclose(a[:, 1:], b[:, 1:])
    np.testing.assert_allclose(duration_summary(t, z), duration_summary([0, 2, 4, 20], [-2., -2., 3., 1.]), atol=2e-6)
    assert not np.allclose(observation_summary(t, z), observation_summary([0, 2, 4, 20], [-2., -2., 3., 1.]))
    assert discrepancy([0, 12], [0, 2]) == 0
    np.testing.assert_allclose(discrepancy([0, 18], [0, 2]), .5)
    return ['same_coverage', 'constant_path_equality', 'empty_and_zero_coverage',
            'distinct_measures', 'same_value_insertion_discriminates', 'discrepancy_known_values']


def prepare(project, cfg, output):
    import pandas as pd
    import torch
    from pytabkit.models.torch_utils import torch_np_quantile
    from scripts.cem_targeted_common import ROOT
    started = time.time()
    project, output = Path(project).resolve(strict=True), Path(output).resolve()
    assert output.is_relative_to(project) and not output.exists()
    output.mkdir(parents=True)
    parent, measures = cfg['parent'], cfg['measurements']
    keys, norm = measures['keys'], measures['normalization']
    assert len(keys) == 32
    prep, raw, test = project/parent['input_preparation'], Path(parent['raw_inputs']), project/parent['test_inputs']
    record = read(prep/'completion.json')
    expected = record['artifacts']['normalization.json']['sha256'] if cfg['database'] == 'mimiciii' else record['trajectory_normalization_sha256']
    assert expected == measures['normalization_sha256']
    ids, arrays, originals, totals, numbers = {}, {}, {}, {}, {}
    for split, folder in [('train', raw), ('dev', raw), ('test', test)]:
        with np.load(folder/f'{split}_targets.npz', allow_pickle=False) as a:
            ids[split] = a['ids'].copy()
        assert len(np.unique(ids[split][:, -1])) == len(ids[split])
        originals[split] = np.load(folder/f'{split}_raw.npy', mmap_mode='r')
        arrays[split] = np.lib.format.open_memmap(output/f'{split}_observations.npy', mode='w+', dtype='float32', shape=(len(ids[split]), 480))
        arrays[split][:] = 0
        totals[split], numbers[split] = np.zeros(len(ids[split])), np.zeros(len(ids[split]), np.int32)
    for l, r in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not np.intersect1d(ids[l][:, 0], ids[r][:, 0]).size
    where = {int(sid): (split, row) for split, items in ids.items() for row, sid in enumerate(items[:, -1])}
    assert len(where) == sum(map(len, ids.values()))
    key_index = {key: i for i, key in enumerate(keys)}
    replay_ids = {int(s) for values in ids.values() for s in values[np.linspace(0, len(values)-1, 16, dtype=int), -1]}
    protected, replay_count, rows, seen = {}, 0, 0, set()
    for item in measures['sources']:
        source = project/item['path']
        assert source.resolve().is_relative_to(project)
        protected[str(source)] = sha(source)
        assert protected[str(source)] == item['sha256']
        frame = pd.read_parquet(source)
        admitted = np.concatenate([ids[s][:, -1] for s in item['splits']])
        frame = frame[frame.stay_id.isin(admitted) & frame.available_h.between(0, 24, inclusive='both')].copy()
        frame['key'] = frame.lab.astype(str)+'|'+frame.unit.astype(str)
        frame = frame[frame.key.isin(keys)]
        frame = frame.groupby(['key', 'stay_id', 'available_h'], as_index=False, observed=True).agg(value=('value', 'median'))
        frame['z'] = np.clip((frame.value-frame.key.map({k: norm[k]['train_median'] for k in keys}))/frame.key.map({k: norm[k]['train_iqr_or_std'] for k in keys}), -5, 5)
        frame = frame.sort_values(['key', 'stay_id', 'available_h'], kind='stable').reset_index(drop=True)
        current = set(frame.stay_id.astype(int))
        assert not seen & current
        seen |= current
        t, z = frame.available_h.to_numpy(), frame.z.to_numpy()
        assert np.isfinite(t).all() and np.isfinite(z).all()
        group = frame.groupby(['key', 'stay_id'], sort=False, observed=True)
        code = group.ngroup().to_numpy()
        starts = np.r_[0, np.flatnonzero(np.diff(code))+1]
        ends = np.r_[starts[1:], len(frame)]
        counts = ends-starts
        first = frame.iloc[starts]
        sids = first.stay_id.to_numpy()
        ki = np.array([key_index[k] for k in first.key])
        # The order is identical to the original: time scale first, then basis.
        phi = basis(z)
        for ti, tau in enumerate(TAUS):
            weights = np.exp(-(24-t)/tau)
            sums = np.add.reduceat(weights, starts)
            weighted = np.add.reduceat(weights[:, None]*phi, starts, axis=0)/sums[:, None]
            coverage = -np.expm1(-(24-t[starts])/tau)/(-np.expm1(-24/tau))
            values = (weighted*coverage[:, None]).astype('float32')
            for split in item['splits']:
                chosen = np.array([where[int(s)][0] == split for s in sids])
                rr = np.array([where[int(s)][1] for s in sids[chosen]])
                cc = ki[chosen]*15+ti*5
                arrays[split][rr[:, None], cc[:, None]+np.arange(5)] = values[chosen]
        end_t = np.r_[t[1:], 24.]
        end_t[ends-1] = 24.
        elapsed = end_t-t
        mass = 24-t[starts]
        eligible = (counts >= 2) & (mass > 0)
        value_sum = np.add.reduceat(z, starts)
        duration_sum = np.add.reduceat(elapsed*z, starts)
        gap = abs(duration_sum/np.maximum(mass, 1e-30)-value_sum/counts)
        for sid, value in zip(sids[eligible], gap[eligible]):
            split, rr = where[int(sid)]
            totals[split][rr] += value
            numbers[split][rr] += 1
        for (key, sid), part in frame[frame.stay_id.isin(replay_ids)].groupby(['key', 'stay_id'], sort=False):
            split, rr = where[int(sid)]
            start = key_index[key]*15
            duration = duration_summary(part.available_h, part.z)
            observed = observation_summary(part.available_h, part.z)
            np.testing.assert_allclose(originals[split][rr, -480+start:-480+start+15 if start < 465 else None], duration, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(arrays[split][rr, start:start+15], observed, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(observed.reshape(3, 5)[:, 0], duration.reshape(3, 5)[:, 0], rtol=2e-6, atol=2e-6)
            replay_count += 1
        rows += len(frame)
        atomic(output/'progress.json', dict(stage='observations_prepared', sources=len(protected), rows=rows, replay_checks=replay_count))
        del frame, phi, group
        gc.collect()
    assert replay_count >= 32
    old = dict(np.load(prep/'preprocessing.npz', allow_pickle=False))
    old.update(dict(np.load(prep/'native_summary_scaling.npz', allow_pickle=False)))
    center, factor = old['center'].copy(), old['factor'].copy()
    for begin in range(0, 480, 32):
        block = torch.from_numpy(np.array(arrays['train'][:, begin:begin+32], copy=True))
        median = torch_np_quantile(block, .5, dim=0)
        centered = block-median
        spread = torch_np_quantile(centered, .75, dim=0)-torch_np_quantile(centered, .25, dim=0)
        spread = torch.where(spread == 0, .5*(centered.max(0).values-centered.min(0).values), spread)
        scale = torch.where(spread == 0, torch.zeros_like(spread), 1/(spread+1e-30))
        begin_full = len(center)-480+begin
        center[begin_full:begin_full+32], factor[begin_full:begin_full+32] = median.numpy(), scale.numpy()
    np.testing.assert_array_equal(center[:-480], old['center'][:-480])
    np.testing.assert_array_equal(factor[:-480], old['factor'][:-480])
    np.savez_compressed(output/'scaling.npz', center=center, factor=factor, numeric_indices=old['numeric_indices'], binary=old['binary'])
    gap_by_split = {s:np.divide(totals[s], numbers[s], out=np.full(len(numbers[s]), np.nan), where=numbers[s] > 0) for s in ids}
    threshold = float(np.nanmedian(gap_by_split['train']))
    assert np.isfinite(threshold)
    files, grouping = {}, {}
    for split, values in arrays.items():
        assert np.isfinite(values).all()
        values.flush()
        gap = gap_by_split[split]
        groups = np.where(~np.isfinite(gap), 2, (gap > threshold).astype('int8'))
        np.savez_compressed(output/f'{split}_groups.npz', ids=ids[split], discrepancy=gap, eligible_labs=numbers[split], group=groups)
        grouping[split] = {str(g):int(np.sum(groups == g)) for g in (0, 1, 2)}
        for name in (f'{split}_observations.npy', f'{split}_groups.npz'):
            files[name] = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
    files['scaling.npz'] = dict(sha256=sha(output/'scaling.npz'), bytes=(output/'scaling.npz').stat().st_size)
    result = dict(complete=True, database=cfg['database'], changed_coordinates=480, keys=keys,
                  definition='kernel-mass coverage times observation-point kernel-weighted basis mean; same 5 bases and 3 taus as duration',
                  normalization='same full-window kernel mass as original duration', threshold=threshold, threshold_fit_split='train',
                  groups={'0':'discrepancy <= training median', '1':'discrepancy > training median', '2':'no eligible lab'},
                  discrepancy='mean over labs with >=2 observations and positive coverage of abs(time-weighted mean - observation mean), using original standardized clipped values',
                  grouping=grouping, source_hashes=protected, normalization_sha256=measures['normalization_sha256'],
                  artifacts=files, total_bytes=sum(x['bytes'] for x in files.values()), duration_replay_checks=replay_count,
                  unit_checks=checks(), labels_used=False, seconds=time.time()-started,
                  source_sha256=sha(Path(__file__)))
    atomic(output/'completion.json', result)
    return result


if __name__ == '__main__':
    print(checks())
