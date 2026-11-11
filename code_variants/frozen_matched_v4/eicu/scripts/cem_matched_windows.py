"""Equal-width, training-fitted window summaries for the approved CEM controls."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


WINDOWS = (1., 6., 24.)
WIDTH = 480


def window_summary(times, values):
    t, z = np.asarray(times, float), np.asarray(values, float)
    assert t.shape == z.shape and t.ndim == 1
    assert np.isfinite(t).all() and np.isfinite(z).all()
    assert np.all((t >= 0) & (t <= 24)) and np.all(np.diff(t) > 0)
    result = np.zeros((3, 5), np.float32)
    for i, w in enumerate(WINDOWS):
        v = z[t >= 24-w]
        if len(v):
            result[i] = (np.log1p(len(v)), v.mean(), v.min(), v.max(), v[-1])
    return result.ravel()


class OverlayMatrix:
    """Read existing dense rows, then replace only the final 480 coordinates."""
    def __init__(self, original, window):
        assert original.ndim == window.ndim == 2
        assert window.shape == (len(original), WIDTH)
        self.original, self.window = original, window
        self.shape, self.ndim, self.dtype = original.shape, 2, original.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        rows, columns = index if isinstance(index, tuple) else (index, slice(None))
        block = np.array(self.original[rows], dtype=np.float32, copy=True)
        block[..., -WIDTH:] = self.window[rows]
        return block[..., columns]


def checks():
    assert not window_summary([], []).any()
    np.testing.assert_array_equal(window_summary([24], [0]),
                                  np.tile([np.log(2), 0, 0, 0, 0], 3).astype('float32'))
    x = window_summary([0, 18, 23, 24], [-2, 1, 3, 4]).reshape(3, 5)
    np.testing.assert_allclose(x, [[np.log(3), 3.5, 3, 4, 4],
                                [np.log(4), 8/3, 1, 4, 4],
                                [np.log(5), 1.5, -2, 4, 4]], rtol=1e-7)
    a = np.arange(3*484, dtype='float32').reshape(3, 484)
    b = np.full((3, 480), 7, 'float32'); o = OverlayMatrix(a, b)
    np.testing.assert_array_equal(o[[2, 0], :4], a[[2, 0], :4])
    np.testing.assert_array_equal(o[1, -480:], b[1])
    assert a[1, -1] != 7
    return ['closed_window_boundaries', 'empty_vs_observed_zero', 'window_major_layout',
            'overlay_preserves_other_coordinates_and_source']


def prepare(project, database, parent, output, measurements):
    import pandas as pd
    import torch
    from scripts.tracehr_multiview_common import read, sha, atomic
    from scripts.tracehr_value_features import duration_summary
    from pytabkit.models.torch_utils import torch_np_quantile

    started = time.time()
    project, output = Path(project).resolve(strict=True), Path(output).resolve()
    assert output.is_relative_to(project) and not output.exists()
    output.mkdir(parents=True)
    prep = project/parent['input_preparation']; raw = Path(parent['raw_inputs'])
    test = project/parent['test_inputs']
    keys = measurements['keys']
    sources = [project/item['path'] for item in measurements['sources']]
    assert len(keys) == 32 and len(set(keys)) == 32
    norm = measurements['normalization']
    record = read(prep/'completion.json')
    expected_norm = record['artifacts']['normalization.json']['sha256'] if database == 'mimiciii' else record['trajectory_normalization_sha256']
    assert measurements['normalization_sha256'] == expected_norm
    ids, originals, arrays = {}, {}, {}
    for split, folder in [('train', raw), ('dev', raw), ('test', test)]:
        # Only identities are read here; evaluation labels never enter feature fitting.
        with np.load(folder/f'{split}_targets.npz') as targets:
            ids[split] = targets['ids'][:, -1].astype('int64')
        assert len(np.unique(ids[split])) == len(ids[split])
        originals[split] = np.load(folder/f'{split}_raw.npy', mmap_mode='r')
        arrays[split] = np.lib.format.open_memmap(output/f'{split}_windows.npy', mode='w+',
                                                dtype='float32', shape=(len(ids[split]), WIDTH))
        arrays[split][:] = 0
    combined = np.concatenate(list(ids.values()))
    assert len(np.unique(combined)) == len(combined)
    split_of = {int(sid): (split, i) for split, values in ids.items() for i, sid in enumerate(values)}
    key_index = {k: i for i, k in enumerate(keys)}
    source_hashes, matches, scanned = {}, 0, 0
    selected_checks = {int(v) for values in ids.values() for v in values[np.linspace(0, len(values)-1, 16, dtype=int)]}
    seen = set()
    for source, description in zip(sources, measurements['sources']):
        assert source.resolve().is_relative_to(project) and source.is_file()
        source_hashes[str(source)] = sha(source)
        assert source_hashes[str(source)] == description['sha256']
        frame = pd.read_parquet(source)
        admitted_ids = np.concatenate([ids[s] for s in description['splits']])
        frame = frame[frame.stay_id.isin(admitted_ids) & frame.available_h.between(0, 24, inclusive='both')].copy()
        frame['key'] = frame.lab.astype(str)+'|'+frame.unit.astype(str)
        frame = frame[frame.key.isin(keys)]
        frame = frame.groupby(['key', 'stay_id', 'available_h'], as_index=False, observed=True).agg(value=('value', 'median'))
        assert np.isfinite(frame.value).all()
        current = set(frame.stay_id.astype(int))
        assert not seen & current, 'Source files overlap in stays'
        seen |= current
        frame['z'] = np.clip((frame.value-frame.key.map({k: norm[k]['train_median'] for k in keys}))/
                             frame.key.map({k: norm[k]['train_iqr_or_std'] for k in keys}), -5, 5)
        frame = frame.sort_values(['key', 'stay_id', 'available_h'], kind='stable')
        scanned += len(frame)
        # Verify the original admission, merge, normalization and layout against saved input.
        for (key, sid), part in frame[frame.stay_id.isin(selected_checks)].groupby(['key', 'stay_id'], sort=False):
            split, row = split_of[int(sid)]; j = key_index[key]
            expected = duration_summary(part.available_h.to_numpy(), part.z.to_numpy())
            np.testing.assert_allclose(originals[split][row, -480+j*15:-480+(j+1)*15 if j < 31 else None],
                                       expected, rtol=2e-6, atol=2e-6)
            matches += 1
        for wi, w in enumerate(WINDOWS):
            part = frame[frame.available_h >= 24-w]
            summary = part.groupby(['key', 'stay_id'], sort=False, observed=True).z.agg(['count', 'mean', 'min', 'max', 'last'])
            summary['count'] = np.log1p(summary['count'])
            for key in keys:
                if key not in summary.index.get_level_values(0):
                    continue
                by_stay = summary.xs(key, level=0)
                active_splits = description['splits']
                for split in active_splits:
                    split_ids = ids[split]
                    aligned = by_stay.reindex(split_ids, fill_value=0).to_numpy(dtype='float32')
                    offset = key_index[key]*15+wi*5
                    arrays[split][:, offset:offset+5] = aligned
        atomic(output/'progress.json', dict(stage='window_summaries', sources_complete=len(source_hashes),
                                           records=scanned, duration_replay_checks=matches))
    assert matches >= 32
    old = dict(np.load(prep/'preprocessing.npz'))
    old.update(dict(np.load(prep/'native_summary_scaling.npz')))
    center, factor = old['center'].copy(), old['factor'].copy()
    for begin in range(0, WIDTH, 32):
        block = torch.from_numpy(np.array(arrays['train'][:, begin:begin+32], copy=True))
        median = torch_np_quantile(block, .5, dim=0)
        centered = block-median
        spread = torch_np_quantile(centered, .75, dim=0)-torch_np_quantile(centered, .25, dim=0)
        spread = torch.where(spread == 0, .5*(centered.max(0).values-centered.min(0).values), spread)
        scale = torch.where(spread == 0, torch.zeros_like(spread), 1/(spread+1e-30))
        start = len(center)-WIDTH+begin
        center[start:start+32], factor[start:start+32] = median.numpy(), scale.numpy()
    np.testing.assert_array_equal(center[:-WIDTH], old['center'][:-WIDTH])
    np.testing.assert_array_equal(factor[:-WIDTH], old['factor'][:-WIDTH])
    assert not np.asarray(old.get('passthrough', old['binary']))[-WIDTH:].any()
    np.savez_compressed(output/'scaling.npz', center=center, factor=factor,
                        numeric_indices=old['numeric_indices'], binary=old['binary'])
    artifacts = {}
    for split, array in arrays.items():
        assert np.isfinite(array).all(); array.flush()
        f = output/f'{split}_windows.npy'
        artifacts[f.name] = dict(sha256=sha(f), bytes=f.stat().st_size, shape=list(array.shape),
                                ids_sha256=__import__('hashlib').sha256(ids[split].tobytes()).hexdigest())
    artifacts['scaling.npz'] = dict(sha256=sha(output/'scaling.npz'), bytes=(output/'scaling.npz').stat().st_size)
    result = dict(complete=True, database=database, keys=keys, windows=list(WINDOWS), statistics=['log1p_count', 'mean_z', 'min_z', 'max_z', 'latest_z'],
                  window_bounds='[24-w,24]', empty_window=[0]*5, changed_coordinates=480,
                  scaling_fit_split='train', response_slots='unchanged original numeric_indices',
                  duration_replay_checks=matches, unit_checks=checks(), source_hashes=source_hashes,
                  normalization_sha256=measurements['normalization_sha256'], artifacts=artifacts, seconds=time.time()-started,
                  total_bytes=sum(x['bytes'] for x in artifacts.values()), labels_used=False)
    atomic(output/'completion.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', required=True); p.add_argument('--database', required=True)
    p.add_argument('--parent', required=True); p.add_argument('--output', required=True); p.add_argument('--measurements', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.project, a.database, json.loads(Path(a.parent).read_text()), a.output,
                             json.loads(Path(a.measurements).read_text()))))
