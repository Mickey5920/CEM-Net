"""Matched CPU construction costs for the three 15-coordinate lab summaries."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from scripts.cem_targeted_common import configuration, RUN
from scripts.cem_targeted_features import observation_summary
from scripts.cem_matched_windows import window_summary
from scripts.tracehr_value_features import duration_summary
from scripts.tracehr_multiview_common import read, sha, atomic


METHODS = ('duration', 'window', 'observation')


def worker(project, output, method, size, repetition):
    import pandas as pd
    cfg = configuration()
    source = next(x for x in cfg['measurements']['sources'] if 'test' in x['splits'])
    raw = project/source['path']
    assert sha(raw) == source['sha256']
    with np.load(project/cfg['parent']['test_inputs']/'test_targets.npz', allow_pickle=False) as z:
        all_ids = z['ids'][:, -1]
    ids = all_ids[np.unique(np.linspace(0, len(all_ids)-1, min(size, len(all_ids)), dtype=int))]
    # Reading the same file during verification warms the same source before every trial.
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    frame = pd.read_parquet(raw, columns=['stay_id', 'available_h', 'lab', 'unit', 'value'])
    frame = frame[frame.stay_id.isin(ids) & frame.available_h.between(0, 24, inclusive='both')].copy()
    frame['key'] = frame.lab.astype(str)+'|'+frame.unit.astype(str)
    keys, norm = cfg['measurements']['keys'], cfg['measurements']['normalization']
    frame = frame[frame.key.isin(keys)]
    frame = frame.groupby(['key', 'stay_id', 'available_h'], as_index=False, observed=True).agg(value=('value', 'median'))
    frame['z'] = np.clip((frame.value-frame.key.map({k:norm[k]['train_median'] for k in keys}))/frame.key.map({k:norm[k]['train_iqr_or_std'] for k in keys}), -5, 5)
    frame = frame.sort_values(['key', 'stay_id', 'available_h'], kind='stable')
    preparation = time.perf_counter()-begin
    preparation_cpu = time.process_time()-cpu_begin
    row_of = {int(s):i for i,s in enumerate(ids)}
    key_of = {k:i for i,k in enumerate(keys)}
    operation = dict(duration=duration_summary, window=window_summary, observation=observation_summary)[method]
    matrix = np.zeros((len(ids), 480), np.float32)
    tick = time.perf_counter()
    cpu_tick = time.process_time()
    for (key, stay), part in frame.groupby(['key', 'stay_id'], sort=False, observed=True):
        offset = key_of[key]*15
        matrix[row_of[int(stay)], offset:offset+15] = operation(part.available_h.to_numpy(), part.z.to_numpy())
    aggregate = time.perf_counter()-tick
    aggregate_cpu = time.process_time()-cpu_tick
    elapsed = time.perf_counter()-begin
    elapsed_cpu = time.process_time()-cpu_begin
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert np.isfinite(matrix).all()
    if method == 'duration':
        original = np.load(project/cfg['parent']['test_inputs']/'test_raw.npy', mmap_mode='r')
        rows = np.unique(np.linspace(0, len(all_ids)-1, len(ids), dtype=int))
        np.testing.assert_allclose(matrix, original[rows, -480:], rtol=2e-6, atol=2e-6)
    elif method == 'observation':
        original = np.load(project/cfg['overlay']/'test_observations.npy', mmap_mode='r')
        rows = np.unique(np.linspace(0, len(all_ids)-1, len(ids), dtype=int))
        np.testing.assert_allclose(matrix, original[rows], rtol=2e-6, atol=2e-6)
    else:
        matched = read(project/cfg['original_matched_source']/'work/cem_matched_controls_20261105/config.json')
        original = np.load(project/matched['window_preparation']/'test_windows.npy', mmap_mode='r')
        rows = np.unique(np.linspace(0, len(all_ids)-1, len(ids), dtype=int))
        np.testing.assert_allclose(matrix, original[rows], rtol=2e-6, atol=2e-6)
    result = dict(database=cfg['database'], method=method, stays=len(ids), repetition=repetition,
                  records=len(frame), columns=480, preprocessing_seconds=preparation,
                  aggregation_seconds=aggregate, end_to_end_seconds=elapsed,
                  preprocessing_cpu_seconds=preparation_cpu, aggregation_cpu_seconds=aggregate_cpu, end_to_end_cpu_seconds=elapsed_cpu,
                  initial_peak_rss_mib=rss0, peak_rss_mib=peak_rss,
                  source_sha256=source['sha256'], replay_passed=True, load_average=os.getloadavg(),
                  cache='source read immediately before each trial; warm OS page cache, fresh process',
                  scope='15-summary laboratory block: parquet read, admission, merge, normalization, sort, aggregation; excludes shared event/simple-lab blocks and model training',
                  implementation='same per-stay/lab grouped traversal and output layout for all three operators', threads=2,
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
    atomic(output, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--size', type=int)
    p.add_argument('--repetition', type=int)
    a = p.parse_args()
    project, output = a.project.resolve(strict=True), a.output.resolve()
    assert output.is_relative_to(project)
    if a.worker:
        assert not output.exists()
        worker(project, output, a.method, a.size, a.repetition)
        return
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output/'completion.json').exists()
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2
    os.sched_setaffinity(0, set(allowed[-2:]))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    rows = []
    for size in (128, 512, 2048):
        for repetition in range(3):
            order = METHODS[repetition:]+METHODS[:repetition]
            for method in order:
                target = output/f'{method}_{size}_{repetition}.json'
                if not target.exists():
                    atomic(output/'progress.json', dict(size=size, repetition=repetition, method=method, completed=len(rows)))
                    cmd = [sys.executable, '-B', str(Path(__file__)), '--project', str(project), '--output', str(target),
                           '--worker', '--method', method, '--size', str(size), '--repetition', str(repetition)]
                    subprocess.run(cmd, env=env, check=True)
                rows.append(read(target))
    summary = []
    for size in (128,512,2048):
        for method in METHODS:
            current = [r for r in rows if r['method'] == method and r['stays'] == size]
            assert len(current) == 3 and len({r['records'] for r in current}) == 1
            summary.append(dict(method=method, stays=size, records=current[0]['records'],
                                **{name:float(np.median([r[name] for r in current])) for name in ('preprocessing_seconds','aggregation_seconds','end_to_end_seconds','peak_rss_mib','preprocessing_cpu_seconds','aggregation_cpu_seconds','end_to_end_cpu_seconds')}))
    atomic(output/'results.json', dict(complete=True, database=configuration()['database'], rows=rows, summary=summary,
                                      cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown'),
                                      scope=rows[0]['scope'], implementation=rows[0]['implementation']))
    atomic(output/'completion.json', dict(complete=True, measurements=27, files={'results.json':sha(output/'results.json')}))
    print(json.dumps(dict(complete=True, measurements=27)))


if __name__ == '__main__':
    main()
