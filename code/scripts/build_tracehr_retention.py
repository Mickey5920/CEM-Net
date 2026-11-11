"""Streaming full train/dev reconstruction with an exact historical-A gate."""
import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''):
            h.update(b)
    return h.hexdigest()


def sample_sha(sample):
    return hashlib.sha256(json.dumps(sample, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bridge_failure(sample, expected_sha, expected_fields, seen, window, partition):
    """Return non-PHI diagnostic evidence without weakening exact identity."""
    stay = sample['stay_id']
    actual = sample_sha(sample)
    if stay not in seen and actual == expected_sha:
        return None
    actual_fields = {key: sample_sha(value) for key, value in sample.items()}
    return dict(window=window, partition=partition,
        reason='duplicate_stay' if stay in seen else 'sample_content_mismatch',
        stay_fingerprint=hashlib.sha256(str(stay).encode()).hexdigest(),
        expected_sha256=expected_sha, actual_sha256=actual,
        differing_fields=sorted(key for key in set(actual_fields)|set(expected_fields)
            if actual_fields.get(key) != expected_fields.get(key)))


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    # Windows readers can briefly deny replacement. Retain the old valid JSON
    # and retry only access conflicts, with a bounded deadline.
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def configure_dataframe_runtime():
    import pandas as pd
    import numpy as np
    # Local NumExpr 2.11 / 16-thread comparisons reproducibly returned all-False
    # masks for valid million-row inputs. Keep exact predicates on NumPy's path.
    pd.set_option('compute.use_numexpr', False)
    return dict(pandas=pd.__version__,numpy=np.__version__,use_numexpr=False,
                reason='reproducible local multithreaded comparison-mask failure')


def main():
    parser = argparse.ArgumentParser()
    for name in ['project', 'source-project', 'raw-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--partitions', type=int, default=64)
    parser.add_argument('--pilot-raw', type=Path)
    parser.add_argument('--reuse-raw-dir', type=Path)
    parser.add_argument('--raw-audit', type=Path)
    parser.add_argument('--split-scope', choices=['train_dev', 'test'], default='train_dev')
    args = parser.parse_args()
    splits = ['test'] if args.split_scope == 'test' else ['train', 'dev']
    assert args.split_scope != 'test' or not (args.reuse_raw_dir or args.pilot_raw)
    assert bool(args.reuse_raw_dir) == bool(args.raw_audit)
    assert not (args.pilot_raw and args.reuse_raw_dir)
    root, source, raw_root = [p.resolve(strict=True) for p in (args.project,args.source_project,args.raw_root)]
    out = args.output.resolve()
    assert out == Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(root) and out != root and out.drive.lower() != 'c:'
    assert 8 <= args.partitions <= 256
    assert not any(p.name not in {'.storage-run.lock','storage-launch.json','stdout.log','stderr.log','launch.json'} for p in out.iterdir())
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    dataframe_runtime=configure_dataframe_runtime()
    sys.path.insert(0, str(source/'Experiments/scripts'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import build_mimiciv_exp2_event_windows as old
    import build_mimiciv_exp2_event_windows_v2 as old_v2
    import convert_mimiciv_exp2_to_prediction_pkl_v2 as conv
    from scripts.tracehr_retention_policy import select
    state = dict(pid=os.getpid(), status='preflight', test_loaded=args.split_scope == 'test', split_scope=args.split_scope, pilot=bool(args.pilot_raw),
                 started=time.time(), windows={}, partitions=args.partitions,dataframe_runtime=dataframe_runtime)
    def save(**changes):
        state.update(changes,updated=time.time())
        atomic(out/'state.json',state)
    try:
        pilot = json.loads((root/'runs/tracehr_retention_raw_pilot_20261002/report.json').read_text())
        assert pilot['complete'] and all(v['exact_all_fields_match'] for v in pilot['windows'].values())
        source_hashes={name:sha(source/'Experiments/scripts'/name) for name in pilot['source_hashes'] if name != 'tracehr_retention_pilot.py'}
        assert all(pilot['source_hashes'][k] == v for k,v in source_hashes.items())
        freeze=json.loads((root/'paper/ICDE2027/protocol.freeze.json').read_text())
        assert sha(root/freeze['protocol']) == freeze['sha256']
        old_bridge=json.loads((root/'runs/tracehr_selected_full_bridge_20261002/report.json').read_text())
        assert old_bridge['complete']
        headers, indices, hashes, refs, metadata, vocabs = {}, {}, {}, {}, {}, {}
        reference_fields = {}
        subset = None
        if args.pilot_raw:
            raw_pilot = pd.read_parquet(args.pilot_raw.resolve(strict=True))
            subset=set(raw_pilot.stay_id.astype(int))
        for w in (24,48):
            fold=root/'data'/f'MimicExperiment2ClinicalV2Full{w}h'/'fold0'
            metadata[w]=json.loads((fold/'metadata.json').read_text())
            vocabs[w]=json.loads((fold/'token_to_id.json').read_text())
            headers[w], indices[w], hashes[w], refs[w] = {}, {}, {}, {}
            reference_fields[w] = {}
            for split in splits:
                path=fold/f'{split}.pkl'
                hashes[w][split]=sha(path)
                if split != 'test':
                    assert hashes[w][split] == old_bridge['windows'][str(w)]['input_sha256'][split]
                with path.open('rb') as f:
                    payload=pickle.load(f)
                headers[w][split]={k:v for k,v in payload.items() if k != split}
                samples=payload[split]
                if subset is not None:
                    samples=[s for s in samples if s['stay_id'] in subset]
                indices[w][split]=[s['stay_id'] for s in samples]
                refs[w].update({s['stay_id']:sample_sha(s) for s in samples})
                reference_fields[w].update({s['stay_id']:{k:sample_sha(v) for k,v in s.items()} for s in samples})
                del payload,samples
                gc.collect()
        all_stays=set(refs[24])|set(refs[48])
        assert all_stays
        processed=source/'data/processed/mimiciv_exp2_v2_full'
        cohort=pd.read_parquet(processed/'cohort_labels_v2.parquet')
        cohort=cohort[cohort.stay_id.isin(all_stays)].copy()
        assert set(cohort.stay_id) == all_stays and cohort.stay_id.is_unique
        estimated_raw=pilot['raw_parquet_bytes']/pilot['stays']*len(all_stays)
        # 32 GiB covers a large compression/partition overhead factor and two PKL copies.
        budget=32*2**30
        assert estimated_raw*8 < budget/2
        assert shutil.disk_usage(out).free > budget+10*2**30
        assert psutil.virtual_memory().available > 12*2**30
        save(status='extracting',input_hashes=hashes,source_hashes=source_hashes,
             protocol_sha256=freeze['sha256'],builder_sha256=sha(Path(__file__)),
             policy_sha256=sha(Path(__file__).with_name('tracehr_retention_policy.py')),
             storage=dict(expected_gib=32,raw_pilot_scaled_bytes=estimated_raw,
                 overhead_factor=8,scope='all generated data/raw/shards/state below output; temp/cache declared by guard'),
             cohort_stays=len(all_stays))
        if args.reuse_raw_dir:
            raw_dir=args.reuse_raw_dir.resolve(strict=True)
            assert raw_dir.is_relative_to(root) and raw_dir != out
            audit=json.loads(args.raw_audit.resolve(strict=True).read_text())
            assert audit['status']=='complete' and audit['all_exact'] and audit['test_loaded'] is False
            assert audit.get('raw_counts_exact') is True and audit.get('use_numexpr') is False
            assert len(audit['raw_manifest']) == args.partitions
            assert set(audit['raw_manifest']) == {p.name for p in raw_dir.glob('*.parquet')}
            for name,entry in audit['raw_manifest'].items():
                path=raw_dir/name
                assert path.stat().st_size == entry['bytes'] and sha(path)==entry['sha256']
            save(raw_reuse=dict(path=str(raw_dir),audit_sha256=sha(args.raw_audit.resolve(strict=True)),read_only=True))
        else:
            raw_dir=out/'raw';raw_dir.mkdir()
        writers={};raw_rows=0
        def sink(frame):
            nonlocal raw_rows
            if frame.empty:
                return
            raw_rows+=len(frame)
            # Write the historical float64 extraction output; float32 conversion stays at historical processing stage.
            for bucket,part in frame.groupby(frame.stay_id.astype('int64') % args.partitions, sort=False):
                table=pa.Table.from_pandas(part, preserve_index=False)
                bucket=int(bucket)
                if bucket not in writers:
                    writers[bucket]=pq.ParquetWriter(raw_dir/f'{bucket:03}.parquet',table.schema,compression='snappy')
                writers[bucket].write_table(table)
            if shutil.disk_usage(out).free < 12*2**30:
                raise RuntimeError('Output reserve exhausted')
        if args.reuse_raw_dir:
            raw_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in raw_dir.glob('*.parquet'))
        elif subset is not None:
            sink(raw_pilot)
            del raw_pilot
        else:
            prepared=old.prepare_cohort(cohort,24.)
            assert set(prepared.stay_id) == all_stays
            prepared['window_end']=prepared['window_start']+pd.to_timedelta(
                np.where(prepared.stay_id.isin(refs[48]),48.,24.),unit='h')
            original_finalize=old.finalize_events
            def streaming_finalize(*items):
                frame=original_finalize(*items)
                sink(frame)
                return old.empty_events_frame()
            old.finalize_events=streaming_finalize
            extract_args=SimpleNamespace(window_hours=48.,chunksize=250000,
                sources=['chartevents','labevents','inputevents','procedureevents'])
            try:
                discarded,tables=old.extract_sources(extract_args,raw_root,prepared)
                assert discarded.empty
                state['source_tables']=tables
            finally:
                old.finalize_events=original_finalize
        for writer in writers.values():
            writer.close()
        del writers
        save(status='building_policies',raw_rows=raw_rows,
             raw_bytes=sum(p.stat().st_size for p in raw_dir.glob('*.parquet')))
        for w in (24,48):
            shard_dir=out/'shards'/str(w);shard_dir.mkdir(parents=True)
            seen=set();stats={k:dict(samples=0,events=0,oov=0) for k in 'ABCD'}
            for path in sorted(raw_dir.glob('*.parquet')):
                save(status='building_policies',window=w,current_partition=path.stem,verified_stays=len(seen))
                raw=pd.read_parquet(path)
                raw=raw[raw.stay_id.isin(refs[w]) & raw.rel_time_hours.between(0,w)]
                if raw.empty:
                    continue
                events,_,_=old_v2.process_raw_events(raw,SimpleNamespace(window_hours=float(w),min_len=5))
                del raw
                events=events.merge(cohort[['stay_id']+metadata[w]['tasks']+metadata[w]['aux_tasks']],on='stay_id',validate='many_to_one')
                grouped=list(events.groupby('stay_id',sort=False))
                for policy in 'ABCD':
                    chosen=pd.concat([select(g,policy) for _,g in grouped],ignore_index=True)
                    mapped=conv.map_tokens(chosen,vocabs[w],set(vocabs[w])-set(conv.source_unk_tokens()))
                    samples=conv.build_samples(mapped,metadata[w]['tasks'],metadata[w]['aux_tasks'],float(w))
                    stats[policy]['samples']+=len(samples)
                    stats[policy]['events']+=len(mapped)
                    stats[policy]['oov']+=int(mapped['_oov'].sum())
                    if policy == 'A':
                        for sample in samples:
                            stay=sample['stay_id']
                            failure=bridge_failure(sample,refs[w][stay],reference_fields[w][stay],seen,w,path.stem)
                            if failure:
                                atomic(out/'bridge_failure.json',failure)
                                raise ValueError(f"{w}h partition {path.stem}: {failure['reason']}; fields={failure['differing_fields']}")
                            seen.add(stay)
                    else:
                        with (shard_dir/f'{policy}_{path.stem}.pkl').open('xb') as f:
                            pickle.dump(samples,f,protocol=pickle.HIGHEST_PROTOCOL)
                    del chosen,mapped,samples
                del grouped,events
                gc.collect()
                save(status='building_policies',window=w,partition=path.stem,verified_stays=len(seen))
                print(f'{w}h partition {path.stem}: A verified {len(seen)}/{len(refs[w])}',flush=True)
            assert seen == set(refs[w]), 'Missing stays after raw reconstruction'
            state['windows'][str(w)]=dict(historical_A_exact=True,policies=stats)
            save(status='assembling',window=w)
            for policy in 'BCD':
                all_samples={}
                for path in sorted(shard_dir.glob(f'{policy}_*.pkl')):
                    with path.open('rb') as f:
                        part=pickle.load(f)
                    for s in part:
                        assert s['stay_id'] not in all_samples
                        all_samples[s['stay_id']]=s
                    del part
                assert set(all_samples)==set(refs[w])
                dest=out/'data'/f'mimic{w}'/policy/'fold0';dest.mkdir(parents=True)
                manifests={}
                for split in splits:
                    samples=[all_samples[k] for k in indices[w][split]]
                    with (dest/f'{split}.pkl').open('xb') as f:
                        pickle.dump(dict(headers[w][split],**{split:samples}),f,protocol=pickle.HIGHEST_PROTOCOL)
                    manifests[split]=dict(samples=len(samples),sha256=sha(dest/f'{split}.pkl'))
                # Copy schema maps, but never misleading historical distribution summaries.
                original=root/'data'/f'MimicExperiment2ClinicalV2Full{w}h'/'fold0'
                for name in ['token_to_id.json','id_to_token.json','source_to_id.json']:
                    shutil.copyfile(original/name,dest/name)
                meta=dict(dataset='MIMIC-IV',window_hours=float(w),tasks=metadata[w]['tasks'],aux_tasks=metadata[w]['aux_tasks'],
                    num_event_types_raw=len(vocabs[w]),vocab_scope='frozen_original_train_only',retention_policy=policy,
                    **{'n_'+s:len(indices[w][s]) for s in splits},test_included=args.split_scope == 'test',
                    source_input_sha256=hashes[w],protocol_sha256=freeze['sha256'],pilot=bool(subset))
                atomic(dest/'metadata.json',meta)
                atomic(dest/'data_manifest.json',dict(splits=manifests,original_A_full_match=True,policy=policy))
                del samples,all_samples
                gc.collect()
            save()
        save(status='complete',complete=True,elapsed_seconds=time.time()-state['started'],
             peak_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
             output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        atomic(out/'completion.json',state)
        print('All A raw bridges and B/C/D input artifacts complete',flush=True)
    except Exception as error:
        save(status='needs_attention',error=str(error))
        raise


if __name__ == '__main__':
    main()
