"""Bounded, availability-aware lab extraction; development identities only.

No outcomes are read from the cohort. Files are immutable per guarded run.
The row-bounded mode is an I/O pilot, never a representative evaluation set.
"""
import argparse, collections, gzip, json, os, sys, time, zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_tracehr_retention import atomic, sha, configure_dataframe_runtime

SCHEMA = pa.schema([('stay_id', pa.int64()), ('lab', pa.string()),
    ('unit', pa.string()), ('sample_h', pa.float64()),
    ('available_h', pa.float64()), ('value', pa.float64()), ('row_id', pa.int64())])


def identities(db):
    name = 'tracehr_fullstream_features_20261003' if db == 'mimic' else 'tracehr_fullstream_eicu_features_20261003_r2'
    frames, proofs = [], {}
    for w in [24, 48]:
        for split in ['train', 'dev']:
            p = ROOT/'runs'/name/(db+str(w))/(split+'_targets.npz')
            with np.load(p) as z: ids = z['ids']
            frames.append(pd.DataFrame(ids, columns=['subject_id','hadm_id','stay_id']))
            proofs[str(p)] = sha(p)
    return pd.concat(frames).drop_duplicates('stay_id'), proofs


def normalized_unit(s):
    return s.fillna('').astype(str).str.strip().str.lower().str.replace('µ', 'u', regex=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--database', choices=['mimic','eicu'], required=True)
    p.add_argument('--max-rows', type=int, default=2000000, help='0 means full source scan')
    p.add_argument('--max-output-gib', type=float, default=2.)
    a = p.parse_args(); configure_dataframe_runtime()
    out = Path(os.environ['PROJECT_RUN_DIR']).resolve()
    assert out.is_relative_to(ROOT) and not (out/'events.parquet').exists()
    started = time.monotonic(); ids, proofs = identities(a.database)
    selected = set(ids.stay_id); counters = collections.Counter(); vocab = collections.Counter()
    source = Path('F:/Data/mimic-iv-3.1/hosp/labevents.csv.gz') if a.database=='mimic' else Path('F:/Data/eicu-collaborative-research-database-2.0.zip')
    if a.database=='mimic':
        cohort = Path('F:/Code/Mamba-Hawkes-Process/data/processed/mimiciv_exp2_v2_full/cohort_labels_v2.parquet')
        info = pd.read_parquet(cohort, columns=['subject_id','hadm_id','stay_id','intime'])
        info = info[info.stay_id.isin(selected)]
        assert len(info)==len(selected) and not info.stay_id.duplicated().any()
    with ExitStack() as stack:
        if a.database=='mimic':
            handle = stack.enter_context(gzip.open(source, 'rb'))
            cols=['labevent_id','subject_id','hadm_id','itemid','charttime','storetime','value','valuenum','valueuom']
        else:
            zf=stack.enter_context(zipfile.ZipFile(source))
            raw=stack.enter_context(zf.open('eicu-collaborative-research-database-2.0/lab.csv.gz'))
            handle=stack.enter_context(gzip.GzipFile(fileobj=raw))
            cols=['labid','patientunitstayid','labresultoffset','labname','labresult','labresulttext','labmeasurenamesystem','labmeasurenameinterface','labresultrevisedoffset']
        reader=pd.read_csv(handle, usecols=cols, chunksize=200000, dtype=str, nrows=a.max_rows or None)
        writer=stack.enter_context(pq.ParquetWriter(out/'events.parquet', SCHEMA, compression='zstd'))
        for ci, chunk in enumerate(reader):
            counters['source_rows_scanned']+=len(chunk)
            if a.database=='mimic':
                for col in ['subject_id','hadm_id']: chunk[col]=pd.to_numeric(chunk[col],errors='coerce')
                chunk=chunk[chunk.hadm_id.isin(info.hadm_id)].copy()
                chunk=chunk.merge(info,on=['subject_id','hadm_id'],how='inner',validate='many_to_many')
                sample=(pd.to_datetime(chunk.charttime,errors='coerce')-chunk.intime).dt.total_seconds()/3600
                available=(pd.to_datetime(chunk.storetime,errors='coerce')-chunk.intime).dt.total_seconds()/3600
                lab=chunk.itemid; unit=normalized_unit(chunk.valueuom)
                value=pd.to_numeric(chunk.valuenum,errors='coerce'); txt=chunk.value.fillna('')
                rowid=chunk.labevent_id
            else:
                chunk['stay_id']=pd.to_numeric(chunk.patientunitstayid,errors='coerce')
                chunk=chunk[chunk.stay_id.isin(selected)].copy()
                sample=pd.to_numeric(chunk.labresultoffset,errors='coerce')/60
                available=pd.to_numeric(chunk.labresultrevisedoffset,errors='coerce')/60
                lab=chunk.labname.fillna('').str.strip().str.lower()
                system=normalized_unit(chunk.labmeasurenamesystem); interface=normalized_unit(chunk.labmeasurenameinterface)
                unit=system.where(system.ne(''),interface)
                counters['unit_both_nonempty_different']+=int((system.ne('') & interface.ne('') & system.ne(interface)).sum())
                value=pd.to_numeric(chunk.labresult,errors='coerce'); txt=chunk.labresulttext.fillna('')
                rowid=chunk.labid
            inside=sample.between(0,48,inclusive='both')
            counters['development_join_rows']+=len(chunk)
            counters['measurement_0_48h']+=int(inside.sum())
            counters['availability_missing_0_48h']+=int((inside & available.isna()).sum())
            counters['availability_before_measurement_0_48h']+=int((inside & available.lt(sample)).sum())
            for w in [24,48]:
                mask=sample.between(0,w,inclusive='both')
                counters[f'measured_by_{w}h']+=int(mask.sum())
                counters[f'not_recorded_available_by_{w}h']+=int((mask & (available.isna()|available.gt(w))).sum())
            # A recorded revised value cannot be assigned its earlier draw time.
            # Earlier-than-draw entry anomalies use max(draw, entry), recorded separately.
            available=available.where(available.isna(),np.maximum(sample,available))
            numeric=np.isfinite(value)
            censored=txt.str.contains(r'[<>≤≥]',regex=True,na=False)
            counters['numeric_0_48h']+=int((inside & numeric).sum())
            counters['censored_numeric_0_48h']+=int((inside & numeric & censored).sum())
            counters['missing_unit_numeric_0_48h']+=int((inside & numeric & unit.eq('')).sum())
            keep=inside & available.between(0,48,inclusive='both') & numeric & ~censored & lab.ne('') & unit.ne('')
            frame=pd.DataFrame({'stay_id':chunk.stay_id,'lab':lab,'unit':unit,'sample_h':sample,
                'available_h':available,'value':value,'row_id':pd.to_numeric(rowid,errors='raise')})[keep]
            for key, count in frame.groupby(['lab','unit'],observed=True).size().items(): vocab[key]+=int(count)
            counters['retained_numeric_rows']+=len(frame)
            if len(frame): writer.write_table(pa.Table.from_pandas(frame, schema=SCHEMA,preserve_index=False))
            size=(out/'events.parquet').stat().st_size
            state=dict(database=a.database,pilot=bool(a.max_rows),counters=dict(counters),elapsed_seconds=time.monotonic()-started,output_bytes=size)
            atomic(out/'progress.json',state)
            if (ci+1)%5==0: print(json.dumps(state),flush=True)
            if size>a.max_output_gib*2**30: raise RuntimeError('Bounded output cap reached; do not use incomplete data')
    report=dict(database=a.database,complete=True,source_scan_complete=not bool(a.max_rows),pilot=bool(a.max_rows),
        source=str(source),source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
        counters=dict(counters),rows_bound=a.max_rows,elapsed_seconds=time.monotonic()-started,
        output_bytes=(out/'events.parquet').stat().st_size,events_sha256=sha(out/'events.parquet'),
        development_ids=proofs,test_targets_loaded=False,cohort_outcomes_loaded=False,
        time_rule='measurement in [0,48]; available=max(measurement,store/revised); available<=48; missing available time excluded',
        unit_rule='exact lab identifier and normalized recorded unit; no inferred conversions; censored numeric values excluded',
        limitations=['Row-prefix pilot is for throughput/storage and schema auditing only'] if a.max_rows else [])
    atomic(out/'completion.json',report)
    pd.DataFrame([dict(lab=k[0],unit=k[1],count=v) for k,v in vocab.most_common()]).to_csv(out/'lab_units.csv',index=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
