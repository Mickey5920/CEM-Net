"""MIMIC-III 1.4: fixed 24h tasks and zero-update data preparation.

Reads gzip members directly from the existing ZIP. Never starts model training.
All feature definitions are reused from the current project. MIMIC-III labs
have specimen CHARTTIME only; their time is explicitly NOT result availability.
"""
import argparse
import collections
import contextlib
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tracehr_exact_statistics import EventStatistics
from scripts.tracehr_value_features import simple_summary, duration_summary

ARCHIVE = Path('F:/Data/mimic-iii-clinical-database-1.4.zip')
LABEL_CODE = Path('F:/Code/Mamba-Hawkes-Process/Experiments/scripts/build_mimiciv_exp2_cohort_labels_v2.py')
SOURCES = {
    'CHARTEVENTS': ('chart', 'charttime', True, True),
    'LABEVENTS': ('lab', 'charttime', False, False),
    'INPUTEVENTS_CV': ('input', 'charttime', True, True),
    'INPUTEVENTS_MV': ('input', 'starttime', True, True),
    'PROCEDUREEVENTS_MV': ('procedure_icu', 'starttime', True, True),
}
CAT_SCHEMA = pa.schema([('stay_id', pa.int64()), ('time', pa.float64()),
                       ('token', pa.string()), ('source', pa.string())])
NUM_SCHEMA = pa.schema([('stay_id', pa.int64()), ('lab', pa.string()),
                       ('unit', pa.string()), ('sample_h', pa.float64()),
                       ('available_h', pa.float64()), ('value', pa.float64())])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def emit(out, **value):
    value['rss_gib'] = psutil.Process().memory_info().rss / 2**30
    save(out/'progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def member(z, name):
    hits = [n for n in z.namelist() if n.upper().endswith('/'+name+'.CSV.GZ')]
    assert len(hits) == 1, (name, len(hits))
    return hits[0]


@contextlib.contextmanager
def table_stream(z, name):
    with z.open(member(z, name)) as raw, gzip.GzipFile(fileobj=raw) as f:
        yield f


def small_table(z, name):
    with table_stream(z, name) as f: frame = pd.read_csv(f)
    frame.columns = frame.columns.str.lower()
    return frame


def parse_time(s):
    return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')


def label_module():
    sys.path.insert(0, str(LABEL_CODE.parent))
    spec = importlib.util.spec_from_file_location('mimic_existing_labels', LABEL_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(out):
    assert not (out/'cohort.parquet').exists(), 'Do not overwrite a cohort'
    started = time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as z:
        icu, admissions, patients = [small_table(z, t) for t in ('ICUSTAYS','ADMISSIONS','PATIENTS')]
        items, labs = small_table(z, 'D_ITEMS'), small_table(z, 'D_LABITEMS')
        items.to_parquet(out/'item_dictionary.parquet', index=False)
        labs.to_parquet(out/'lab_dictionary.parquet', index=False)
    icu = icu.rename(columns={'icustay_id':'stay_id'})
    assert icu.stay_id.is_unique and admissions.hadm_id.is_unique and patients.subject_id.is_unique
    # MIMIC-III serializes its date-only DOD with a midnight timestamp. The
    # unchanged label implementation expects YYYY-MM-DD; normalize the format,
    # not the clinical meaning or the exclusion rule.
    parsed_dod = parse_time(patients.dod)
    assert parsed_dod.notna().sum() == patients.dod.notna().sum()
    patients['dod'] = parsed_dod.dt.strftime('%Y-%m-%d')
    lm = label_module()
    labels = lm.build_cohort(icu, admissions, patients, [24.], -1, 42)
    labels = labels.merge(icu[['stay_id','dbsource']], on='stay_id', validate='one_to_one')
    labels = labels.merge(patients[['subject_id','dob']], on='subject_id', validate='many_to_one')
    dob = parse_time(labels.dob)
    # Calendar arithmetic avoids pandas' ~292-year timedelta overflow in shifted ages >89.
    age = (labels.intime.dt.year-dob.dt.year).astype(float)
    birthday_not_yet = (labels.intime.dt.month < dob.dt.month) | ((labels.intime.dt.month == dob.dt.month) & (labels.intime.dt.day < dob.dt.day))
    age -= birthday_not_yet.astype(int)
    labels['adult'] = dob.notna() & (age >= 18)
    labels['age_shifted_over89'] = age >= 200
    stage = dict(raw_stays=len(labels), raw_subjects=int(labels.subject_id.nunique()))
    active = pd.Series(True, index=labels.index)
    for name, criterion in (
        ('adult', labels.adult),
        ('valid_at_least_24h', labels.eligible_24h),
        ('alive_at_24h', labels.alive_at_24h_landmark),
        ('precise_common_labels', labels.icu_mortality_after_24h.notna() & labels.icu_los_gt_7d_after_24h_eligible.notna())):
        before = int(active.sum()); active &= criterion
        stage[name] = dict(retained=int(active.sum()), removed=before-int(active.sum()))
    final = labels.loc[active].copy()
    final['mortality'] = final.icu_mortality_after_24h.astype('int8')
    final['los'] = final.icu_los_gt_7d_after_24h_eligible.astype('int8')
    # Same string-ID ordering and RNG as the existing patient splitter.
    subjects = sorted(final.subject_id.astype(str).unique())
    random.Random(42).shuffle(subjects)
    ntr, ndv = int(len(subjects)*.7), int(len(subjects)*.1)
    assignments = {p:s for s,ps in [('train',subjects[:ntr]),('dev',subjects[ntr:ntr+ndv]),('test',subjects[ntr+ndv:])] for p in ps}
    final['split'] = final.subject_id.astype(str).map(assignments)
    final = final.sort_values(['subject_id','intime','stay_id'], kind='stable').reset_index(drop=True)
    assert final.stay_id.is_unique and final.groupby('subject_id').split.nunique().max() == 1
    final.to_parquet(out/'cohort.parquet', index=False)
    labels[['subject_id','hadm_id','stay_id','adult','eligible_24h','alive_at_24h_landmark','uncertain_icu_death_time']].to_parquet(out/'cohort_flow.parquet', index=False)
    summary = []
    for (split, source), g in final.groupby(['split','dbsource'], dropna=False):
        summary.append(dict(split=split, system=str(source), stays=len(g), patients=int(g.subject_id.nunique()),
                            mortality_positive=int(g.mortality.sum()), los_positive=int(g.los.sum())))
    # Label boundary fixtures call the exact existing implementation.
    start = pd.Timestamp('2100-01-01')
    fixture = pd.DataFrame(dict(subject_id=range(8), hadm_id=range(8), stay_id=range(8),
        intime=[start]*8, outtime=[start+pd.Timedelta(hours=h) for h in (24,24,168,169,48,48,48,48)],
        admittime=[start]*8, dischtime=[start+pd.Timedelta(days=30)]*8,
        deathtime=[pd.NaT,start+pd.Timedelta(hours=24),pd.NaT,pd.NaT,start+pd.Timedelta(hours=25),start+pd.Timedelta(hours=49),pd.NaT,pd.NaT],
        dod=[pd.NaT]*6+[start+pd.Timedelta(days=1),start+pd.Timedelta(days=5)]))
    fixture['icu_los_hours'] = (fixture.outtime-fixture.intime).dt.total_seconds()/3600
    fixture['hospital_los_hours'] = (fixture.dischtime-fixture.admittime).dt.total_seconds()/3600
    result = lm.add_window_labels(fixture,24.)
    actual = [[None if pd.isna(v) else int(v) for v in result[c]] for c in ('icu_mortality_after_24h','icu_los_gt_7d_after_24h_eligible')]
    assert actual == [[0,None,0,0,1,0,None,0],[0,None,0,1,0,0,None,0]], actual
    record = dict(complete=True, dataset='MIMIC-III 1.4', archive=dict(path=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,mtime_ns=ARCHIVE.stat().st_mtime_ns),
        reused_archive_verification='Prior full member SHA256/CRC verification; unchanged archive size and mtime',
        source_label_code=str(LABEL_CODE), source_label_sha256=sha(LABEL_CODE), flow=stage,
        stays=len(final), patients=int(final.subject_id.nunique()), admissions=int(final.hadm_id.nunique()),
        over89_shifted_stays=int(final.age_shifted_over89.sum()), raw_systems=icu.dbsource.value_counts().to_dict(),
        split_seed=42, split_ratios=[.7,.1,.2], split_system_counts=summary,
        mortality_rule='24h < DEATHTIME-INTIME <= OUTTIME-INTIME; alive at 24h; date-only ICU death uncertainty excluded',
        los_rule='total OUTTIME-INTIME >168h in the identical mortality cohort',
        label_boundary_checks=8, patient_split_overlap=0, optimizer_updates=0,
        seconds=time.monotonic()-started, cohort_sha256=sha(out/'cohort.parquet'))
    save(out/'cohort-summary.json',record)
    emit(out, stage='cohort_complete', stays=len(final), patients=record['patients'], flow=stage)


def extract(out, row_bound, cap_gib):
    folder = out/('pilot' if row_bound else 'events')
    assert not folder.exists(), 'Do not overwrite an extraction or duplicate its writer'
    folder.mkdir()
    info = pd.read_parquet(out/'cohort.parquet', columns=['subject_id','hadm_id','stay_id','intime','dbsource'])
    selected = set(info.stay_id)
    audit = {}; peak = 0.; started = time.monotonic()
    # Bucket by stay, so cross-chunk duplicate removal is exact in the next stage.
    with contextlib.ExitStack() as stack:
        z = stack.enter_context(zipfile.ZipFile(ARCHIVE))
        cats = [stack.enter_context(pq.ParquetWriter(folder/f'cat_{i:02}.parquet', CAT_SCHEMA, compression='zstd')) for i in range(32)]
        nums = stack.enter_context(pq.ParquetWriter(folder/'numeric.parquet', NUM_SCHEMA, compression='zstd'))
        for table,(source, timecol, store, has_stay) in SOURCES.items():
            tick=time.monotonic(); stat=collections.Counter(); audit[table]=stat
            columns=['SUBJECT_ID','HADM_ID','ITEMID',timecol.upper()]+(['ICUSTAY_ID'] if has_stay else [])+(['STORETIME'] if store else ['VALUE','VALUENUM','VALUEUOM','FLAG'])
            types={k:'Int64' for k in ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID'] if k in columns}
            types.update({k:'str' for k in columns if k not in types})
            with table_stream(z, table) as f:
                for ci,c in enumerate(pd.read_csv(f,usecols=columns,dtype=types,chunksize=250000,nrows=row_bound or None)):
                    c.columns=c.columns.str.lower();c=c.rename(columns={'icustay_id':'stay_id'})
                    stat['scanned']+=len(c)
                    c=c[c.stay_id.isin(selected)] if has_stay else c[c.hadm_id.isin(info.hadm_id)]
                    keys=['subject_id','hadm_id']+(['stay_id'] if has_stay else [])
                    c=c.merge(info,on=keys,how='inner',validate='many_to_one' if has_stay else 'many_to_many')
                    stat['cohort_join_candidates']+=len(c)
                    sample=(parse_time(c[timecol])-c.intime).dt.total_seconds()/3600
                    inside=sample.between(0,24,inclusive='both')
                    stat['measured_in_24h']+=int(inside.sum())
                    if store:
                        entry=(parse_time(c.storetime)-c.intime).dt.total_seconds()/3600
                        stat['store_missing_in_window']+=int((inside & entry.isna()).sum())
                        stat['stored_after_24h']+=int((inside & entry.gt(24)).sum())
                        stat['store_before_measurement']+=int((inside & entry.lt(sample)).sum())
                        event=entry.where(entry.isna(),np.maximum(sample,entry))
                    else:
                        event=sample
                        stat['result_availability_unknown_in_window']+=int(inside.sum())
                    keep=inside & event.between(0,24,inclusive='both') & c.itemid.notna()
                    for system,g in c[keep].groupby('dbsource'):
                        stat['retained_'+str(system)]+=len(g)
                    token=source+'::'+c.itemid.astype(str)
                    if source=='lab': token=token+'::'+c.flag.fillna('normal').str.strip().str.lower().str.replace(' ','_',regex=False).replace('','normal')
                    cat=pd.DataFrame(dict(stay_id=c.stay_id,time=event,token=token,source=source))[keep]
                    cat=cat.drop_duplicates(['stay_id','time','token'])
                    stat['retained_chunk_dedup']+=len(cat)
                    for bucket,g in cat.groupby(cat.stay_id.astype('int64')%32,sort=False):
                        cats[int(bucket)].write_table(pa.Table.from_pandas(g,schema=CAT_SCHEMA,preserve_index=False))
                    if source=='lab':
                        value=pd.to_numeric(c.valuenum,errors='coerce')
                        unit=c.valueuom.fillna('').str.strip().str.lower().str.replace('µ','u',regex=False)
                        censored=c.value.fillna('').str.contains(r'[<>≤≥]',regex=True,na=False)
                        numeric=keep & np.isfinite(value) & ~censored & unit.ne('')
                        stat['numeric_missing_unit']+=int((keep & np.isfinite(value) & unit.eq('')).sum())
                        stat['censored_numeric']+=int((keep & np.isfinite(value) & censored).sum())
                        frame=pd.DataFrame(dict(stay_id=c.stay_id,lab=c.itemid.astype(str),unit=unit,sample_h=sample,available_h=event,value=value))[numeric]
                        stat['retained_numeric']+=len(frame)
                        if len(frame): nums.write_table(pa.Table.from_pandas(frame,schema=NUM_SCHEMA,preserve_index=False))
                    peak=max(peak,psutil.Process().memory_info().rss/2**30)
                    if ci%4==0:
                        size=sum(p.stat().st_size for p in folder.glob('*.parquet'))
                        assert size<=cap_gib*2**30,'Output storage cap reached'
                        emit(out,stage='extract_pilot' if row_bound else 'extract_full',table=table,scanned=stat['scanned'],retained=stat['retained_chunk_dedup'],seconds=round(time.monotonic()-started,1),output_gib=size/2**30)
                stat['seconds']=round(time.monotonic()-tick,3)
                stat['gzip_member_bytes']=z.getinfo(member(z,table)).file_size
    size=sum(p.stat().st_size for p in folder.glob('*.parquet'))
    save(folder/'completion.json',dict(complete=True,pilot=bool(row_bound),rows_bound=row_bound,audit={k:dict(v) for k,v in audit.items()},
        bytes=size,peak_rss_gib=peak,seconds=time.monotonic()-started,archive_bytes=ARCHIVE.stat().st_size,
        cohort_sha256=sha(out/'cohort.parquet'),source_sha256=sha(__file__),optimizer_updates=0,
        outcome_fields_loaded=False,lab_time_semantics='specimen CHARTTIME; result availability not recorded'))
    emit(out,stage='pilot_complete' if row_bound else 'extraction_complete',seconds=time.monotonic()-started,output_gib=size/2**30,peak_rss_gib=peak)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cohort','pilot','extract'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cap-gib',type=float,default=8.)
    a=p.parse_args();out=a.output.resolve();assert out.is_relative_to(ROOT) and out.drive.upper()=='F:'
    assert out==Path(os.environ['PROJECT_RUN_DIR']).resolve();out.mkdir(parents=True,exist_ok=True)
    if a.mode=='cohort':cohort(out)
    else:extract(out,1000000 if a.mode=='pilot' else 0,a.cap_gib)


if __name__=='__main__':main()
