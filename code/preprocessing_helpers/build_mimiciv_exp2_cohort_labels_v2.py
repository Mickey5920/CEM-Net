import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mimiciv_utils import detect_mimic_root, describe_table, load_table, table_format_policy_text


BASE_COLUMNS = [
    'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'admittime', 'dischtime',
    'deathtime', 'dod', 'hospital_expire_flag', 'icu_los_hours', 'hospital_los_hours',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Build Experiment 2B post-landmark cohort labels.')
    parser.add_argument('--mimic_root', type=str, default=None)
    parser.add_argument('--unit', type=str, default='stay_id', choices=['stay_id'])
    parser.add_argument('--windows', nargs='+', type=float, default=[24.0, 48.0])
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_units', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--primary_death_source', type=str, default='deathtime', choices=['deathtime'])
    return parser.parse_args()


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'n/a'
    if isinstance(value, (int, bool)):
        return str(int(value))
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def markdown_table(columns, rows):
    return '\n'.join(
        ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        + ['| ' + ' | '.join(fmt(row.get(column)) for column in columns) + ' |' for row in rows]
    )


def parse_datetime_columns(frame, columns):
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NaT
        elif column == 'dod':
            frame[column] = pd.to_datetime(frame[column], errors='coerce', format='%Y-%m-%d')
        else:
            parsed = pd.to_datetime(frame[column], errors='coerce', format='%Y-%m-%d %H:%M:%S')
            if parsed.notna().sum() == 0 and frame[column].notna().any():
                parsed = pd.to_datetime(frame[column], errors='coerce')
            frame[column] = parsed
    return frame


def normalize_ids(frame, columns):
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').astype('Int64')
    return frame


def hours_between(end, start):
    return (end - start).dt.total_seconds() / 3600.0


def timestamp_in_interval(value, start, end, include_start=True):
    left = value >= start if include_start else value > start
    return value.notna() & start.notna() & end.notna() & left & (value <= end)


def date_in_interval(date_value, start, end):
    value = pd.to_datetime(date_value, errors='coerce').dt.normalize()
    start_date = pd.to_datetime(start, errors='coerce').dt.normalize()
    end_date = pd.to_datetime(end, errors='coerce').dt.normalize()
    return value.notna() & start_date.notna() & end_date.notna() & (value >= start_date) & (value <= end_date)


def serialize_table_description(desc):
    payload = dict(desc)
    if payload.get('path') is not None:
        payload['path'] = str(payload['path'])
    return payload


def load_raw_tables(mimic_root):
    return (
        load_table(mimic_root, 'icu', 'icustays'),
        load_table(mimic_root, 'hosp', 'admissions'),
        load_table(mimic_root, 'hosp', 'patients'),
    )


def add_window_labels(cohort, window):
    label = str(int(window)) if float(window).is_integer() else str(window).replace('.', 'p')
    landmark = cohort['intime'] + pd.to_timedelta(float(window), unit='h')
    eligible = (
        cohort['subject_id'].notna()
        & cohort['hadm_id'].notna()
        & cohort['stay_id'].notna()
        & cohort['intime'].notna()
        & cohort['outtime'].notna()
        & (cohort['icu_los_hours'] >= float(window))
        & (cohort['outtime'] >= landmark)
    )
    precise_death = cohort['deathtime']
    early_icu_death = eligible & timestamp_in_interval(precise_death, cohort['intime'], landmark, include_start=False)
    early_hospital_death = eligible & timestamp_in_interval(precise_death, cohort['admittime'], landmark, include_start=False)
    alive_at_landmark = eligible & ~(precise_death.notna() & (precise_death <= landmark))
    precise_icu_after = alive_at_landmark & (precise_death > landmark) & (precise_death <= cohort['outtime'])
    precise_hospital_after = alive_at_landmark & (precise_death > landmark) & (precise_death <= cohort['dischtime'])
    dod_icu_overlap = date_in_interval(cohort['dod'], cohort['intime'], cohort['outtime'])
    uncertain_icu = cohort['deathtime'].isna() & dod_icu_overlap
    primary_sample = alive_at_landmark & ~uncertain_icu

    cohort[f'eligible_{label}h'] = eligible.fillna(False)
    cohort[f'alive_at_{label}h_landmark'] = alive_at_landmark.fillna(False)
    cohort[f'excluded_early_icu_death_{label}h'] = early_icu_death.fillna(False)
    cohort[f'excluded_early_hospital_death_{label}h'] = early_hospital_death.fillna(False)
    cohort[f'icu_mortality_after_{label}h'] = pd.Series(pd.NA, index=cohort.index, dtype='Int64')
    cohort.loc[primary_sample, f'icu_mortality_after_{label}h'] = precise_icu_after.loc[primary_sample].astype('int8')
    cohort[f'icu_los_gt_7d_after_{label}h_eligible'] = pd.Series(pd.NA, index=cohort.index, dtype='Int64')
    cohort.loc[primary_sample, f'icu_los_gt_7d_after_{label}h_eligible'] = (cohort.loc[primary_sample, 'icu_los_hours'] > 168.0).astype('int8')
    cohort[f'in_hospital_mortality_after_{label}h_aux'] = pd.Series(pd.NA, index=cohort.index, dtype='Int64')
    cohort.loc[primary_sample, f'in_hospital_mortality_after_{label}h_aux'] = precise_hospital_after.loc[primary_sample].astype('int8')
    cohort[f'hospital_los_gt_7d_after_{label}h_aux'] = pd.Series(pd.NA, index=cohort.index, dtype='Int64')
    cohort.loc[primary_sample, f'hospital_los_gt_7d_after_{label}h_aux'] = (cohort.loc[primary_sample, 'hospital_los_hours'] > 168.0).astype('int8')
    return cohort


def build_cohort(icustays, admissions, patients, windows, max_units, seed):
    icustays = normalize_ids(parse_datetime_columns(icustays, ['intime', 'outtime']), ['subject_id', 'hadm_id', 'stay_id'])
    admissions = normalize_ids(parse_datetime_columns(admissions, ['admittime', 'dischtime', 'deathtime']), ['subject_id', 'hadm_id'])
    patients = normalize_ids(parse_datetime_columns(patients, ['dod']), ['subject_id'])
    required = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']
    missing = [column for column in required if column not in icustays.columns]
    if missing:
        raise KeyError(f'icustays is missing required columns: {missing}')

    cohort = icustays[required].dropna(subset=['subject_id', 'hadm_id', 'stay_id']).drop_duplicates('stay_id').copy()
    admission_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']
    if 'hospital_expire_flag' in admissions.columns:
        admission_cols.append('hospital_expire_flag')
    cohort = cohort.merge(admissions[admission_cols].drop_duplicates('hadm_id'), on=['subject_id', 'hadm_id'], how='left')
    cohort = cohort.merge(patients[['subject_id', 'dod']].drop_duplicates('subject_id'), on='subject_id', how='left')
    if 'hospital_expire_flag' not in cohort.columns:
        cohort['hospital_expire_flag'] = pd.NA

    cohort['icu_los_hours'] = hours_between(cohort['outtime'], cohort['intime'])
    cohort['hospital_los_hours'] = hours_between(cohort['dischtime'], cohort['admittime'])
    cohort['deathtime_available'] = cohort['deathtime'].notna()
    cohort['dod_available'] = cohort['dod'].notna()
    cohort['death_time_source'] = 'none'
    cohort.loc[cohort['deathtime_available'], 'death_time_source'] = 'admissions.deathtime'
    cohort.loc[~cohort['deathtime_available'] & cohort['dod_available'], 'death_time_source'] = 'patients.dod_date_only'
    cohort['uncertain_icu_death_time'] = (cohort['deathtime'].isna() & date_in_interval(cohort['dod'], cohort['intime'], cohort['outtime'])).fillna(False)

    for window in windows:
        cohort = add_window_labels(cohort, float(window))

    cohort = cohort.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').reset_index(drop=True)
    if max_units is not None and max_units > 0 and len(cohort) > max_units:
        cohort = cohort.sample(n=max_units, random_state=seed).sort_values(['subject_id', 'intime', 'stay_id']).reset_index(drop=True)
    return cohort


def label_count(frame, column):
    values = pd.to_numeric(frame[column], errors='coerce').dropna()
    return {'n_samples': int(len(values)), 'positives': int(values.sum()), 'pos_rate': None if len(values) == 0 else float(values.mean())}


def build_metadata(args, mimic_root, cohort):
    table_selection = {
        'icu/icustays': serialize_table_description(describe_table(mimic_root, 'icu', 'icustays')),
        'hosp/admissions': serialize_table_description(describe_table(mimic_root, 'hosp', 'admissions')),
        'hosp/patients': serialize_table_description(describe_table(mimic_root, 'hosp', 'patients')),
    }
    window_summaries = []
    for window in args.windows:
        label = str(int(window)) if float(window).is_integer() else str(window).replace('.', 'p')
        primary = label_count(cohort, f'icu_mortality_after_{label}h')
        aux = label_count(cohort, f'in_hospital_mortality_after_{label}h_aux')
        los = label_count(cohort, f'icu_los_gt_7d_after_{label}h_eligible')
        window_summaries.append({
            'window': float(window),
            'eligible': int(cohort[f'eligible_{label}h'].sum()),
            'alive_at_landmark': int(cohort[f'alive_at_{label}h_landmark'].sum()),
            'early_icu_deaths': int(cohort[f'excluded_early_icu_death_{label}h'].sum()),
            'post_landmark_icu_deaths': primary['positives'],
            'post_landmark_icu_pos_rate': primary['pos_rate'],
            'post_landmark_icu_n_samples': primary['n_samples'],
            'early_hospital_deaths': int(cohort[f'excluded_early_hospital_death_{label}h'].sum()),
            'post_landmark_hospital_deaths': aux['positives'],
            'post_landmark_hospital_pos_rate': aux['pos_rate'],
            'post_landmark_hospital_n_samples': aux['n_samples'],
            'icu_los_gt_7d': los,
        })
    return {
        'dataset': 'MIMIC-IV',
        'experiment': 'Experiment2B_ProtocolFix_V2',
        'unit': args.unit,
        'windows': [float(window) for window in args.windows],
        'primary_death_source': args.primary_death_source,
        'max_units_requested': args.max_units,
        'seed': args.seed,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'mimic_root': str(mimic_root),
        'table_format_policy': table_format_policy_text(),
        'table_selection': table_selection,
        'n_subjects': int(cohort['subject_id'].nunique()),
        'n_hadm_id': int(cohort['hadm_id'].nunique()),
        'n_stay_id': int(cohort['stay_id'].nunique()),
        'duplicated_stay_id_count': int(cohort['stay_id'].duplicated().sum()),
        'window_summaries': window_summaries,
        'death_time_source': {
            'deathtime available': int(cohort['deathtime_available'].sum()),
            'dod available': int(cohort['dod_available'].sum()),
            'both available': int((cohort['deathtime_available'] & cohort['dod_available']).sum()),
            'deathtime missing, dod available': int((~cohort['deathtime_available'] & cohort['dod_available']).sum()),
            'uncertain ICU death time': int(cohort['uncertain_icu_death_time'].sum()),
        },
        'label_rules': {
            'primary_mortality': 'alive at landmark, no date-only ICU death uncertainty, admissions.deathtime > landmark and <= ICU outtime',
            'early_death_exclusion': 'admissions.deathtime > ICU intime and <= landmark',
            'dod_handling': 'patients.dod is date-only uncertainty metadata and is not used as primary event time',
            'icu_los': 'ICU LOS > 168h among primary landmark-eligible samples',
            'hospital_auxiliary': 'in-hospital mortality after landmark is auxiliary only; formal hospital tasks require a hadm_id cohort',
        },
    }


def write_qc(path, metadata):
    cohort_row = [{
        'item': 'n_subjects', 'value': metadata['n_subjects'],
    }, {'item': 'n_hadm_id', 'value': metadata['n_hadm_id']},
        {'item': 'n_stay_id', 'value': metadata['n_stay_id']}]
    for summary in metadata['window_summaries']:
        label = str(int(summary['window']))
        cohort_row.append({'item': f'n_eligible_{label}h', 'value': summary['eligible']})
        cohort_row.append({'item': f'n_alive_at_{label}h', 'value': summary['alive_at_landmark']})
    mortality_rows = [
        {
            'window': row['window'],
            'eligible': row['eligible'],
            'alive_at_landmark': row['alive_at_landmark'],
            'early_icu_deaths': row['early_icu_deaths'],
            'post_landmark_icu_deaths': row['post_landmark_icu_deaths'],
            'pos_rate': row['post_landmark_icu_pos_rate'],
        }
        for row in metadata['window_summaries']
    ]
    aux_rows = [
        {
            'window': row['window'],
            'eligible': row['eligible'],
            'alive_at_landmark': row['alive_at_landmark'],
            'early_hospital_deaths': row['early_hospital_deaths'],
            'post_landmark_hospital_deaths': row['post_landmark_hospital_deaths'],
            'pos_rate': row['post_landmark_hospital_pos_rate'],
        }
        for row in metadata['window_summaries']
    ]
    los_rows = []
    for row in metadata['window_summaries']:
        los_rows.append({
            'window': row['window'],
            'label': f'icu_los_gt_7d_after_{int(row["window"])}h_eligible',
            **row['icu_los_gt_7d'],
        })
    death_rows = [{'source': key, 'count': value} for key, value in metadata['death_time_source'].items()]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# Experiment 2B Cohort Label V2 QC\n\n## Cohort Size\n\n')
        handle.write(markdown_table(['item', 'value'], cohort_row))
        handle.write('\n\n## Landmark Mortality\n\n')
        handle.write(markdown_table(['window', 'eligible', 'alive_at_landmark', 'early_icu_deaths', 'post_landmark_icu_deaths', 'pos_rate'], mortality_rows))
        handle.write('\n\n## Auxiliary Hospital Mortality\n\n')
        handle.write(markdown_table(['window', 'eligible', 'alive_at_landmark', 'early_hospital_deaths', 'post_landmark_hospital_deaths', 'pos_rate'], aux_rows))
        handle.write('\n\n## LOS Labels\n\n')
        handle.write(markdown_table(['window', 'label', 'n_samples', 'positives', 'pos_rate'], los_rows))
        handle.write('\n\n## Death Time Source\n\n')
        handle.write(markdown_table(['source', 'count'], death_rows))
        handle.write('\n\n## Recommendation\n\n')
        handle.write('* primary tasks retained: post-landmark ICU mortality and ICU LOS > 7d at stay_id level.\n')
        handle.write('* auxiliary tasks retained: post-landmark in-hospital mortality and hospital LOS, explicitly auxiliary only.\n')
        handle.write('* excluded samples: input-window deaths and date-only ICU death uncertainty are excluded from primary mortality labels.\n')
        handle.write('* pending hospital-level tasks: formal in-hospital mortality/readmission should use a later hadm_id cohort.\n')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mimic_root = detect_mimic_root(args.mimic_root)
    icustays, admissions, patients = load_raw_tables(mimic_root)
    cohort = build_cohort(icustays, admissions, patients, args.windows, args.max_units, args.seed)
    cohort_path = output_dir / 'cohort_labels_v2.parquet'
    cohort.to_parquet(cohort_path, index=False)
    metadata = build_metadata(args, mimic_root, cohort)
    save_json(output_dir / 'label_metadata_v2.json', metadata)
    write_qc(output_dir / 'cohort_qc_report_v2.md', metadata)
    print(f'n_subjects={metadata["n_subjects"]}')
    print(f'n_hadm_id={metadata["n_hadm_id"]}')
    print(f'n_stay_id={metadata["n_stay_id"]}')
    for row in metadata['window_summaries']:
        label = int(row['window'])
        print(f'n_eligible_{label}h={row["eligible"]}')
        print(f'n_alive_at_{label}h={row["alive_at_landmark"]}')
        print(f'icu_mortality_after_{label}h_positive_rate={fmt(row["post_landmark_icu_pos_rate"])}')
    print(f'cohort_labels={cohort_path}')
    print(f'output_dir={output_dir}')


if __name__ == '__main__':
    main()
