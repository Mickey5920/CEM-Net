import os
import platform
from pathlib import Path

import pandas as pd


DEFAULT_MIMIC_ROOT_CANDIDATES = [
    Path('/home/data/MIMIC/mimic-iv-3.1'),
    Path('/home/data/mimic-iv-3.1'),
    Path('/data/mimic-iv-3.1'),
    Path('/data/MIMIC/mimic-iv-3.1'),
    Path('/mnt/data/mimic-iv-3.1'),
    Path('/mnt/f/Data/mimic-iv-3.1'),
    Path(r'F:\Data\mimic-iv-3.1'),
]

SUPPORTED_EXTENSIONS = ['csv.gz', 'parquet', 'csv']
UNZIPPED_DIR_NAMES = {'hosp_unzipped', 'icu_unzipped'}


def table_format_policy_text():
    return (
        'prefer official compressed .csv.gz tables; use parquet only when no .csv.gz exists; '
        'fall back to raw .csv only when compressed/parquet versions are unavailable'
    )


def detect_mimic_root(explicit_root=None):
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.extend(DEFAULT_MIMIC_ROOT_CANDIDATES)
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        'Could not find a valid MIMIC-IV root. Checked: '
        + ', '.join(str(candidate) for candidate in candidates)
    )


def table_candidates(root, group_name, table_name):
    root = Path(root)
    exact = []
    for extension in SUPPORTED_EXTENSIONS:
        exact.append(root / group_name / f'{table_name}.{extension}')

    recursive = []
    for extension in SUPPORTED_EXTENSIONS:
        recursive.extend(root.rglob(f'{table_name}.{extension}'))

    def is_group_match(path):
        if len(path.parents) < 1:
            return False
        if path.parent.name in UNZIPPED_DIR_NAMES:
            return False
        return path.parent.name == group_name

    filtered_recursive = [path for path in recursive if is_group_match(path)]
    all_candidates = []
    for path in exact + filtered_recursive:
        if path.exists():
            all_candidates.append(path)

    unique_candidates = []
    seen = set()
    for path in all_candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(path)
    return unique_candidates


def choose_table_path(root, group_name, table_name):
    candidates = table_candidates(root, group_name, table_name)
    if not candidates:
        return None

    def score(path):
        relative = path.relative_to(root)
        extension = '.'.join(path.name.split('.')[1:])
        try:
            ext_rank = SUPPORTED_EXTENSIONS.index(extension)
        except ValueError:
            ext_rank = len(SUPPORTED_EXTENSIONS)
        # Prefer the compact official tables even if a stray uncompressed CSV
        # exists closer to the provided root from a previous manual gunzip.
        return (ext_rank, len(relative.parts), len(str(relative)))

    return sorted(candidates, key=score)[0]


def describe_table(root, group_name, table_name):
    candidates = table_candidates(root, group_name, table_name)
    path = choose_table_path(root, group_name, table_name)
    if path is None:
        return {
            'found': False,
            'path': None,
            'format': None,
            'size_bytes': None,
            'size_human': None,
            'candidate_count': 0,
            'alternate_candidates': [],
            'uncompressed_duplicate_count': 0,
            'uncompressed_duplicate_size_bytes': 0,
            'selection_note': f'not found; policy={table_format_policy_text()}',
        }
    extension = '.'.join(path.name.split('.')[1:])
    size_bytes = path.stat().st_size
    alternate_candidates = []
    uncompressed_duplicate_count = 0
    uncompressed_duplicate_size_bytes = 0
    for candidate in candidates:
        candidate_extension = '.'.join(candidate.name.split('.')[1:])
        if candidate == path:
            continue
        candidate_size = candidate.stat().st_size
        alternate_candidates.append({
            'path': str(candidate),
            'format': candidate_extension,
            'size_bytes': candidate_size,
            'size_human': human_size(candidate_size),
        })
        if candidate_extension == 'csv':
            uncompressed_duplicate_count += 1
            uncompressed_duplicate_size_bytes += candidate_size

    selection_note = f'selected {extension}; policy={table_format_policy_text()}'
    if uncompressed_duplicate_count:
        selection_note += (
            f'; ignored {uncompressed_duplicate_count} uncompressed .csv duplicate(s) '
            f'totaling {human_size(uncompressed_duplicate_size_bytes)}'
        )
    return {
        'found': True,
        'path': path,
        'format': extension,
        'size_bytes': size_bytes,
        'size_human': human_size(size_bytes),
        'candidate_count': len(candidates),
        'alternate_candidates': alternate_candidates,
        'uncompressed_duplicate_count': uncompressed_duplicate_count,
        'uncompressed_duplicate_size_bytes': uncompressed_duplicate_size_bytes,
        'selection_note': selection_note,
    }


def load_table(root, group_name, table_name, usecols=None, chunksize=None):
    path = choose_table_path(root, group_name, table_name)
    if path is None:
        raise FileNotFoundError(f'Cannot find {group_name}/{table_name} under {root}')
    extension = '.'.join(path.name.split('.')[1:])
    if extension == 'parquet':
        if chunksize is None:
            return pd.read_parquet(path, columns=usecols)
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)

        def iterator():
            for batch in parquet_file.iter_batches(batch_size=chunksize, columns=usecols):
                yield batch.to_pandas()

        return iterator()
    return pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        compression='infer',
    )


def human_size(size_bytes):
    if size_bytes is None:
        return None
    if size_bytes == 0:
        return '0 B'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f'{size:.2f} {units[unit_index]}'


def detect_platform_label():
    if os.name == 'nt':
        return 'windows'
    return platform.system().lower()
