import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class ClinicalEventDataset(torch.utils.data.Dataset):
    """Clinical prediction dataset for early EHR event histories."""

    def __init__(self, data, tasks=None, source_names=None):
        if isinstance(data, (str, Path)):
            data = load_clinical_split(data)
        self.samples = list(data)
        if tasks is None:
            tasks = infer_tasks(self.samples)
        if source_names is None:
            source_names = infer_source_names(self.samples)
        self.tasks = list(tasks)
        self.source_names = list(source_names)
        self.source_index = {name: index for index, name in enumerate(self.source_names)}
        self.source_feature_dim = len(self.source_names) * 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        labels = []
        label_mask = []
        for task in self.tasks:
            value = sample.get('labels', {}).get(task)
            if value is None:
                labels.append(0.0)
                label_mask.append(0.0)
            else:
                labels.append(float(value))
                label_mask.append(1.0)
        event_type = [int(value) + 1 for value in sample['type_event']]
        source_features = torch.zeros(self.source_feature_dim, dtype=torch.float32)
        source_counts = sample.get('source_counts', {}) or {}
        if self.source_names:
            total_source_events = max(sum(int(value) for value in source_counts.values()), 1)
            for source, value in source_counts.items():
                index = self.source_index.get(str(source))
                if index is None:
                    continue
                value = float(value)
                source_features[index] = float(np.log1p(value))
                source_features[len(self.source_names) + index] = value / float(total_source_events)
        return {
            'time_since_start': torch.tensor(sample['time_since_start'], dtype=torch.float32),
            'time_since_last_event': torch.tensor(sample['time_since_last_event'], dtype=torch.float32),
            'event_type': torch.tensor(event_type, dtype=torch.long),
            'source_features': source_features,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'label_mask': torch.tensor(label_mask, dtype=torch.float32),
            'length': int(sample.get('length', len(sample['type_event']))),
            'subject_id': int(sample['subject_id']),
            'hadm_id': int(sample['hadm_id']),
            'stay_id': int(sample['stay_id']),
        }


def infer_tasks(samples):
    tasks = []
    for sample in samples:
        for task in sample.get('labels', {}).keys():
            if task not in tasks:
                tasks.append(task)
    return tasks


def infer_source_names(samples):
    names = set()
    for sample in samples:
        names.update(str(key) for key in sample.get('source_counts', {}).keys())
    return sorted(names)


def load_clinical_split(path):
    path = Path(path)
    split_name = path.stem
    with open(path, 'rb') as handle:
        payload = pickle.load(handle, encoding='latin-1')
    if isinstance(payload, list):
        return payload
    if split_name in payload:
        return payload[split_name]
    for key in ['train', 'dev', 'test']:
        if key in payload:
            return payload[key]
    raise KeyError(f'Cannot find clinical samples in {path}')


def load_clinical_payload(path):
    path = Path(path)
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin-1')


def pad_1d_float(insts):
    max_len = max(len(inst) for inst in insts)
    batch = np.array([inst.tolist() + [Constants.PAD] * (max_len - len(inst)) for inst in insts], dtype=np.float32)
    return torch.tensor(batch, dtype=torch.float32)


def pad_1d_long(insts):
    max_len = max(len(inst) for inst in insts)
    batch = np.array([inst.tolist() + [Constants.PAD] * (max_len - len(inst)) for inst in insts], dtype=np.int64)
    return torch.tensor(batch, dtype=torch.long)


def clinical_collate_fn(insts):
    time_since_start = pad_1d_float([item['time_since_start'] for item in insts])
    time_since_last_event = pad_1d_float([item['time_since_last_event'] for item in insts])
    event_type = pad_1d_long([item['event_type'] for item in insts])
    labels = torch.stack([item['labels'] for item in insts], dim=0)
    label_mask = torch.stack([item['label_mask'] for item in insts], dim=0)
    lengths = torch.tensor([item['length'] for item in insts], dtype=torch.long)
    subject_id = torch.tensor([item['subject_id'] for item in insts], dtype=torch.long)
    hadm_id = torch.tensor([item['hadm_id'] for item in insts], dtype=torch.long)
    stay_id = torch.tensor([item['stay_id'] for item in insts], dtype=torch.long)
    source_features = torch.stack([item['source_features'] for item in insts], dim=0)
    return {
        'time_since_start': time_since_start,
        'time_since_last_event': time_since_last_event,
        'event_type': event_type,
        'source_features': source_features,
        'labels': labels,
        'label_mask': label_mask,
        'lengths': lengths,
        'subject_id': subject_id,
        'hadm_id': hadm_id,
        'stay_id': stay_id,
    }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_clinical_dataloader(
    data,
    tasks,
    batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    seed=None,
    source_names=None,
):
    dataset = ClinicalEventDataset(data, tasks=tasks, source_names=source_names)
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': clinical_collate_fn,
        'num_workers': int(num_workers),
        'pin_memory': bool(pin_memory),
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader_kwargs['generator'] = generator
        loader_kwargs['worker_init_fn'] = seed_worker
    if int(num_workers) > 0:
        loader_kwargs['persistent_workers'] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    return torch.utils.data.DataLoader(**loader_kwargs)
