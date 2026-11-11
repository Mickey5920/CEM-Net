"""Exact event sufficient statistics and source/time retention ledgers.

Deduplication belongs upstream: these routines count every supplied event.
Inputs use the PKL's zero-based type IDs, not the padded loader IDs.
"""
import numpy as np


class EventStatistics:
    def __init__(self, num_types, source_names, window, decay=12.):
        self.num_types = int(num_types)
        self.source_names = tuple(source_names)
        self.window, self.decay = float(window), float(decay)
        if self.window <= 0 or self.decay <= 0:
            raise ValueError('Positive window and decay required')
        self.counts = np.zeros(num_types, dtype=np.int64)
        self.last = np.zeros(num_types, dtype=np.float64)
        self.decayed = np.zeros(num_types, dtype=np.float64)
        self.sources = np.zeros(len(source_names), dtype=np.int64)
        self.n, self.time_sum, self.time_max = 0, 0., 0.

    def add(self, types, times, sources):
        ids = np.asarray(types, dtype=np.int64)
        t = np.asarray(times, dtype=np.float32).astype(np.float64)
        src = np.asarray(sources, dtype=np.int64)
        if ids.ndim != 1 or ids.shape != t.shape or ids.shape != src.shape:
            raise ValueError('Aligned one-dimensional event arrays required')
        if (not np.isfinite(t).all() or np.any((t < 0) | (t > self.window))
                or np.any((ids < 0) | (ids >= self.num_types))
                or np.any((src < 0) | (src >= len(self.source_names)))):
            raise ValueError('Event outside schema or landmark')
        np.add.at(self.counts, ids, 1)
        np.maximum.at(self.last, ids, t)
        np.add.at(self.decayed, ids, np.exp(-(self.window-t)/self.decay))
        np.add.at(self.sources, src, 1)
        self.n += len(t)
        self.time_sum += t.sum()
        self.time_max = max(self.time_max, float(t.max(initial=0)))
        return self

    def merge(self, other):
        if (self.num_types, self.source_names, self.window, self.decay) != (other.num_types, other.source_names, other.window, other.decay):
            raise ValueError('Cannot merge different reference windows or schemas')
        self.counts += other.counts
        self.last = np.maximum(self.last, other.last)
        self.decayed += other.decayed
        self.sources += other.sources
        self.n += other.n
        self.time_sum += other.time_sum
        self.time_max = max(self.time_max, other.time_max)
        return self

    def features(self):
        # Match historical padding semantics even for an empty event record.
        length = max(self.n, 1)
        return np.concatenate((np.log1p(self.counts), self.counts > 0,
            self.last/self.window, self.decayed, np.log1p(self.sources),
            self.sources/max(int(self.sources.sum()), 1),
            [length/512., self.time_max/self.window,
             self.time_sum/length/self.window, length/max(self.time_max, 1.)])).astype(np.float32)


def retention_ledger(times, sources, retained_indices, n_sources, window, bins=4, decay=12.):
    """Coupled source/bin budgets, not independent type-wise missing counts.

    Bounds describe aggregate discarded decay mass in each source/bin.
    All potential per-type allocations must share that bin's count budget.
    """
    t = np.asarray(times, dtype=np.float32).astype(np.float64)
    s = np.asarray(sources, dtype=np.int64)
    kept = np.asarray(retained_indices, dtype=np.int64)
    if window <= 0 or bins < 1 or decay <= 0 or s.shape != t.shape or t.ndim != 1:
        raise ValueError('Invalid ledger schema')
    if not np.isfinite(t).all() or np.any((t < 0) | (t > window)) or np.any((s < 0) | (s >= n_sources)):
        raise ValueError('Invalid source or future event')
    if len(np.unique(kept)) != len(kept) or np.any((kept < 0) | (kept >= len(t))):
        raise ValueError('Retention must be a subset of distinct raw events')
    b = np.minimum((t/window*bins).astype(int), bins-1)
    before = np.zeros((n_sources, bins), dtype=np.int64)
    after = np.zeros_like(before)
    np.add.at(before, (s, b), 1)
    np.add.at(after, (s[kept], b[kept]), 1)
    dropped = before-after
    edges = np.linspace(0, window, bins+1)
    return dict(before=before, after=after, discarded=dropped,
        decay_lower=dropped*np.exp(-(window-edges[:-1])/decay),
        decay_upper=dropped*np.exp(-(window-edges[1:])/decay))
