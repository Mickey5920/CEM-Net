"""Fixed-width clinical temporal evidence construction at a 24-hour landmark."""
import numpy as np

TAUS = np.asarray([1.0, 6.0, 24.0], dtype=np.float64)


class EventStatistics:
    """Exact per-type and per-source event summaries over [0, window]."""

    def __init__(self, num_types: int, source_names, window: float = 24.0, decay: float = 12.0):
        self.num_types = int(num_types)
        self.source_names = tuple(source_names)
        self.window = float(window)
        self.decay = float(decay)
        if self.window <= 0 or self.decay <= 0:
            raise ValueError("Positive window and decay required.")
        self.counts = np.zeros(self.num_types, dtype=np.int64)
        self.last = np.zeros(self.num_types, dtype=np.float64)
        self.decayed = np.zeros(self.num_types, dtype=np.float64)
        self.sources = np.zeros(len(self.source_names), dtype=np.int64)
        self.n = 0
        self.time_sum = 0.0
        self.time_max = 0.0

    def add(self, type_ids, times, source_ids):
        ids = np.asarray(type_ids, dtype=np.int64)
        t = np.asarray(times, dtype=np.float64)
        src = np.asarray(source_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.shape != t.shape or ids.shape != src.shape:
            raise ValueError("Aligned 1-D arrays required.")
        if len(ids) == 0:
            return self
        if (
            not np.isfinite(t).all()
            or np.any((t < 0.0) | (t > self.window))
            or np.any((ids < 0) | (ids >= self.num_types))
            or np.any((src < 0) | (src >= len(self.source_names)))
        ):
            raise ValueError("Event outside admissibility window or schema.")
        np.add.at(self.counts, ids, 1)
        np.maximum.at(self.last, ids, t)
        np.add.at(self.decayed, ids, np.exp(-(self.window - t) / self.decay))
        np.add.at(self.sources, src, 1)
        self.n += len(t)
        self.time_sum += float(t.sum())
        self.time_max = max(self.time_max, float(t.max()))
        return self

    def features(self) -> np.ndarray:
        """Return fixed-width vector of length 4*V + 2*S + 4."""
        length = max(self.n, 1)
        return np.concatenate((
            np.log1p(self.counts),
            (self.counts > 0).astype(np.float64),
            self.last / self.window,
            self.decayed,
            np.log1p(self.sources),
            self.sources / max(int(self.sources.sum()), 1),
            np.array([
                length / 512.0,
                self.time_max / self.window,
                self.time_sum / length / self.window,
                length / max(self.time_max, 1.0),
            ], dtype=np.float64),
        )).astype(np.float32)


def lab_basis(z: np.ndarray) -> np.ndarray:
    """Five-component basis b(z) = [1, z, z^2, (z-1)_+, (-z-1)_+]."""
    z = np.asarray(z, dtype=np.float64)
    return np.stack([
        np.ones_like(z),
        z,
        z * z,
        np.maximum(z - 1.0, 0.0),
        np.maximum(-z - 1.0, 0.0),
    ], axis=-1)


def simple_lab_summary(times, values, sample_times=None, window: float = 24.0) -> np.ndarray:
    """Return 11 observation and value summaries for one standardized laboratory key."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    s = t if sample_times is None else np.asarray(sample_times, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(11, dtype=np.float32)
    dt = t - t.mean()
    den = float(np.dot(dt, dt))
    slope = float(np.dot(dt, v - v.mean()) / den) * window if den > 1e-8 else 0.0
    return np.asarray([
        1.0,
        np.log1p(len(t)),
        t[0] / window,
        (window - t[-1]) / window,
        (window - s[-1]) / window,
        v.mean(),
        v.min(),
        v.max(),
        v[-1],
        v[-1] - v[0],
        slope,
    ], dtype=np.float32)


def duration_lab_summary(times, values, window: float = 24.0, taus=TAUS) -> np.ndarray:
    """Exact closed-form recorded-state duration integrals (15 coordinates = 3 taus x 5 bases)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(len(taus) * 5, dtype=np.float32)
    if not (t.ndim == 1 and t.shape == v.shape and np.all(np.diff(t) > 0)):
        raise ValueError("Strictly increasing 1-D timestamps required.")
    if not (np.all((t >= 0.0) & (t <= window)) and np.isfinite(v).all()):
        raise ValueError("Invalid timestamps or values.")
    end = np.r_[t[1:], window]
    mass = np.exp(-(window - end[:, None]) / taus) - np.exp(-(window - t[:, None]) / taus)
    normal = 1.0 - np.exp(-window / taus)
    return ((mass.T @ lab_basis(v)) / normal[:, None]).astype(np.float32).ravel()


def observation_weighted_summary(times, values, window: float = 24.0, taus=TAUS) -> np.ndarray:
    """Coverage-matched observation-point exponential control (15 coordinates)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(len(taus) * 5, dtype=np.float32)
    w = np.exp(-(window - t[:, None]) / taus)
    cov = (1.0 - np.exp(-(window - t[0]) / taus)) / (1.0 - np.exp(-window / taus))
    avg = (w.T @ lab_basis(v)) / np.maximum(w.sum(axis=0)[:, None], 1e-12)
    return (cov[:, None] * avg).astype(np.float32).ravel()
