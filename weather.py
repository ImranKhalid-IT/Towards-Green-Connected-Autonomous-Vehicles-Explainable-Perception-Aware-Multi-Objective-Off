"""Task release and weather process: Eqs. (1)-(2)."""
from __future__ import annotations

import numpy as np

from .config import SimConfig


def release_times(n: int, f_s: float, deadline: float):
    """Eq. (1): t_i = (i-1)/f_s,  t_i^dl = t_i + D."""
    t = np.arange(n) / f_s
    return t, t + deadline


def weather_sequence(cfg: SimConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Eq. (2): w_i ~ Cat(varpi) with n_w = N varpi_w.

    With ``exact_weather_counts`` the counts n_w are exact (26,100 / 2,400 / 900 / 600
    for N = 30,000).  Tasks are grouped into segments of geometric length
    (mean ``weather_mean_segment``) so that weather persists over consecutive frames
    as in a drive; the marginal distribution is still Cat(varpi).  Set
    ``weather_mean_segment = 1`` for i.i.d. weather.
    """
    shares = np.asarray(cfg.weather_shares, dtype=float)
    shares = shares / shares.sum()
    if not cfg.exact_weather_counts:
        counts = rng.multinomial(n, shares)
    else:
        counts = np.floor(shares * n).astype(int)
        counts[0] += n - counts.sum()
    segments = []
    for w, c in enumerate(counts):
        left = int(c)
        while left > 0:
            L = int(min(left, rng.geometric(1.0 / max(cfg.weather_mean_segment, 1))))
            segments.append(np.full(L, w, dtype=np.int64))
            left -= L
    order = rng.permutation(len(segments))
    return np.concatenate([segments[k] for k in order]) if segments else np.zeros(0, dtype=np.int64)


def weather_estimate(w: int, acc: float, rng: np.random.Generator, k: int = 4) -> np.ndarray:
    """Noisy on-vehicle weather probabilities p^w_i (state feature)."""
    p = np.full(k, (1.0 - acc) / (k - 1))
    guess = w if rng.random() < acc else int(rng.choice([j for j in range(k) if j != w]))
    p[guess] = acc
    return p
