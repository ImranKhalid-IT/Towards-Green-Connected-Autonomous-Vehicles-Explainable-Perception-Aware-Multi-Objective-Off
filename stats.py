"""Statistics of Eq. (31) and Supplementary Section S9."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats as st


def relative_gain(y_drl: float, y_prop: float) -> float:
    """Delta^rel = (Y_DRL - Y_prop) / Y_DRL * 100 (positive = better for lower-is-better metrics)."""
    return (y_drl - y_prop) / y_drl * 100.0


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """d_C = (mean_1 - mean_2) / sqrt((SD_1^2 + SD_2^2) / 2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("inf")


def paired_test(prop: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired runs plus two kinds of 95% interval.

    ``ci_normal`` reproduces the interval used in the paper (mean +- SD of each policy,
    treated as independent); ``ci_paired`` is the t-interval of the paired differences,
    which the paper lists as future work and which run-level data make possible.
    """
    prop, base = np.asarray(prop, float), np.asarray(base, float)
    n = len(prop)
    diff = prop - base
    try:
        p = float(st.wilcoxon(prop, base).pvalue)
    except ValueError:
        p = float("nan")
    se_ind = np.sqrt(prop.var(ddof=1) / n + base.var(ddof=1) / n)
    md = float(diff.mean())
    se_p = diff.std(ddof=1) / np.sqrt(n)
    tcrit = st.t.ppf(0.975, n - 1)
    return {"mean_diff": md, "rel_change_pct": md / base.mean() * 100.0,
            "ci_normal_lo": md - 1.96 * se_ind, "ci_normal_hi": md + 1.96 * se_ind,
            "ci_paired_lo": md - tcrit * se_p, "ci_paired_hi": md + tcrit * se_p,
            "p_wilcoxon": p, "cohens_d": abs(cohens_d(prop, base))}
