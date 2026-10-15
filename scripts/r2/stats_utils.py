"""Statistics used in the R2 analyses (numpy/scipy)."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def cluster_bootstrap(cluster_counts: np.ndarray, metric: Callable[[np.ndarray], float],
                      B: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Percentile CI by resampling clusters with replacement.

    ``cluster_counts`` has shape (n_clusters, k): per-cluster sufficient
    statistics (e.g. tp, fn, fp, tn); ``metric`` maps a summed count vector to
    the statistic.
    """
    rng = np.random.default_rng(seed)
    n = cluster_counts.shape[0]
    idx = rng.integers(0, n, size=(B, n))
    sums = cluster_counts[idx].sum(axis=1)  # (B, k)
    vals = np.array([metric(s) for s in sums])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def prf(tp: float, fn: float, fp: float, tn: float) -> Dict[str, float]:
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": 1 - rec if rec == rec else float("nan")}
