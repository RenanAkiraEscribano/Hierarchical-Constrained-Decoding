"""Independent, C2F, F2C, JO, joint-head and seven-candidate HCD decoding."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .config import ALPHAS


@dataclass(frozen=True)
class HCDParameters:
    temperatures: tuple[float, ...]
    alpha: float

    def __post_init__(self):
        if self.alpha not in ALPHAS:
            raise ValueError("Alpha must belong to the paper's seven-candidate grid")
        if not self.temperatures or any(not math.isfinite(t) or t <= 0 for t in self.temperatures):
            raise ValueError("Temperatures must be positive and finite")


def _inputs(logits, tuples):
    heads = [np.asarray(head, dtype=np.float32) for head in logits]
    valid = np.asarray(tuples, dtype=np.int64)
    if valid.ndim != 2 or not len(valid) or len(heads) != valid.shape[1]:
        raise ValueError("Expected one head per level and a nonempty [V,K] ontology")
    if any(head.ndim != 2 for head in heads):
        raise ValueError("Each head must have shape [N,C]")
    samples = len(heads[0])
    if not samples or any(len(head) != samples or not np.isfinite(head).all() for head in heads):
        raise ValueError("Logits must be finite and have equal, nonempty sample counts")
    for level, head in enumerate(heads):
        if valid[:, level].min() < 0 or valid[:, level].max() >= head.shape[1]:
            raise ValueError("Ontology label is outside its head dimension")
    return heads, valid


def independent(logits):
    return np.stack([np.asarray(head).argmax(axis=1) for head in logits], axis=1)


def valid_mask(predictions, tuples):
    predicted = np.asarray(predictions)
    valid = np.asarray(tuples)
    return (predicted[:, None, :] == valid[None, :, :]).all(axis=2).any(axis=1)


def _preserve(predicted, heads, tuples):
    base = independent(heads)
    keep = valid_mask(base, tuples)
    predicted[keep] = base[keep]
    return predicted


def c2f(logits, tuples):
    heads, valid = _inputs(logits, tuples)
    allowed = np.ones((len(heads[0]), len(valid)), dtype=bool)
    columns = []
    for level, head in enumerate(heads):
        candidates = np.where(allowed, head[:, valid[:, level]], -np.inf)
        chosen = valid[candidates.argmax(axis=1), level]
        columns.append(chosen)
        allowed &= valid[None, :, level] == chosen[:, None]
    return _preserve(np.stack(columns, axis=1), heads, valid)


def hcd(logits, tuples, parameters):
    """Eq. (3); alpha=0 denotes the lexicographic 0+ endpoint, never zeroing ancestors."""
    heads, valid = _inputs(logits, tuples)
    if len(parameters.temperatures) != len(heads):
        raise ValueError("Expected one temperature per level")
    fine = heads[-1][:, valid[:, -1]] / float(parameters.temperatures[-1])
    coarse = np.zeros_like(fine, dtype=np.float32)
    for level, head in enumerate(heads[:-1]):
        coarse += head[:, valid[:, level]] / float(parameters.temperatures[level])
    if parameters.alpha == 0.0:
        fine_argmax = heads[-1].argmax(axis=1)
        allowed = valid[None, :, -1] == fine_argmax[:, None]
        if not allowed.any(axis=1).all():
            raise ValueError("A predicted fine label has no valid tuple")
        chosen = np.where(allowed, coarse, -np.inf).argmax(axis=1)
    else:
        chosen = (fine + np.float32(parameters.alpha) * coarse).argmax(axis=1)
    return _preserve(valid[chosen].copy(), heads, valid)


def f2c(logits, tuples):
    return hcd(logits, tuples, HCDParameters((1.0,) * len(logits), 0.0))


def jo(logits, tuples):
    return hcd(logits, tuples, HCDParameters((1.0,) * len(logits), 1.0))


def joint_head(logits, tuples):
    scores = np.asarray(logits)
    valid = np.asarray(tuples, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(valid) or not np.isfinite(scores).all():
        raise ValueError("Expected finite [N,V] joint-head logits")
    return valid[scores.argmax(axis=1)]


def fit_temperature(logits, labels):
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or truth.ndim != 1 or len(scores) != len(truth) or not len(truth):
        raise ValueError("Expected [N,C] logits and [N] validation labels")
    if not np.isfinite(scores).all() or truth.min() < 0 or truth.max() >= scores.shape[1]:
        raise ValueError("Nonfinite logits or labels outside the class range")
    rows = np.arange(len(truth))

    def nll(log_t):
        scaled = scores * np.exp(-log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[rows, truth]))

    fitted = minimize_scalar(nll, bounds=(-np.log(20.0), np.log(20.0)), method="bounded",
                             options={"xatol": 1e-4, "maxiter": 100})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError("Validation temperature fitting failed")
    return float(np.exp(fitted.x))


def fit_hcd(logits, labels, tuples):
    """Fit marginal temperatures, then alpha, using validation labels only."""
    heads, valid = _inputs(logits, tuples)
    truth = np.asarray(labels, dtype=np.int64)
    if truth.shape != (len(heads[0]), len(heads)) or not valid_mask(truth, valid).all():
        raise ValueError("Validation truth must consist of complete valid tuples")
    temperatures = tuple(fit_temperature(head, truth[:, level]) for level, head in enumerate(heads))
    counts = {
        alpha: int((hcd(heads, valid, HCDParameters(temperatures, alpha)) == truth).all(axis=1).sum())
        for alpha in ALPHAS
    }
    alpha = min(ALPHAS, key=lambda value: (-counts[value],
                abs(math.log2(value)) if value > 0 else math.inf, value))
    return HCDParameters(temperatures, alpha)
