"""Per-run metrics of the main experiment, computed from supplied predictions."""
from __future__ import annotations

import numpy as np

from .decoding import valid_mask


def prediction_metrics(predictions, labels, tuples, reference=None):
    """Return accuracy/invalid rates as fractions; optionally check Eq. (4) recovery."""
    predicted = np.asarray(predictions)
    truth = np.asarray(labels)
    if predicted.shape != truth.shape or truth.ndim != 2 or not len(truth):
        raise ValueError("Expected matching nonempty [N,K] predictions and labels")
    if not valid_mask(truth, tuples).all():
        raise ValueError("Ground-truth tuples must belong to the ontology")
    correct = (predicted == truth).all(axis=1)
    output = {"samples": len(truth), "exact_count": int(correct.sum()),
              "exact_accuracy": float(correct.mean()),
              "invalid_rate": float((~valid_mask(predicted, tuples)).mean()),
              "level_accuracy": (predicted == truth).mean(axis=0).tolist()}
    if reference is not None:
        base = np.asarray(reference)
        if base.shape != truth.shape:
            raise ValueError("Reference prediction shape differs from the labels")
        keep = valid_mask(base, tuples)
        if not np.array_equal(predicted[keep], base[keep]):
            raise ValueError("A decoder changed an already-valid prediction")
        invalid_count = int((~keep).sum())
        recovered = int((correct & ~keep).sum())
        gain_count = int(correct.sum()) - int((base == truth).all(axis=1).sum())
        if gain_count != recovered:
            raise AssertionError("Gain is inconsistent with invalid-prediction recovery")
        output.update(reference_invalid_count=invalid_count, recovered_count=recovered,
                      recovery_rate=recovered / invalid_count if invalid_count else None)
    return output
