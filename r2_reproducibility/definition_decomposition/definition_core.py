"""Pure fixed effect definitions; no fitting, feature selection or external tuning."""
import numpy as np

ZERO_TOLERANCE = 1e-12
DEFINITIONS = ('raw_mean', 'raw_median', 'clr_mean', 'clr_median')


def clr(raw, pseudocount):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or min(raw.shape) == 0 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('A nonempty finite nonnegative sample by feature matrix is required')
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('A fixed positive pseudocount is required')
    logs = np.log(raw + pseudocount)
    return logs - logs.mean(axis=1, keepdims=True)


def effect_grid(raw, y, pseudocount):
    raw = np.asarray(raw, dtype=float)
    z = clr(raw, pseudocount)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(raw) or set(y.tolist()) != {0, 1}:
        raise ValueError('Aligned binary labels with both classes are required')
    out = {}
    for scale, x in [('raw', raw), ('clr', z)]:
        for name, reducer in [('mean', np.mean), ('median', np.median)]:
            out[f'{scale}_{name}'] = reducer(x[y == 1], axis=0) - reducer(x[y == 0], axis=0)
    return out, (raw > 0).any(axis=0)


def concordance(effect, reference_sign, detected):
    effect, reference_sign = np.asarray(effect, float), np.asarray(reference_sign, float)
    detected = np.asarray(detected, bool)
    if (effect.shape != reference_sign.shape or effect.shape != detected.shape
            or not np.isfinite(effect).all() or not np.isfinite(reference_sign).all()):
        raise ValueError('Effects, reference signs and detection flags must align and be finite')
    return (np.abs(effect) > ZERO_TOLERANCE) & detected & (np.sign(effect) == np.sign(reference_sign)) & (reference_sign != 0)


def agreement(concordant):
    a = np.asarray(concordant, bool)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError('A nonempty study by feature concordance matrix is required')
    return a.mean(axis=0)

