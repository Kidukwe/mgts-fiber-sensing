"""Spatial spectral features; the frequency unit is cycles per sample, not Hz."""

import numpy as np


def spatial_high_frequency_fraction(values, cutoff=.25):
    y = np.asarray(values, float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected contiguous finite spatial samples')
    if not 0 < cutoff <= .5:
        raise ValueError('Require a spatial cutoff in (0, 0.5] cycles/sample')
    power = np.abs(np.fft.rfft(y - np.median(y))) ** 2
    total = float(power.sum())
    frequencies = np.fft.rfftfreq(len(y))
    return float(power[frequencies >= cutoff].sum() / total) if total > 0 else 0.0
