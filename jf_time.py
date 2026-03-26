"""Utilities for mapping snapshots to cosmological times."""

from __future__ import annotations

import numpy as np

from jf_constants import SNAP_TO_A


def snap_to_scale_factor(snapshot):
    """Return the scale factor a for a given snapshot id (tabulated subset)."""
    return SNAP_TO_A.get(int(snapshot), None) if snapshot is not None else None


def scale_to_redshift(scale_factor):
    """Convert a scale factor to redshift."""
    if scale_factor is None or scale_factor <= 0:
        return None
    return (1.0 / scale_factor) - 1.0


def cosmic_time_gyr_LCDM(
    scale_factor,
    h=0.6774,
    omega_m=0.3089,
    omega_lambda=0.6911,
):
    """Analytic ΛCDM cosmic time t(a) in Gyr (Planck-like cosmology)."""
    if scale_factor is None or scale_factor <= 0:
        return None
    h0_inv_gyr = 9.778 / h
    root = (omega_lambda / max(omega_m, 1e-12)) ** 0.5
    return (2.0 / (3.0 * (omega_lambda ** 0.5))) * h0_inv_gyr * np.arcsinh(root * (scale_factor ** 1.5))


def _estimate_scale_factor_from_layer(layer):
    nodes = layer.get("nodes") or []
    ratios = []
    for node in nodes:
        props = node.get("props", {})
        r_phys = props.get("R200_phys_kpc_h_from_sub")
        r_com = props.get("R200_com_kpc_h_from_sub")
        if r_phys is None or r_com is None:
            continue
        try:
            r_phys = float(r_phys)
            r_com = float(r_com)
        except (TypeError, ValueError):
            continue
        if r_com <= 0.0 or r_phys <= 0.0:
            continue
        ratios.append(r_phys / r_com)
        if len(ratios) >= 64:
            break
    if ratios:
        ratios = [r for r in ratios if np.isfinite(r) and r > 0.0]
    if not ratios:
        return None
    return float(np.median(ratios))


def ensure_layer_times_physical(layers, overwrite=False):
    """Fill each layer's 'time' and 'scale_factor' fields with cosmic values."""
    changed = 0
    for layer in layers:
        if layer.get("time") is not None and not overwrite:
            continue
        scale = snap_to_scale_factor(layer.get("snap"))
        if scale is None:
            scale = _estimate_scale_factor_from_layer(layer)
        if scale is None or scale <= 0.0:
            continue
        time_val = cosmic_time_gyr_LCDM(scale)
        if time_val is None:
            continue
        layer["scale_factor"] = float(scale)
        layer["time"] = float(time_val)
        changed += 1
    return changed


__all__ = [
    "snap_to_scale_factor",
    "scale_to_redshift",
    "cosmic_time_gyr_LCDM",
    "ensure_layer_times_physical",
]
