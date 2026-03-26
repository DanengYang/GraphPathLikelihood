"""Helpers for pulling typed data arrays from layered graph nodes."""

from __future__ import annotations

import numpy as np

from jf_utils import get_prop_any

# --- BEGIN RHALF PATCH HELPERS ---
_RHALF_INDEX = {"rhalf_gas": 0, "rhalf_dm": 1, "rhalf_star": 4}
_RHALF_KEYS_MULTI = ("SubhaloHalfmassRadType", "HalfmassRadType")
_RHALF_KEYS_SINGLE = ("SubhaloHalfmassRad", "HalfmassRad")
_RHALF_LOG_KEYS = {
    "rhalf_gas":  ("logrhalf_gas","logR_gas","logR_half_gas"),
    "rhalf_dm":   ("logrhalf_dm", "logR_dm", "logR_half_dm"),
    "rhalf_star": ("logrhalf_star","logR_star","logR_half_star"),
}
_RHALF_MIN = 1e-6  # positive floor for log-plots


def _as_float(x):
    try:
        # handles scalar/1-element arrays/lists
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.array(x)
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[0])
        return float(x)
    except Exception:
        return None

def _resolve_rhalf_for_node(node, logical_name, get_prop):
    """Return a single linear radius value (float or None) for one node."""
    j = _RHALF_INDEX[logical_name]

    # 1) Predicted linear radius
    v = get_prop(node, logical_name)
    v = _as_float(v) if v is not None else None
    if v is not None and np.isfinite(v) and v > 0:
        return max(v, _RHALF_MIN)

    # 2) Predicted log radius -> exponentiate
    for lk in _RHALF_LOG_KEYS[logical_name]:
        lv = get_prop(node, lk)
        lv = _as_float(lv) if lv is not None else None
        if lv is not None and np.isfinite(lv):
            vv = float(np.exp(lv))
            return max(vv, _RHALF_MIN) if vv > 0 else None

    # 3) Truth from multi-component arrays (gas,dm,*,*,star)
    for key in _RHALF_KEYS_MULTI:
        arr = get_prop(node, key)
        if arr is None:
            continue
        try:
            arr = np.array(arr, dtype=float).reshape(-1)
            if arr.size > j and np.isfinite(arr[j]) and arr[j] > 0:
                return max(float(arr[j]), _RHALF_MIN)
        except Exception:
            pass

    # 4) Truth from single-component scalar
    for key in _RHALF_KEYS_SINGLE:
        sv = get_prop(node, key)
        sv = _as_float(sv) if sv is not None else None
        if sv is not None and np.isfinite(sv) and sv > 0:
            return max(sv, _RHALF_MIN)

    return None

def _extract_rhalf_array(layer, logical_name, get_prop):
    n = len(layer.get("nodes", []))
    out = np.full(n, np.nan, dtype=float)
    for i, node in enumerate(layer["nodes"]):
        out[i] = _resolve_rhalf_for_node(node, logical_name, get_prop) or np.nan
    return out
# --- END RHALF PATCH HELPERS ---

def extract_field_array(layer, logical_name):
    """
    Existing implementation extended with radii mapping.
    IMPORTANT: Do not change your SubhaloMassType handling.
    We insert the rhalf branch early, then fall through to your original logic.
    """
    get_prop = get_prop_any  # or however you currently fetch nested props

    # --- BEGIN SPECIAL DISPATCHES ---
    if logical_name in ("rhalf_gas","rhalf_dm","rhalf_star"):
        return _extract_rhalf_array(layer, logical_name, get_prop)

    if logical_name == "halo_pos":
        n = len(layer.get("nodes", []))
        out = np.full((n, 3), np.nan, dtype=float)
        for idx, node in enumerate(layer["nodes"]):
            raw = get_prop_any(node, "pos")
            if raw is None:
                raw = get_prop_any(node, "SubhaloPos")
            if raw is None:
                raw = get_prop_any(node, "position")
            if raw is not None:
                try:
                    arr = np.array(raw, dtype=float).reshape(-1)
                    if arr.size >= 3:
                        out[idx] = arr[:3]
                except Exception:
                    pass
        return out

    if logical_name == "halo_vel":
        n = len(layer.get("nodes", []))
        out = np.full((n, 3), np.nan, dtype=float)
        for idx, node in enumerate(layer["nodes"]):
            raw = get_prop_any(node, "vel")
            if raw is None:
                raw = get_prop_any(node, "velocity")
            if raw is None:
                raw = get_prop_any(node, "SubhaloVel")
            if raw is not None:
                try:
                    arr = np.array(raw, dtype=float).reshape(-1)
                    if arr.size >= 3:
                        out[idx] = arr[:3]
                except Exception:
                    pass
        return out
    # --- END SPECIAL DISPATCHES ---

    count = len(layer["nodes"])
    result = np.full(count, np.nan, dtype=float)

    def _read_mass_type(node, index):
        array = get_prop_any(node, "SubhaloMassType")
        if array is None:
            return None
        try:
            arr = np.array(array, dtype=float).reshape(-1)
            if index < arr.size:
                return float(arr[index])
        except Exception:
            return None
        return None

    for idx, node in enumerate(layer["nodes"]):
        value = None
        if logical_name == "M_gas":
            value = _read_mass_type(node, 0)
            if value is None:
                raw = get_prop_any(node, logical_name)
                if raw is None:
                    raw = get_prop_any(node, "M_gas")
                if raw is not None:
                    try:
                        value = float(raw)
                    except Exception:
                        value = None

        elif logical_name == "M_halo":
            value = _read_mass_type(node, 1)
            if value is None:
                raw = get_prop_any(node, "M_halo")
                if raw is not None:
                    try:
                        value = float(raw)
                    except Exception:
                        value = None
            if value is None:
                raw = get_prop_any(node, logical_name)
                value = float(raw) if raw is not None else None

        elif logical_name == "M_star":
            value = _read_mass_type(node, 4)
            if value is None:
                raw = get_prop_any(node, "M_star")
                if raw is None:
                    raw = get_prop_any(node, logical_name)
                if raw is not None:
                    try:
                        value = float(raw)
                    except Exception:
                        value = None

        elif logical_name == "M_bh":
            raw = get_prop_any(node, "M_bh")
            if raw is not None:
                try:
                    value = float(raw)
                except Exception:
                    value = None
            if value is None:
                value = _read_mass_type(node, 5)
                if value is None:
                    raw = get_prop_any(node, "M_bh")
                    value = float(raw) if raw is not None else None

        else:
            raw = get_prop_any(node, logical_name)
            if raw is not None:
                try:
                    if isinstance(raw, (list, tuple, np.ndarray)):
                        value = float(np.array(raw).reshape(-1)[0])
                    else:
                        value = float(raw)
                except Exception:
                    value = None

        result[idx] = np.nan if value is None else value

    return result


__all__ = ["extract_field_array"]
