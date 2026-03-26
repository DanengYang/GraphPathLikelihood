"""Generic helpers for property lookups."""

from jf_constants import FIELD_ALIASES


def get_first(mapping, keys):
    """Return the first present, non-None value found under any key in keys."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def get_prop_any(node_like, logical_name, default=None):
    """Fetch a property using logical aliases across props/weights fields."""
    aliases = FIELD_ALIASES.get(logical_name, [logical_name])
    props = node_like.get("props", {}) if isinstance(node_like, dict) else {}
    value = get_first(props, aliases)
    if value is not None:
        return value

    weights = node_like.get("weights", {}) if isinstance(node_like, dict) else {}
    value = get_first(weights, aliases)
    if value is not None:
        return value

    value = get_first(node_like, aliases) if isinstance(node_like, dict) else None
    return value if value is not None else default

def canonical_field_name(name: str) -> str:
    """Map any alias to its canonical logical field name."""
    for canonical, aliases in FIELD_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name


__all__ = ["get_first", "get_prop_any", "canonical_field_name"]
