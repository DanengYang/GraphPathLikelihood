"""Shared constants for the jointFitter toolkit."""

# Snapshot → scale factor map provided by the user (subset for central network)
SNAP_TO_A = {
    2: 0.0769, 3: 0.0833, 4: 0.0909, 6: 0.1000, 8: 0.1111, 11: 0.1250, 13: 0.1429,
    17: 0.1667, 21: 0.2000, 25: 0.2500, 33: 0.3333, 40: 0.4000, 50: 0.5000,
    59: 0.5882, 67: 0.6667, 72: 0.7143, 78: 0.7692, 84: 0.8333, 91: 0.9091, 99: 1.0000,
}

BACKGROUND_FIELDS = {"M_halo", "rhalf_dm"}

# Fields treated as diagnostics (not trained, not in loss)
DERIVED_FIELDS = {"SFR", "Z_star"}

FIELD_ALIASES = {
    "M_halo": ["M_halo", "SubhaloMass", "Group_M_Crit200", "M200c", "mass_halo"],
    "M_gas": ["M_gas", "GasMass", "SubhaloGasMass"],
    "M_star": ["M_star", "StellarMass", "SubhaloStellarMass"],
    "M_wind": ["M_wind", "SubhaloWindMass", "mass_wind"],
    "M_bh": ["M_bh", "BHMass", "SubhaloBHMass"],
    "MassType": ["SubhaloMassType", "MassType"],
    "HalfmassRadType": ["SubhaloHalfmassRadType", "HalfmassRadType"],
    "SFR": ["SFR", "SubhaloSFR", "StarFormationRate", "sfr"],
    "Z_gas": ["Z_gas", "SubhaloGasMetallicity", "Zgas"],
    "Z_star": ["Z_star", "SubhaloStarMetallicity", "SubhaloStellarMetallicity", "Zstar"],
    "cool_rate": ["cool_rate", "CoolingRate", "GasCoolingRate", "cooling_rate"],
    "pos": ["pos", "SubhaloPos", "position", "xyz", "CM", "SubhaloCM", "pos_ckpch"],
    "vel": ["vel", "SubhaloVel", "velocity", "vxyz"],
    "R200c": ["R200c", "Group_R_Crit200", "R_200c", "r200c", "R200_com_kpc_h_from_sub"],
    "Vmax": ["Vmax", "SubhaloVmax", "V_max"],
    "host": ["host", "HostID", "host_id"],
}

TEMPORAL_EDGE_KEYS = {"temporal", "time", "sublink", "progenitor", "temporal_progenitor"}
HOST_EDGE_KEYS = {"host", "spatial", "within", "host_edge"}

__all__ = [
    "SNAP_TO_A",
    "BACKGROUND_FIELDS",
    "DERIVED_FIELDS",
    "FIELD_ALIASES",
    "TEMPORAL_EDGE_KEYS",
    "HOST_EDGE_KEYS",
]
