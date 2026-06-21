"""envs/metrics.py
Shared metric utilities for training and evaluation scripts.

All functions here are agent-agnostic — they operate on environment
outputs (decoded_obs, extra_json, cfg) only. 

    from envs.metrics import compute_sla_rates, nan_or_round
"""

from __future__ import annotations

from typing import Dict, List, Any

from envs.slice_gym_env import SLICE_NAMES


def compute_sla_rates(
    decoded_obs: Dict,
    cfg: Dict,
    extra_json: Dict | None = None,
) -> Dict[str, float]:
  
    max_thr = cfg["env"]["max_thr_mbps"]
    min_thr = cfg["env"]["min_thr_mbps"]
    max_lat = cfg["env"]["max_lat_ms"]

    demand_active = [1, 1, 1]
    if (
        extra_json
        and isinstance(extra_json.get("demand_active"), list)
        and len(extra_json["demand_active"]) == 3
    ):
        demand_active = [int(x) for x in extra_json["demand_active"]]

    per_slice: List[float] = []
    sat = 0
    den = 0

    for i, s in enumerate(SLICE_NAMES):
        if demand_active[i] == 0:
            per_slice.append(1.0)   # inactive — not a violation
            continue
        den += 1
        thr = float(decoded_obs["throughput"][s]) * float(max_thr[s])

        
        lat_ms_raw = (
            extra_json.get("lat_ms")
            if extra_json and isinstance(extra_json.get("lat_ms"), list) and len(extra_json["lat_ms"]) == 3
            else None
        )
        if lat_ms_raw is not None:
            lat_ok = float(lat_ms_raw[i]) <= float(max_lat[s])
        else:
            lat_norm = float(decoded_obs["latency"][s])
            lat_ok = lat_norm <= 0.5

        pelr_raw = (
            extra_json.get("pkt_loss_rate")
            if extra_json and isinstance(extra_json.get("pkt_loss_rate"), list) and len(extra_json["pkt_loss_rate"]) == 3
            else None
        )
        pelr_ok = True if s != "URLLC" else (
            float(pelr_raw[1]) <= 1e-4 if pelr_raw is not None else True
        )

        ok = (thr >= float(min_thr[s])) and lat_ok and pelr_ok
        per_slice.append(1.0 if ok else 0.0)
        if ok:
            sat += 1

    return {
        "overall": 1.0 if den == 0 else sat / den,
        "embb":    per_slice[0],
        "urllc":   per_slice[1],
        "mmtc":    per_slice[2],
    }


def nan_or_round(v: float, decimals: int = 6) -> float | None:
    """Round a float for JSON serialisation; return None for NaN."""
    if isinstance(v, float) and v != v:   
        return None
    return round(float(v), decimals)
