"""Numerical screening helpers retained from the original 50-set analysis.

Only pure metrics are exposed here. Run evaluate_calibration_timecorrected_v2.py
for the final symmetric-baseline observational screening protocol.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

PEAK_RESPONSE_THRESHOLD_M = 0.02


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def summary(values: list[float], prefix: str) -> dict:
    array = finite(values)
    if not len(array):
        return {
            f"{prefix}_median": np.nan,
            f"{prefix}_q1": np.nan,
            f"{prefix}_q3": np.nan,
            f"{prefix}_iqr": np.nan,
        }
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        f"{prefix}_median": float(median),
        f"{prefix}_q1": float(q1),
        f"{prefix}_q3": float(q3),
        f"{prefix}_iqr": float(q3 - q1),
    }


def pearson_r(obs: np.ndarray, sim: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(sim)
    if int(valid.sum()) < 6:
        return np.nan
    obs_v = obs[valid]
    sim_v = sim[valid]
    if float(np.std(obs_v)) < 1e-9 or float(np.std(sim_v)) < 1e-9:
        return np.nan
    return float(np.corrcoef(obs_v, sim_v)[0, 1])


def pair_metrics(
    times: pd.DatetimeIndex,
    obs: np.ndarray,
    sim: np.ndarray,
) -> dict:
    valid = np.isfinite(obs) & np.isfinite(sim)
    obs_v = obs[valid]
    sim_v = sim[valid]
    times_v = times[valid]
    residual = sim_v - obs_v
    sse = float(np.sum(residual**2))
    centred = obs_v - float(np.mean(obs_v))
    sst = float(np.sum(centred**2))
    nse = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    obs_sum = float(np.sum(obs_v))
    pbias = (
        float(100.0 * np.sum(residual) / obs_sum)
        if abs(obs_sum) > 1e-9
        else np.nan
    )
    obs_peak = max(0.0, float(np.max(obs_v)))
    sim_peak = max(0.0, float(np.max(sim_v)))
    peak_error = sim_peak - obs_peak

    timing_error = np.nan
    if obs_peak >= PEAK_RESPONSE_THRESHOLD_M:
        obs_peak_index = int(np.flatnonzero(obs_v == np.max(obs_v))[0])
        sim_peak_index = int(np.flatnonzero(sim_v == np.max(sim_v))[0])
        timing_error = float(
            (times_v[sim_peak_index] - times_v[obs_peak_index]).total_seconds()
            / 60.0
        )

    return {
        "nse": nse,
        "rmse_m": rmse,
        "pbias_pct": pbias,
        "pearson_r": pearson_r(obs_v, sim_v),
        "peak_error_m_signed": peak_error,
        "peak_error_m_abs": abs(peak_error),
        "peak_timing_min_signed": timing_error,
        "peak_timing_min_abs": abs(timing_error),
        "sse": sse,
        "sst": sst,
        "n_points": int(len(obs_v)),
        "obs_sum": obs_sum,
        "sim_sum": float(np.sum(sim_v)),
    }


def aggregate_set(rows: list[dict], set_id: int) -> dict:
    output = {
        "set_id": set_id,
        "n_station_event_pairs": len(rows),
        "n_nse_pairs": int(sum(np.isfinite(row["nse"]) for row in rows)),
        "n_peak_timing_pairs": int(
            sum(np.isfinite(row["peak_timing_min_signed"]) for row in rows)
        ),
    }
    for key, prefix in (
        ("nse", "nse"),
        ("rmse_m", "rmse_m"),
        ("pbias_pct", "pbias_pct"),
        ("pearson_r", "pearson_r"),
        ("peak_error_m_signed", "peak_error_m_signed"),
        ("peak_error_m_abs", "peak_error_m_abs"),
        ("peak_timing_min_signed", "peak_timing_min_signed"),
        ("peak_timing_min_abs", "peak_timing_min_abs"),
    ):
        output.update(summary([row[key] for row in rows], prefix))

    total_sse = float(sum(row["sse"] for row in rows))
    total_sst = float(sum(row["sst"] for row in rows))
    total_points = int(sum(row["n_points"] for row in rows))
    total_obs = float(sum(row["obs_sum"] for row in rows))
    total_sim = float(sum(row["sim_sum"] for row in rows))
    output["pooled_nse_station_event_centered"] = (
        float(1.0 - total_sse / total_sst) if total_sst > 1e-12 else np.nan
    )
    output["pooled_rmse_m"] = (
        float(math.sqrt(total_sse / total_points)) if total_points else np.nan
    )
    output["pooled_pbias_pct"] = (
        float(100.0 * (total_sim - total_obs) / total_obs)
        if abs(total_obs) > 1e-9
        else np.nan
    )
    return output


def fisher_mean_correlations(values: list[float]) -> float:
    array = finite(values)
    if not len(array):
        return np.nan
    clipped = np.clip(array, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))
