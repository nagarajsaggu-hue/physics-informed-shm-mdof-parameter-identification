"""
data_mdof.py

Utilities for loading 2DOF dataset runs and creating collocation points.

"""

from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
import torch


def load_full_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[data_mdof] Loaded CSV '{csv_path}' with shape {df.shape}")
    return df


def get_all_run_ids(df: pd.DataFrame) -> List[int]:
    run_ids = sorted(df["RunID"].unique())
    print(f"[data_mdof] Found {len(run_ids)} runs: "
          f"{run_ids[:5]}{' ...' if len(run_ids) > 5 else ''}")
    return run_ids


def _pick_cols(sub: pd.DataFrame, preferred: List[str], fallback: List[str]) -> List[str]:
    """Pick preferred columns if all exist; else fallback; else raise."""
    if all(c in sub.columns for c in preferred):
        return preferred
    if all(c in sub.columns for c in fallback):
        return fallback
    raise KeyError(
        f"Missing columns. Tried preferred={preferred} and fallback={fallback}. "
        f"Available={list(sub.columns)[:30]}..."
    )


def load_run_data(
    df: pd.DataFrame,
    run_id: int,
    device: torch.device,
    return_va: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
          torch.Tensor, torch.Tensor, Dict[str, float]],
]:
    """
    Returns (default):
        t_data  : [N,1]  time in seconds
        F_data  : [N,2]  forces [F1, F2]  — F2≠0 for dual excitation
        u_meas  : [N,2]  measured displacements
        true_par: dict   m1,m2,k1,k2,alpha,beta,omega1,omega2,zeta1,zeta2

    Returns (if return_va=True):
        t_data, F_data, u_meas, v_meas, a_meas, true_par

    Rayleigh convention used throughout:
        C = alpha * M  +  beta * K
        alpha  ≈ 0.35   (mass-proportional,      CSV column: 'beta_ray')
        beta   ≈ 3e-5   (stiffness-proportional, CSV column: 'alpha'  )
    """
    sub = df[df["RunID"] == run_id].copy()
    if sub.empty:
        raise ValueError(f"[data_mdof] No rows found for RunID={run_id}")

    # ── Time ──────────────────────────────────────────────────────────────
    t = sub["time"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # ── Forces (F1 + F2 both active — dual excitation) ────────────────────
    F = sub[["F1", "F2"]].to_numpy(dtype=np.float32)

    # ── Displacements ─────────────────────────────────────────────────────
    u_cols = _pick_cols(sub, preferred=["u1_meas", "u2_meas"], fallback=["u1", "u2"])
    u_meas = sub[u_cols].to_numpy(dtype=np.float32)

    # ── Velocity + Acceleration (Stage-2) ─────────────────────────────────
    if return_va:
        v_cols = _pick_cols(sub, preferred=["v1_meas", "v2_meas"], fallback=["v1", "v2"])
        a_cols = _pick_cols(sub, preferred=["a1_meas", "a2_meas"], fallback=["a1", "a2"])
        v_meas = sub[v_cols].to_numpy(dtype=np.float32)
        a_meas = sub[a_cols].to_numpy(dtype=np.float32)

    # ── True parameters ───────────────────────────────────────────────────
    #  CSV column roles are physically swapped.
    #   CSV 'beta_ray' = mass-proportional Rayleigh α  (large, ~0.35)
    #   CSV 'alpha'    = stiffness-proportional     β  (tiny,  ~3e-5)
    # Mapping corrected here so downstream code uses standard convention:
    #   C = alpha*M + beta*K,  zeta = 0.5*(alpha/omega + beta*omega)
    true_par: Dict[str, float] = {
        "m1":  float(sub["m1"].iloc[0]),
        "m2":  float(sub["m2"].iloc[0]),
        "k1":  float(sub["k1"].iloc[0]),
        "k2":  float(sub["k2"].iloc[0]),
        # ↓ SWAPPED relative to old version ↓
        "alpha": float(sub["beta_ray"].iloc[0]),  # mass-prop coeff  (~0.35)
        "beta":  float(sub["alpha"].iloc[0]),     # stiff-prop coeff (~3e-5)
        # modal targets
        "omega1": float(sub["omega1"].iloc[0]),
        "omega2": float(sub["omega2"].iloc[0]),
        "zeta1":  float(sub["zeta1"].iloc[0]),
        "zeta2":  float(sub["zeta2"].iloc[0]),
    }

    # ── Tensors ───────────────────────────────────────────────────────────
    t_tensor = torch.from_numpy(t).to(device)
    F_tensor = torch.from_numpy(F).to(device)
    u_tensor = torch.from_numpy(u_meas).to(device)

    if not return_va:
        print(f"[data_mdof] RunID {run_id}: N={t_tensor.shape[0]} | "
              f"m1={true_par['m1']:.0f}  m2={true_par['m2']:.0f}  "
              f"k1={true_par['k1']:.0f}  k2={true_par['k2']:.0f} | "
              f"alpha={true_par['alpha']:.4f}  beta={true_par['beta']:.3e}")
        return t_tensor, F_tensor, u_tensor, true_par

    v_tensor = torch.from_numpy(v_meas).to(device)
    a_tensor = torch.from_numpy(a_meas).to(device)

    print(f"[data_mdof] RunID {run_id}: N={t_tensor.shape[0]} | "
          f"m1={true_par['m1']:.0f}  m2={true_par['m2']:.0f}  "
          f"k1={true_par['k1']:.0f}  k2={true_par['k2']:.0f} | "
          f"alpha={true_par['alpha']:.4f}  beta={true_par['beta']:.3e} (u+v+a)")
    return t_tensor, F_tensor, u_tensor, v_tensor, a_tensor, true_par


def create_collocation_points(
    t_data: torch.Tensor,
    F_data: torch.Tensor,
    n_colloc: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collocation points for physics loss.
    Returns all points if n_colloc >= N, else a random subset on same device.
    """
    N = t_data.shape[0]
    if n_colloc >= N:
        return t_data, F_data
    idx = torch.randperm(N, device=t_data.device)[:n_colloc]
    return t_data[idx], F_data[idx]
