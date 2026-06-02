"""
evaluate_stage1.py — Post-training evaluation for Stage 1 Baseline 2DOF PINN

Run this AFTER trainer_stage1.py has completed.
It loads saved model.pt + data for each run and produces:

Per-run outputs (runs/run_XXX/):
  eval_signal_metrics.csv       — MSE, RMSE, MAE, R², NRMSE for u/v/a (both DOFs)
  eval_residuals.csv            — physics residual stats (mean|R|, RMS, max, std)
  eval_param_metrics.csv        — true/pred/error for all 10 parameters
  plots/eval_signal_metrics.png — R² and NRMSE bar chart
  plots/eval_residuals.png      — residual waveform over time (DOF1 & DOF2)
  plots/eval_param_true_vs_pred.png — 2×5 grid: True vs Predicted per parameter
  plots/eval_summary_heatmap.png    — colour heatmap of % errors across parameters

Global outputs (stage1/):
  evaluation_summary.csv        — all run metrics in one table
  plots/eval_r2_all_runs.png    — R² per signal per run (grouped bar)
  plots/eval_nrmse_all_runs.png — NRMSE per signal per run (grouped bar)
  plots/eval_phys_residual.png  — RMS physics residual per run
  plots/eval_params_within_threshold.png — params within 5%/10% per run
  evaluation_report.md          — Markdown report with all tables

Usage:
  python evaluate_stage2.py                  # evaluate runs 1-5
  python -m mdof_2dof_pinn.evaluate_stage1 --start 2 --end 4
  python evaluate_stage2.py --run 3          # single run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# PATH SETUP  (mirrors trainer_stage2.py)
# ──────────────────────────────────────────────────────────────────────────────
current_script_path = Path(__file__).resolve().parent
_candidate = current_script_path
project_root = _candidate
for _ in range(3):
    if (_candidate / "mdof_2dof_pinn" / "__init__.py").exists():
        project_root = _candidate
        break
    _candidate = _candidate.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from mdof_2dof_pinn.data_mdof  import load_full_csv, load_run_data, create_collocation_points
    from mdof_2dof_pinn.model_mdof import TwoDOFPINN
except ImportError:
    print("Critical Error: Could not import 'mdof_2dof_pinn'.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match trainer_stage2.py)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data": {
        "csv_path":        "Data/mdof_2dof_216runs_pinn.csv",
        "n_colloc_factor": 4.0,
    },
    "training": {
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
        "warm_epochs": 500,
    },
    "loss": {
        "perturb_level": 0.10,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER METADATA
# ──────────────────────────────────────────────────────────────────────────────
PARAM_META = {
    "m1":     ("m1",     "kg"),
    "m2":     ("m2",     "kg"),
    "k1":     ("k1",     "N/m"),
    "k2":     ("k2",     "N/m"),
    "alpha":  ("alpha",  "–"),
    "beta":   ("beta",   "–"),
    "omega1": ("omega1", "rad/s"),
    "omega2": ("omega2", "rad/s"),
    "zeta1":  ("zeta1",  "–"),
    "zeta2":  ("zeta2",  "–"),
}

SIGNAL_LABELS = ["u_DOF1", "u_DOF2", "v_DOF1", "v_DOF2", "a_DOF1", "a_DOF2"]

# ──────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ──────────────────────────────────────────────────────────────────────────────
_DARK = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size":        10,
}
_C = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
      "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#e3b341"]

def _style(): plt.rcParams.update(_DARK)

def _safe_save(fig, path):
    w, h = fig.get_size_inches()
    fig.set_size_inches(max(w, 4), max(h, 3))
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"  WARNING: could not save {Path(path).name}: {e}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL SCAFFOLD  (matches ScaledTwoDOFPINN in trainer)
# ──────────────────────────────────────────────────────────────────────────────
class ScaledTwoDOFPINN(TwoDOFPINN):
    def __init__(self, init_params: Dict[str, float]):
        super().__init__(init_params)
        for name in ["m1", "m2", "k1", "k2"]:
            if name in self._parameters: del self._parameters[name]
            if hasattr(self, name):       delattr(self, name)
        self.register_buffer("m1_ref", torch.tensor(init_params["m1"], dtype=torch.float32))
        self.register_buffer("m2_ref", torch.tensor(init_params["m2"], dtype=torch.float32))
        self.register_buffer("k1_ref", torch.tensor(init_params["k1"], dtype=torch.float32))
        self.register_buffer("k2_ref", torch.tensor(init_params["k2"], dtype=torch.float32))
        self.m1_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.m2_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.k1_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.k2_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    @property
    def m1(self): return self.m1_ref * self.m1_scale
    @property
    def m2(self): return self.m2_ref * self.m2_scale
    @property
    def k1(self): return self.k1_ref * self.k1_scale
    @property
    def k2(self): return self.k2_ref * self.k2_scale

def _alpha_beta_vals(model):
    if hasattr(model, "alpha_val") and hasattr(model, "beta_val"):
        return model.alpha_val(), model.beta_val()
    return model.alpha, model.beta

def _safe_pct_err(pred, true):
    return abs(pred - true) / (abs(true) + 1e-12) * 100.0

# ──────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def signal_metrics(true_arr: np.ndarray, pred_arr: np.ndarray, label: str) -> List[dict]:
    rows = []
    n_cols = true_arr.shape[1] if true_arr.ndim == 2 else 1
    for col in range(n_cols):
        t = true_arr[:, col] if true_arr.ndim == 2 else true_arr.squeeze()
        p = pred_arr[:, col] if pred_arr.ndim == 2 else pred_arr.squeeze()
        mse   = float(np.mean((t - p) ** 2))
        rmse  = float(np.sqrt(mse))
        mae   = float(np.mean(np.abs(t - p)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2    = float(1.0 - ss_res / (ss_tot + 1e-12))
        nrmse = rmse / (float(np.max(np.abs(t))) + 1e-12)
        rows.append({
            "signal": f"{label}_DOF{col+1}",
            "MSE":    mse,   "RMSE":  rmse,
            "MAE":    mae,   "R2":    r2,
            "NRMSE":  nrmse,
        })
    return rows

def residual_metrics(res_np: np.ndarray) -> List[dict]:
    rows = []
    for col, lbl in enumerate(["DOF1", "DOF2"]):
        r = res_np[:, col] if res_np.ndim == 2 else res_np
        rows.append({
            "DOF":               lbl,
            "mean_abs_residual": float(np.mean(np.abs(r))),
            "max_abs_residual":  float(np.max(np.abs(r))),
            "rms_residual":      float(np.sqrt(np.mean(r ** 2))),
            "std_residual":      float(np.std(r)),
        })
    return rows

def param_metrics(metrics_csv_row: dict) -> List[dict]:
    rows = []
    for p, (lbl, unit) in PARAM_META.items():
        if f"{p}_true" not in metrics_csv_row:
            continue
        tv = float(metrics_csv_row[f"{p}_true"])
        pv = float(metrics_csv_row[f"{p}_pred"])
        pct = _safe_pct_err(pv, tv)
        rows.append({
            "parameter":    lbl,
            "unit":         unit,
            "true":         tv,
            "predicted":    pv,
            "abs_error":    abs(pv - tv),
            "rel_error":    abs(pv - tv) / (abs(tv) + 1e-12),
            "pct_error":    pct,
            "within_5pct":  pct < 5.0,
            "within_10pct": pct < 10.0,
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# PER-RUN PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_signal_metrics(signal_df: pd.DataFrame, run_id: int, save_path: Path):
    """R² and NRMSE bar chart for u/v/a both DOFs."""
    _style()
    sig_names  = signal_df["signal"].tolist()
    r2_vals    = signal_df["R2"].tolist()
    nrmse_vals = signal_df["NRMSE"].tolist()
    bar_cols   = [_C[i % len(_C)] for i in range(len(sig_names))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Signal Reconstruction Metrics — Run {run_id}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(sig_names, r2_vals, color=bar_cols, alpha=0.85, edgecolor="#30363d")
    ax.axhline(1.0,  color=_C[2], ls="--", lw=0.8, label="Perfect R²=1")
    ax.axhline(0.99, color="white", ls=":", lw=0.8, alpha=0.6, label="R²=0.99")
    lo = max(min(r2_vals) - 0.02, 0.0)
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel("R² Score"); ax.set_title("R² per Signal")
    for j, v in enumerate(r2_vals):
        ax.text(j, v + 0.001, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.legend(fontsize=7); ax.grid(True, axis="y")

    ax = axes[1]
    ax.bar(sig_names, nrmse_vals, color=bar_cols, alpha=0.85, edgecolor="#30363d")
    ax.set_ylabel("NRMSE"); ax.set_title("NRMSE per Signal  (lower = better)")
    for j, v in enumerate(nrmse_vals):
        ax.text(j, v + 0.0002, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.grid(True, axis="y")

    fig.subplots_adjust(bottom=0.15, wspace=0.3, top=0.88)
    _safe_save(fig, save_path)


def plot_residuals(t_phys_np: np.ndarray, res_np: np.ndarray,
                   residual_df: pd.DataFrame, run_id: int, save_path: Path):
    """Physics residual waveform over time for both DOFs."""
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Physics Residual (ODE Violation) — Run {run_id}",
                 fontsize=13, fontweight="bold")

    for col, (dof_lbl, ax) in enumerate(zip(["DOF 1", "DOF 2"], axes)):
        r = res_np[:, col] if res_np.ndim == 2 else res_np
        stats = residual_df.iloc[col]
        ax.plot(t_phys_np, r, color=_C[col], lw=0.7, alpha=0.9, label="Residual R(t)")
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(t_phys_np, r, 0, color=_C[col], alpha=0.12)
        ax.set_ylabel("Residual value"); ax.set_title(
            f"{dof_lbl}  |  RMS = {stats['rms_residual']:.3e}"
            f"  |  Mean|R| = {stats['mean_abs_residual']:.3e}"
            f"  |  Max|R| = {stats['max_abs_residual']:.3e}")
        ax.legend(fontsize=8); ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.35, top=0.90)
    _safe_save(fig, save_path)


def plot_param_true_vs_pred(param_df: pd.DataFrame, run_id: int, save_path: Path):
    """2×5 subplot grid: True vs Predicted bar for each parameter."""
    _style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(f"Parameter True vs Predicted — Run {run_id}",
                 fontsize=13, fontweight="bold")

    for i, (_, row) in enumerate(param_df.iterrows()):
        ax = axes.flatten()[i]
        tv, pv = row["true"], row["predicted"]
        bar_c = _C[1] if row["pct_error"] < 5 else _C[4] if row["pct_error"] < 10 else _C[2]
        ax.bar(["True", "Predicted"], [tv, pv],
               color=[_C[1], bar_c], alpha=0.85, edgecolor="#30363d")
        ax.set_title(
            f"{row['parameter']}  ({row['unit']})\nerr = {row['pct_error']:.2f}%",
            fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.grid(True, axis="y")

    fig.subplots_adjust(hspace=0.55, wspace=0.45, top=0.88)
    _safe_save(fig, save_path)


def plot_summary_heatmap(param_df: pd.DataFrame, run_id: int, save_path: Path):
    """Single-row colour heatmap of % error for all parameters."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.suptitle(f"Parameter % Error Heatmap — Run {run_id}",
                 fontsize=12, fontweight="bold")

    pnames   = param_df["parameter"].tolist()
    pct_errs = param_df["pct_error"].values.reshape(1, -1)

    im = ax.imshow(pct_errs, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=15)
    ax.set_xticks(range(len(pnames))); ax.set_xticklabels(pnames, fontsize=9)
    ax.set_yticks([])
    for j, v in enumerate(param_df["pct_error"].values):
        ax.text(j, 0, f"{v:.2f}%", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if v > 10 else "black")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, fraction=0.05)
    cbar.set_label("% Error  (green ≤ 5% | yellow 5-10% | red > 10%)", fontsize=8)
    fig.subplots_adjust(top=0.78)
    _safe_save(fig, save_path)


def plot_response_comparison(
    t_np, u_meas_np, v_meas_np, a_meas_np,
    u_pred_np, v_pred_np, a_pred_np,
    run_id: int, save_path: Path,
):
    """3-row × 2-col: Measured vs PINN for u, v, a at both DOFs."""
    _style()
    row_labels = ["Displacement (m)", "Velocity (m/s)", "Accel (m/s²)"]
    meas_data  = [u_meas_np, v_meas_np, a_meas_np]
    pred_data  = [u_pred_np, v_pred_np, a_pred_np]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Response: Measured vs PINN Prediction — Run {run_id}",
                 fontsize=14, fontweight="bold")

    for row, (rlbl, meas, pred) in enumerate(zip(row_labels, meas_data, pred_data)):
        for col in range(2):
            ax  = axes[row, col]
            m   = meas[:, col] if meas.ndim == 2 else meas
            p   = pred[:, col] if pred.ndim == 2 else pred
            ax.plot(t_np, m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.85)
            ax.plot(t_np, p, color=_C[0], lw=1.0, label="PINN pred")
            ax.set_ylabel(rlbl, fontsize=8)
            if row == 0:
                ax.set_title(f"DOF {col+1}", fontsize=10, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True)
            if row == 0 and col == 1:
                ax.legend(fontsize=7)

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    _safe_save(fig, save_path)


def plot_psd_comparison(
    t_np, u_meas_np, u_pred_np, run_id: int, save_path: Path,
    omega1_true=None, omega2_true=None,
    omega1_pred=None, omega2_pred=None,
):
    """Welch PSD of measured vs predicted displacement, with natural freq lines."""
    from scipy.signal import welch as sp_welch
    _style()
    dt = float(np.mean(np.diff(t_np)))
    fs = 1.0 / dt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Power Spectral Density (Displacement) — Run {run_id}",
                 fontsize=13, fontweight="bold")

    for col, (dof_lbl, ax) in enumerate(zip(["DOF 1", "DOF 2"], axes)):
        m = u_meas_np[:, col] if u_meas_np.ndim == 2 else u_meas_np
        p = u_pred_np[:, col] if u_pred_np.ndim == 2 else u_pred_np
        f_m, psd_m = sp_welch(m, fs=fs, nperseg=min(256, len(m) // 4))
        f_p, psd_p = sp_welch(p, fs=fs, nperseg=min(256, len(p) // 4))

        ax.semilogy(f_m, psd_m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.85)
        ax.semilogy(f_p, psd_p, color=_C[0], lw=1.0, label="PINN pred")

        if col == 0 and omega1_true:
            ax.axvline(omega1_true / (2*np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₁ true ({omega1_true/(2*np.pi):.2f} Hz)")
        if col == 0 and omega1_pred:
            ax.axvline(omega1_pred / (2*np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₁ pred ({omega1_pred/(2*np.pi):.2f} Hz)")
        if col == 1 and omega2_true:
            ax.axvline(omega2_true / (2*np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₂ true ({omega2_true/(2*np.pi):.2f} Hz)")
        if col == 1 and omega2_pred:
            ax.axvline(omega2_pred / (2*np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₂ pred ({omega2_pred/(2*np.pi):.2f} Hz)")

        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("PSD (m²/Hz)", fontsize=9)
        ax.set_title(dof_lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True)

    fig.subplots_adjust(wspace=0.3, top=0.88)
    _safe_save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-RUN DASHBOARD PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_r2_all_runs(eval_summary: pd.DataFrame, plots_dir: Path):
    """Grouped bar: R² per signal per run."""
    _style()
    run_ids = sorted(eval_summary["RunID"].tolist())
    n_runs  = len(run_ids)
    sig_cols = [c for c in eval_summary.columns if c.startswith("r2_")]
    sig_names = [c.replace("r2_", "") for c in sig_cols]

    x = np.arange(len(sig_names))
    w = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, rid in enumerate(run_ids):
        row  = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        vals = [row[c] for c in sig_cols]
        ax.bar(x + (i - n_runs/2 + 0.5)*w, vals, w,
               label=f"Run {rid}", color=_C[i % len(_C)], alpha=0.85)

    ax.axhline(1.0,  color=_C[2],  ls="--", lw=0.8)
    ax.axhline(0.99, color="white", ls=":",  lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(sig_names, fontsize=9)
    ax.set_ylabel("R² Score"); ax.set_ylim(
        max(eval_summary[sig_cols].values.min() - 0.02, 0), 1.02)
    ax.set_title("R² per Signal — All Runs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_runs); ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12, top=0.90)
    _safe_save(fig, plots_dir / "eval_r2_all_runs.png")


def plot_nrmse_all_runs(eval_summary: pd.DataFrame, plots_dir: Path):
    """Grouped bar: NRMSE per signal per run."""
    _style()
    run_ids   = sorted(eval_summary["RunID"].tolist())
    n_runs    = len(run_ids)
    nrmse_cols = [c for c in eval_summary.columns if c.startswith("nrmse_")]
    sig_names  = [c.replace("nrmse_", "") for c in nrmse_cols]

    x = np.arange(len(sig_names))
    w = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, rid in enumerate(run_ids):
        row  = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        vals = [row[c] for c in nrmse_cols]
        ax.bar(x + (i - n_runs/2 + 0.5)*w, vals, w,
               label=f"Run {rid}", color=_C[i % len(_C)], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(sig_names, fontsize=9)
    ax.set_ylabel("NRMSE  (lower = better)")
    ax.set_title("NRMSE per Signal — All Runs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_runs); ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12, top=0.90)
    _safe_save(fig, plots_dir / "eval_nrmse_all_runs.png")


def plot_phys_residual_all_runs(eval_summary: pd.DataFrame, plots_dir: Path):
    """RMS physics residual per DOF per run."""
    _style()
    run_ids = sorted(eval_summary["RunID"].tolist())
    labels  = [f"Run {r}" for r in run_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Physics Residual (RMS) — All Runs", fontsize=13, fontweight="bold")

    for col, (dof_lbl, ax) in enumerate(zip(["DOF 1", "DOF 2"], axes)):
        col_name = f"phys_rms_dof{col+1}"
        vals = [float(eval_summary[eval_summary["RunID"]==r][col_name].values[0])
                for r in run_ids]
        ax.bar(labels, vals, color=[_C[i % len(_C)] for i in range(len(run_ids))], alpha=0.85)
        ax.set_ylabel("RMS Residual"); ax.set_title(f"{dof_lbl} Physics Residual (RMS)")
        for j, v in enumerate(vals):
            ax.text(j, v * 1.02, f"{v:.2e}", ha="center", fontsize=8)
        ax.grid(True, axis="y")

    fig.subplots_adjust(wspace=0.3, top=0.88)
    _safe_save(fig, plots_dir / "eval_phys_residual.png")


def plot_params_threshold_all_runs(eval_summary: pd.DataFrame, plots_dir: Path):
    """Stacked/grouped bar: params within 5% and 10% per run."""
    _style()
    run_ids = sorted(eval_summary["RunID"].tolist())
    labels  = [f"Run {r}" for r in run_ids]
    w5  = [int(eval_summary[eval_summary["RunID"]==r]["params_within_5pct"].values[0])
           for r in run_ids]
    w10 = [int(eval_summary[eval_summary["RunID"]==r]["params_within_10pct"].values[0])
           for r in run_ids]
    total = 10  # 10 parameters

    x = np.arange(len(run_ids))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, w5,  0.35, label="Within 5%",  color=_C[1], alpha=0.85)
    ax.bar(x + 0.2, w10, 0.35, label="Within 10%", color=_C[4], alpha=0.85)
    ax.axhline(total, color="white", ls="--", lw=0.8, label=f"Total params ({total})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, total + 1)
    ax.set_ylabel("Number of Parameters"); ax.set_title(
        "Parameters Within Error Threshold — All Runs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="y")
    for j, (v5, v10) in enumerate(zip(w5, w10)):
        ax.text(j - 0.2, v5  + 0.1, str(v5),  ha="center", fontsize=9)
        ax.text(j + 0.2, v10 + 0.1, str(v10), ha="center", fontsize=9)
    fig.subplots_adjust(top=0.90)
    _safe_save(fig, plots_dir / "eval_params_within_threshold.png")


# ──────────────────────────────────────────────────────────────────────────────
# MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────────────────────
def save_evaluation_report(eval_summary: pd.DataFrame, stage2_root: Path):
    run_ids = sorted(eval_summary["RunID"].tolist())
    sig_cols   = [c for c in eval_summary.columns if c.startswith("r2_")]
    nrmse_cols = [c for c in eval_summary.columns if c.startswith("nrmse_")]
    sig_names  = [c.replace("r2_", "") for c in sig_cols]

    lines = [
        "# Stage 1 PINN — Evaluation Report\n",
        f"**Runs evaluated:** {len(run_ids)}  \n",
        "---\n",
        "## Signal Reconstruction (R²)\n",
        "| Run | " + " | ".join(sig_names) + " | Mean R² |",
        "|-----|" + "|".join(["------"] * len(sig_names)) + "|---------|",
    ]
    for rid in run_ids:
        row  = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        vals = [f"{row[c]:.4f}" for c in sig_cols]
        mean_r2 = np.mean([row[c] for c in sig_cols])
        lines.append("| " + " | ".join([str(rid)] + vals + [f"{mean_r2:.4f}"]) + " |")

    lines += [
        "\n## Signal Reconstruction (NRMSE)\n",
        "| Run | " + " | ".join(sig_names) + " | Mean NRMSE |",
        "|-----|" + "|".join(["------"] * len(sig_names)) + "|------------|",
    ]
    for rid in run_ids:
        row  = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        vals = [f"{row[c]:.5f}" for c in nrmse_cols]
        mean_nrmse = np.mean([row[c] for c in nrmse_cols])
        lines.append("| " + " | ".join([str(rid)] + vals + [f"{mean_nrmse:.5f}"]) + " |")

    lines += [
        "\n## Physics Residuals\n",
        "| Run | RMS DOF1 | RMS DOF2 |",
        "|-----|----------|----------|",
    ]
    for rid in run_ids:
        row = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        lines.append(f"| {rid} | {row['phys_rms_dof1']:.4e} | {row['phys_rms_dof2']:.4e} |")

    lines += [
        "\n## Parameter Accuracy\n",
        "| Run | Params within 5% | Params within 10% | Mean param err% |",
        "|-----|-----------------|-------------------|-----------------|",
    ]
    for rid in run_ids:
        row = eval_summary[eval_summary["RunID"] == rid].iloc[0]
        lines.append(
            f"| {rid} | {int(row['params_within_5pct'])}/10 "
            f"| {int(row['params_within_10pct'])}/10 "
            f"| {row['mean_param_error_pct']:.3f}% |"
        )

    report_path = stage2_root / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report → {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CORE: evaluate one run
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_run(run_id: int, paths: dict, df_full: pd.DataFrame) -> Optional[dict]:
    device     = torch.device(CONFIG["training"]["device"])
    run_dir    = paths["runs_dir"] / f"run_{run_id:03d}"
    model_path = run_dir / "model.pt"
    cfg_path   = run_dir / "config.json"
    plots_dir  = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not model_path.exists():
        print(f"  [Run {run_id}] ✗  model.pt not found, skipping.")
        return None

    # ── Load saved config to get init params ─────────────────────────────────
    with open(cfg_path, encoding="utf-8") as f:
        cfg_snap = json.load(f)
    init_params = cfg_snap["init_parameters"]
    true_params = cfg_snap["true_parameters"]

    # ── Rebuild model and load weights ───────────────────────────────────────
    # Stage 1 uses base TwoDOFPINN (raw nn.Parameter masses/stiffnesses)
    # Try base class first; fall back to ScaledTwoDOFPINN if state_dict mismatch
    def _try_load(model_cls):
        m = model_cls(init_params=init_params).to(device)
        sd = torch.load(model_path, map_location=device)
        m.load_state_dict(sd, strict=False)
        return m

    try:
        model = _try_load(TwoDOFPINN)
        # Verify it has the stage1 raw parameters
        assert hasattr(model, "m1") and isinstance(model.m1, torch.nn.Parameter),             "Not a Stage 1 model"
    except Exception:
        print(f"  [Run {run_id}] Falling back to ScaledTwoDOFPINN")
        model = _try_load(ScaledTwoDOFPINN)

    model.eval()
    print(f"  [Run {run_id}] Model loaded ({type(model).__name__}) from {model_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    t_data, F_data, u_meas, v_meas, a_meas, true_par = load_run_data(
        df_full, run_id, device, return_va=True
    )
    n_colloc = int(CONFIG["data"]["n_colloc_factor"] * t_data.shape[0])
    t_phys, F_phys = create_collocation_points(t_data, F_data, n_colloc)
    t_phys = t_phys.to(device)
    F_phys = F_phys.to(device)

    # ── Forward pass for u, v, a ──────────────────────────────────────────────
    t_req  = t_data.clone().detach().requires_grad_(True)
    u_p    = model.forward_u(t_req)
    v_p    = model._d_dt(u_p, t_req)
    a_p    = model._d_dt(v_p, t_req)

    u_p_np = u_p.detach().cpu().numpy()
    v_p_np = v_p.detach().cpu().numpy()
    a_p_np = a_p.detach().cpu().numpy()
    u_m_np = u_meas.detach().cpu().numpy()
    v_m_np = v_meas.detach().cpu().numpy()
    a_m_np = a_meas.detach().cpu().numpy()
    t_np   = t_data.detach().cpu().numpy().squeeze()

    # ── Signal metrics ────────────────────────────────────────────────────────
    sig_rows = (
        signal_metrics(u_m_np, u_p_np, "u")
        + signal_metrics(v_m_np, v_p_np, "v")
        + signal_metrics(a_m_np, a_p_np, "a")
    )
    signal_df = pd.DataFrame(sig_rows)
    signal_df.to_csv(run_dir / "eval_signal_metrics.csv", index=False)

    # ── Physics residuals ─────────────────────────────────────────────────────
    t_phys_req = t_phys.clone().detach().requires_grad_(True)
    res_np     = model.compute_residuals(t_phys_req, F_phys).detach().cpu().numpy()
    t_phys_np  = t_phys.detach().cpu().numpy().squeeze()
    res_rows   = residual_metrics(res_np)
    residual_df = pd.DataFrame(res_rows)
    residual_df.to_csv(run_dir / "eval_residuals.csv", index=False)

    # ── Parameter metrics (from saved metrics.csv) ────────────────────────────
    metrics_row = pd.read_csv(run_dir / "metrics.csv").iloc[0].to_dict()
    par_rows    = param_metrics(metrics_row)
    param_df    = pd.DataFrame(par_rows)
    param_df.to_csv(run_dir / "eval_param_metrics.csv", index=False)

    # ── Modal properties ──────────────────────────────────────────────────────
    with torch.no_grad():
        omega_t, zeta_t = model.modal_properties()
    omega1_pred = float(omega_t[0].item())
    omega2_pred = float(omega_t[1].item())

    # ── Per-run plots ─────────────────────────────────────────────────────────
    plot_signal_metrics(signal_df, run_id, plots_dir / "eval_signal_metrics.png")
    print(f"  [Run {run_id}] → eval_signal_metrics.png")

    plot_residuals(t_phys_np, res_np, residual_df, run_id,
                   plots_dir / "eval_residuals.png")
    print(f"  [Run {run_id}] → eval_residuals.png")

    plot_param_true_vs_pred(param_df, run_id,
                            plots_dir / "eval_param_true_vs_pred.png")
    print(f"  [Run {run_id}] → eval_param_true_vs_pred.png")

    plot_summary_heatmap(param_df, run_id,
                         plots_dir / "eval_summary_heatmap.png")
    print(f"  [Run {run_id}] → eval_summary_heatmap.png")

    plot_response_comparison(
        t_np, u_m_np, v_m_np, a_m_np,
        u_p_np, v_p_np, a_p_np,
        run_id, plots_dir / "eval_response_comparison.png",
    )
    print(f"  [Run {run_id}] → eval_response_comparison.png")

    plot_psd_comparison(
        t_np, u_m_np, u_p_np, run_id,
        plots_dir / "eval_psd_comparison.png",
        omega1_true=true_params.get("omega1"),
        omega2_true=true_params.get("omega2"),
        omega1_pred=omega1_pred,
        omega2_pred=omega2_pred,
    )
    print(f"  [Run {run_id}] → eval_psd_comparison.png")

    # ── Build summary row ─────────────────────────────────────────────────────
    summary = {"RunID": run_id}
    for _, sig_row in signal_df.iterrows():
        s = sig_row["signal"]
        summary[f"r2_{s}"]    = sig_row["R2"]
        summary[f"nrmse_{s}"] = sig_row["NRMSE"]
        summary[f"mse_{s}"]   = sig_row["MSE"]
        summary[f"mae_{s}"]   = sig_row["MAE"]
    summary["mean_r2"]    = signal_df["R2"].mean()
    summary["mean_nrmse"] = signal_df["NRMSE"].mean()
    summary["phys_rms_dof1"] = float(residual_df.iloc[0]["rms_residual"])
    summary["phys_rms_dof2"] = float(residual_df.iloc[1]["rms_residual"])
    summary["params_within_5pct"]  = int(param_df["within_5pct"].sum())
    summary["params_within_10pct"] = int(param_df["within_10pct"].sum())
    summary["mean_param_error_pct"] = float(param_df["pct_error"].mean())

    # Print per-run summary
    print(f"\n  ── Run {run_id} Evaluation Summary ──")
    print(f"     Mean R²    : {summary['mean_r2']:.5f}")
    print(f"     Mean NRMSE : {summary['mean_nrmse']:.5f}")
    print(f"     Params <5% : {summary['params_within_5pct']}/10")
    print(f"     Params <10%: {summary['params_within_10pct']}/10")
    print(f"     Mean param err%: {summary['mean_param_error_pct']:.3f}%")
    print(f"     Physics RMS: DOF1={summary['phys_rms_dof1']:.3e}  DOF2={summary['phys_rms_dof2']:.3e}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# PATHS HELPER
# ──────────────────────────────────────────────────────────────────────────────
def get_stage1_paths(project_root: Path) -> dict:
    stage1_root = project_root / "results" / "mdof_pinn" / "stage1"
    return {
        "stage1_root": stage1_root,
        "runs_dir":    stage1_root / "runs",
        "plots_dir":   stage1_root / "plots",
        "eval_csv":    stage1_root / "evaluation_summary.csv",
    }

# backward compat alias
def get_stage2_paths(project_root: Path) -> dict:
    return get_stage1_paths(project_root)

def get_stage2_paths_UNUSED(project_root: Path) -> dict:
    stage2_root = project_root / "results" / "mdof_pinn" / "stage2_fixed"
    return {
        "stage1_root": stage2_root,
        "runs_dir":    stage2_root / "runs",
        "plots_dir":   stage2_root / "plots",
        "eval_csv":    stage2_root / "evaluation_summary.csv",
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for Stage 1 Baseline 2DOF PINN"
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=5)
    parser.add_argument("--run",   type=int, default=None,
                        help="Evaluate a single run (overrides --start/--end)")
    args = parser.parse_args()

    paths = get_stage1_paths(project_root)
    paths["plots_dir"].mkdir(parents=True, exist_ok=True)

    df_full = load_full_csv(CONFIG["data"]["csv_path"])

    run_list = [args.run] if args.run else list(range(args.start, args.end + 1))

    print("=" * 80)
    print("STAGE 1 EVALUATION — Post-training analysis")
    print("=" * 80)
    print(f"Runs     : {run_list}")
    print(f"Results  : {paths['stage1_root']}")
    print("=" * 80)

    all_summaries = []
    for run_id in run_list:
        print(f"\n── Evaluating RunID {run_id} " + "─" * 50)
        summary = evaluate_run(run_id, paths, df_full)
        if summary:
            all_summaries.append(summary)

    if not all_summaries:
        print("\nNo runs evaluated successfully."); return 1

    # ── Global summary CSV ────────────────────────────────────────────────────
    eval_df = pd.DataFrame(all_summaries).sort_values("RunID").reset_index(drop=True)
    eval_df.to_csv(paths["eval_csv"], index=False)
    print(f"\n  Global summary → {paths['eval_csv']}")

    # ── Multi-run dashboard plots ─────────────────────────────────────────────
    if len(all_summaries) > 1:
        print("\n[Dashboard] Generating multi-run evaluation plots...")
        plot_r2_all_runs(eval_df, paths["plots_dir"])
        plot_nrmse_all_runs(eval_df, paths["plots_dir"])
        plot_phys_residual_all_runs(eval_df, paths["plots_dir"])
        plot_params_threshold_all_runs(eval_df, paths["plots_dir"])
        print(f"  Dashboard → {paths['plots_dir']}")

    # ── Markdown report ───────────────────────────────────────────────────────
    save_evaluation_report(eval_df, paths["stage1_root"])

    print("\n" + "=" * 80)
    print("STAGE 1 EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  Per-run outputs : runs/run_XXX/eval_*.csv + plots/eval_*.png")
    print(f"  Global summary  : {paths['eval_csv']}")
    print(f"  Dashboard plots : {paths['plots_dir']}")
    print(f"  Report          : {paths['stage1_root']}/evaluation_report.md")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
