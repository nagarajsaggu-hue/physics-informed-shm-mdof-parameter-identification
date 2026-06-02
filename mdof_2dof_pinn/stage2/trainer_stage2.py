"""
trainer_stage2.py — Fully-blind 2DOF PINN (Stage 2, F2≠0)

Key features:
- Fully blind inverse PINN: no modal supervision, no parameter supervision
- Warm-up on data loss only
- Smooth sigmoid ramp for physics loss
- Scaled m/k parameters for better conditioning
- Adam + optional LBFGS refinement
- Residual normalization for stabler physics training
- Saves into results/mdof_pinn/stage2_fixed/ to preserve old Stage-2 outputs
- NEW: Full training history plots, parameter comparison charts, R²/accuracy
       summary report, and a combined multi-run dashboard generated at the end.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on clusters)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# PATH SETUP

current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from mdof_2dof_pinn.data_mdof import load_full_csv, load_run_data, create_collocation_points
    from mdof_2dof_pinn.model_mdof import TwoDOFPINN
except ImportError:
    print("Critical Error: Could not import 'mdof_2dof_pinn'.")
    sys.exit(1)


# CONFIGURATION

CONFIG = {
    "data": {
        "csv_path": "Data/mdof_2dof_216runs_pinn.csv",
        "n_colloc_factor": 4.0,
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": 4000,
        "print_every": 500,
        "seed_base": 2000,
        "warm_epochs": 800,
        "ramp_epochs": 1600,
        "use_lbfgs": True,
        "lbfgs_max_iter": 600,
        "lbfgs_lr": 0.5,
        "grad_clip_norm": 5.0,
    },
    "loss": {
        "w_u": 2.0,
        "w_v": 0.5,
        "w_a": 1.5,
        "w_phys": 2.0,
        "w_modal": 0.0,
        "w_zeta": 0.0,
        "w_param": 0.0,
        "lambda_scale_prior": 5e-3,
        "perturb_level": 0.10,
        "eps_norm": 1e-12,
    },
    "opt": {
        "lr_nn": 1e-3,
        "lr_k": 1e-3,
        "lr_m": 5e-4,
        "lr_damp": 5e-4,
    },
    "sched": {
        "factor": 0.5,
        "patience": 300,
    }
}


# UTILITY CLASSES AND FUNCTIONS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ScaledTwoDOFPINN(TwoDOFPINN):
    """TwoDOFPINN with scale parameters instead of raw m/k values."""

    def __init__(self, init_params: Dict[str, float]):
        super().__init__(init_params)
        for name in ["m1", "m2", "k1", "k2"]:
            if name in self._parameters:  del self._parameters[name]
            if hasattr(self, name):        delattr(self, name)

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


def perturb_params(true_p: Dict[str, float], level: float) -> Dict[str, float]:
    targets = {"m1", "m2", "k1", "k2", "alpha", "beta"}
    out = {}
    for key, value in true_p.items():
        if key in targets:
            out[key] = value * (1.0 + (np.random.rand() * 2.0 - 1.0) * level)
        else:
            out[key] = value
    return out


def _get_damp_params(model: nn.Module) -> List[nn.Parameter]:
    if hasattr(model, "alpha_log") and hasattr(model, "beta_log"):
        return [model.alpha_log, model.beta_log]
    return [model.alpha, model.beta]


def _alpha_beta_vals(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "alpha_val") and hasattr(model, "beta_val"):
        return model.alpha_val(), model.beta_val()
    return model.alpha, model.beta


def _clamp_scales(model: nn.Module, lo: float = 0.5, hi: float = 2.0) -> None:
    with torch.no_grad():
        model.m1_scale.clamp_(min=lo, max=hi)
        model.m2_scale.clamp_(min=lo, max=hi)
        model.k1_scale.clamp_(min=lo, max=hi)
        model.k2_scale.clamp_(min=lo, max=hi)


def _safe_pct_err(pred: float, true: float) -> float:
    return abs(pred - true) / (abs(true) + 1e-12) * 100.0


def _sigmoid_ramp(epoch: int, warm_epochs: int, ramp_epochs: int) -> float:
    x = (epoch - warm_epochs) / max(1.0, float(ramp_epochs))
    z = 6.0 * x - 3.0
    return float(torch.sigmoid(torch.tensor(z)).item())


def _normalized_residual_mse(residual: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.abs(residual).mean().detach() + eps
    return torch.mean((residual / denom) ** 2)


def get_stage2_paths(project_root: Path) -> Dict[str, Path]:
    stage2_root = project_root / "results" / "mdof_pinn" / "stage2_fixed"
    return {
        "stage2_root": stage2_root,
        "runs_dir":    stage2_root / "runs",
        "logs_dir":    stage2_root / "logs",
        "summary_csv": stage2_root / "identified_parameters.csv",
        "plots_dir":   stage2_root / "plots",
    }


def get_completed_runs(runs_dir: Path) -> Set[int]:
    completed = set()
    if not runs_dir.exists():
        return completed
    for run_folder in runs_dir.glob("run_*"):
        if run_folder.is_dir() and (run_folder / "model.pt").exists() and (run_folder / "metrics.csv").exists():
            try:
                completed.add(int(run_folder.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return completed


# PLOTTING UTILITIES  ← NEW SECTION

PARAM_META = {
    # key : (display_label, unit)
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


def compute_r2(true_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)



# ── Per-run: Displacement / Velocity / Acceleration time-series ───────────────
def plot_response_comparison(
    t_np: np.ndarray,
    u_meas_np: np.ndarray, v_meas_np: np.ndarray, a_meas_np: np.ndarray,
    u_pred_np: np.ndarray, v_pred_np: np.ndarray, a_pred_np: np.ndarray,
    run_id: int, save_path: Path,
) -> None:
    """
    2-column × 3-row figure:
      Col 1 = DOF 1  |  Col 2 = DOF 2
      Row 1 = Displacement (u)
      Row 2 = Velocity     (v)
      Row 3 = Acceleration (a)
    Each subplot overlays measured (dashed) vs PINN-predicted (solid).
    """
    _style()
    dof_labels = ["DOF 1", "DOF 2"]
    row_labels  = ["Displacement (m)", "Velocity (m/s)", "Accel (m/s²)"]
    meas_data   = [u_meas_np, v_meas_np, a_meas_np]
    pred_data   = [u_pred_np, v_pred_np, a_pred_np]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Response Comparison (Measured vs PINN) — Run {run_id}",
                 fontsize=14, fontweight="bold")

    for row, (rlbl, meas, pred) in enumerate(zip(row_labels, meas_data, pred_data)):
        for col in range(2):
            ax = axes[row, col]
            m = meas[:, col] if meas.ndim == 2 else meas
            p = pred[:, col] if pred.ndim == 2 else pred
            ax.plot(t_np, m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.8)
            ax.plot(t_np, p, color=_C[0], lw=1.0, label="PINN pred")
            ax.set_ylabel(rlbl, fontsize=8)
            if row == 0:
                ax.set_title(dof_labels[col], fontsize=10, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Per-run: Power Spectral Density (PSD) ────────────────────────────────────
def plot_psd_comparison(
    t_np: np.ndarray,
    u_meas_np: np.ndarray, u_pred_np: np.ndarray,
    run_id: int, save_path: Path,
    omega1_true: float = None, omega2_true: float = None,
    omega1_pred: float = None, omega2_pred: float = None,
) -> None:
    """
    PSD (Welch) of measured vs predicted displacement for DOF 1 and DOF 2.
    Natural frequencies are marked as vertical lines when provided.
    Frequency axis is in Hz (ω/2π marked separately if needed).
    """
    from scipy.signal import welch

    _style()
    dt = float(np.mean(np.diff(t_np)))
    fs = 1.0 / dt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Power Spectral Density (Displacement) — Run {run_id}",
                 fontsize=13, fontweight="bold")

    dof_labels = ["DOF 1", "DOF 2"]
    for col in range(2):
        ax = axes[col]
        m = u_meas_np[:, col] if u_meas_np.ndim == 2 else u_meas_np
        p = u_pred_np[:, col] if u_pred_np.ndim == 2 else u_pred_np

        f_m, psd_m = welch(m, fs=fs, nperseg=min(256, len(m) // 4))
        f_p, psd_p = welch(p, fs=fs, nperseg=min(256, len(p) // 4))

        ax.semilogy(f_m, psd_m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.85)
        ax.semilogy(f_p, psd_p, color=_C[0], lw=1.0, label="PINN pred")

        # Mark natural frequencies (convert rad/s → Hz)
        if omega1_true and col == 0:
            ax.axvline(omega1_true / (2 * np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₁ true ({omega1_true/(2*np.pi):.2f} Hz)")
        if omega1_pred and col == 0:
            ax.axvline(omega1_pred / (2 * np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₁ pred ({omega1_pred/(2*np.pi):.2f} Hz)")
        if omega2_true and col == 1:
            ax.axvline(omega2_true / (2 * np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₂ true ({omega2_true/(2*np.pi):.2f} Hz)")
        if omega2_pred and col == 1:
            ax.axvline(omega2_pred / (2 * np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₂ pred ({omega2_pred/(2*np.pi):.2f} Hz)")

        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("PSD (m²/Hz)", fontsize=9)
        ax.set_title(dof_labels[col], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.subplots_adjust(wspace=0.3, top=0.88)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Per-run: Detailed training history + parameter convergence ────────────────
def plot_training_history_detailed(
    history_df: pd.DataFrame,
    run_id: int,
    warm_epochs: int,
    true_par: dict,
    init_par: dict,
    param_history: list,   # list of dicts: {epoch, m1, m2, k1, k2, alpha, beta}
    save_path: Path,
) -> None:
    """
    4-row figure:
      Row 1 — Total loss (log)
      Row 2 — Data vs Physics loss (log)
      Row 3 — Physics ramp weight + LR schedule proxy (total-loss smoothed)
      Row 4 — Parameter % error convergence: m1, m2, k1, k2 vs epoch
    """
    _style()
    ep = history_df["epoch"].values

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(f"Detailed Training History — Run {run_id}",
                 fontsize=14, fontweight="bold")

    # Row 1 – total loss
    ax = axes[0]
    ax.semilogy(ep, history_df["loss_total"].values, color=_C[0], lw=1.4, label="Total loss")
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1, label=f"Warm-up end ({warm_epochs})")
    ax.set_ylabel("Loss (log)"); ax.set_title("Total Loss"); ax.legend(fontsize=8); ax.grid(True)

    # Row 2 – data vs physics
    ax = axes[1]
    ax.semilogy(ep, history_df["loss_data"].values, color=_C[1], lw=1.4, label="Data loss")
    ax.semilogy(ep, history_df["loss_phys"].values, color=_C[2], lw=1.4, label="Physics loss")
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1)
    ax.set_ylabel("Loss (log)"); ax.set_title("Data Loss vs Physics Loss")
    ax.legend(fontsize=8); ax.grid(True)

    # Row 3 – ramp + smoothed total loss as LR proxy
    ax = axes[2]
    ax.plot(ep, history_df["ramp"].values, color=_C[4], lw=1.4, label="Physics ramp")
    ax2 = ax.twinx()
    # smooth total loss with rolling window
    smooth = pd.Series(history_df["loss_total"].values).rolling(50, min_periods=1).mean().values
    ax2.semilogy(ep, smooth, color=_C[8] if len(_C) > 8 else _C[3],
                 lw=0.8, alpha=0.5, label="Smoothed total (loss proxy)")
    ax2.set_ylabel("Smoothed Loss", fontsize=8)
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1)
    ax.set_ylabel("Ramp weight"); ax.set_title("Physics Ramp & Smoothed Loss Convergence")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="center right")
    ax.grid(True)

    # Row 4 – parameter % error convergence
    ax = axes[3]
    if param_history:
        ph_df = pd.DataFrame(param_history)
        phys_params = [("m1", _C[0]), ("m2", _C[1]), ("k1", _C[2]), ("k2", _C[4])]
        for pkey, col in phys_params:
            if pkey in ph_df.columns and pkey in true_par:
                err_series = np.abs(ph_df[pkey].values - true_par[pkey]) / (abs(true_par[pkey]) + 1e-12) * 100
                ax.plot(ph_df["epoch"].values, err_series, color=col, lw=1.2, label=f"{pkey} err%")
        ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1)
        ax.axhline(5,  color="white", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(10, color=_C[2],   ls=":", lw=0.8, alpha=0.5)
        ax.set_ylabel("% Error"); ax.set_title("Physical Parameter Convergence (m1, m2, k1, k2)")
        ax.legend(fontsize=8, ncol=4); ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Parameter history not recorded",
                ha="center", va="center", transform=ax.transAxes, color="#8b949e")

    ax.set_xlabel("Epoch", fontsize=9)
    fig.subplots_adjust(hspace=0.38, top=0.94)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Per-run: training history ─────────────────────────────────────────────────
def plot_training_history(history_df: pd.DataFrame, run_id: int,
                          warm_epochs: int, save_path: Path) -> None:
    """
    3-panel figure showing total loss, data/physics losses, and ramp weight
    vs epoch. Both loss panels use a log scale.
    """
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Training History — Run {run_id}",
                 fontsize=14, fontweight="bold")

    ep = history_df["epoch"].values

    ax = axes[0]
    ax.semilogy(ep, history_df["loss_total"].values, color=_C[0], lw=1.5, label="Total")
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1, label=f"Warmup end ({warm_epochs})")
    ax.set_ylabel("Loss (log)"); ax.set_title("Total Loss"); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1]
    ax.semilogy(ep, history_df["loss_data"].values, color=_C[1], lw=1.5, label="Data loss")
    ax.semilogy(ep, history_df["loss_phys"].values, color=_C[2], lw=1.5, label="Physics loss")
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1)
    ax.set_ylabel("Loss (log)"); ax.set_title("Data vs Physics Loss")
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[2]
    ax.plot(ep, history_df["ramp"].values, color=_C[4], lw=1.5)
    ax.axvline(warm_epochs, color=_C[2], ls="--", lw=1)
    ax.set_ylabel("Ramp weight"); ax.set_xlabel("Epoch")
    ax.set_title("Physics Ramp (Sigmoid)"); ax.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── Per-run: True / Init / Predicted bar + % error horizontal bar ─────────────
def plot_parameter_comparison(results: Dict, run_id: int, save_path: Path) -> None:
    """
    Row 1 — Grouped bar chart: True | Init | Predicted (normalised to True=1).
    Row 2 — Horizontal % error bar, colour-coded: green<5%, orange 5-10%, red>10%.
    """
    _style()
    params = list(PARAM_META.keys())
    labels = [PARAM_META[p][0] for p in params]

    true_n = np.ones(len(params))
    init_n = np.array([results.get(f"{p}_init", results[f"{p}_true"]) / results[f"{p}_true"]
                       for p in params])
    pred_n = np.array([results[f"{p}_pred"] / results[f"{p}_true"] for p in params])
    errors = np.array([results[f"{p}_error_pct"] for p in params])

    x, w = np.arange(len(params)), 0.25
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"Parameter Identification — Run {run_id}",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(x - w, true_n, w, label="True",      color=_C[1], alpha=0.85)
    ax.bar(x,     init_n, w, label="Init",      color=_C[4], alpha=0.85)
    ax.bar(x + w, pred_n, w, label="Predicted", color=_C[0], alpha=0.85)
    ax.axhline(1.0, color=_C[2], ls="--", lw=0.8, label="True = 1")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value / True Value")
    ax.set_title("True | Init | Predicted  (normalised to True = 1)")
    ax.legend(fontsize=8); ax.grid(True, axis="y")

    ax = axes[1]
    bar_colors = [_C[1] if e < 5 else _C[4] if e < 10 else _C[2] for e in errors]
    ax.barh(labels, errors, color=bar_colors, edgecolor="#30363d")
    ax.axvline(5,  color=_C[4], ls="--", lw=1, label="5% threshold")
    ax.axvline(10, color=_C[2], ls="--", lw=1, label="10% threshold")
    for i, v in enumerate(errors):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("% Error")
    ax.set_title("Prediction Error  (green < 5% | orange 5-10% | red > 10%)")
    ax.legend(fontsize=8); ax.grid(True, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── Per-run: true-vs-predicted scatter ───────────────────────────────────────
def plot_true_vs_pred_scatter(results: Dict, run_id: int, save_path: Path) -> None:
    """
    Scatter of normalised predicted vs true for all 10 parameters.
    Perfect identification → all points on the y=x diagonal.
    """
    _style()
    params = list(PARAM_META.keys())
    labels = [PARAM_META[p][0] for p in params]
    pred_n = np.array([results[f"{p}_pred"] / results[f"{p}_true"] for p in params])

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f"True vs Predicted (normalised) — Run {run_id}",
                 fontsize=13, fontweight="bold")

    for i, lbl in enumerate(labels):
        ax.scatter(1.0, pred_n[i], s=90, color=_C[i % len(_C)], zorder=5)
        ax.annotate(lbl, (1.0, pred_n[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    lim_lo = min(pred_n.min() - 0.03, 0.85)
    lim_hi = max(pred_n.max() + 0.03, 1.15)
    diag = np.linspace(lim_lo, lim_hi, 100)
    ax.plot(diag, diag, "--", color=_C[2], lw=1.2, label="Perfect prediction")
    ax.set_xlabel("True (normalised = 1)"); ax.set_ylabel("Predicted (normalised)")
    ax.set_xlim(0.88, 1.12); ax.set_ylim(lim_lo, lim_hi)
    ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)



# ── Multi-run dashboard (called after all runs finish) ────────────────────────
def plot_multi_run_dashboard(summary_df: pd.DataFrame,
                             plots_dir: Path, warm_epochs: int) -> None:
    """
    Generates 4 cross-run charts saved to plots_dir:
      dashboard_error_grouped.png  — grouped bar: error% per param per run
      dashboard_mean_error.png     — mean ± std error per param (sorted)
      dashboard_r2.png             — R² per parameter
      dashboard_per_run.png        — mean error & accuracy per run
    """
    _style()
    params  = list(PARAM_META.keys())
    labels  = [PARAM_META[p][0] for p in params]
    run_ids = sorted(summary_df["RunID"].tolist())
    n_runs  = len(run_ids)

    def _safe_save(fig, path):
        """Clamp figure to minimum 4×3 inches before saving to avoid Agg crash."""
        w, h = fig.get_size_inches()
        if w < 4 or h < 3:
            fig.set_size_inches(max(w, 4), max(h, 3))
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"  WARNING: could not save {path.name}: {e}")
        plt.close(fig)

    # (a) Grouped bar ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(params))
    w = 0.8 / n_runs
    for i, rid in enumerate(run_ids):
        row    = summary_df[summary_df["RunID"] == rid].iloc[0]
        errors = [row[f"{p}_error_pct"] for p in params]
        offset = (i - n_runs / 2.0 + 0.5) * w
        ax.bar(x + offset, errors, w, label=f"Run {rid}",
               color=_C[i % len(_C)], alpha=0.85)
    ax.axhline(5,  color="white", ls="--", lw=0.8, alpha=0.5, label="5% line")
    ax.axhline(10, color=_C[2],  ls="--", lw=0.8, alpha=0.5, label="10% line")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% Error")
    ax.set_title("Parameter Prediction Error (%) — All Runs",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=min(n_runs, 6))
    ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12)          # give x-labels room
    _safe_save(fig, plots_dir / "dashboard_error_grouped.png")

    # (b) Mean ± std ──────────────────────────────────────────────────────────
    means = np.array([summary_df[f"{p}_error_pct"].mean() for p in params])
    stds  = np.array([summary_df[f"{p}_error_pct"].std()  for p in params])
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [_C[1] if means[i] < 5 else _C[4] if means[i] < 10 else _C[2]
                  for i in order]
    ax.bar(np.arange(len(params)), means[order], color=bar_colors, alpha=0.85,
           yerr=stds[order], capsize=4,
           error_kw=dict(color="#c9d1d9", lw=1.2))
    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels([labels[i] for i in order], fontsize=10)
    ax.set_ylabel("Mean % Error ± Std")
    ax.set_title("Mean Prediction Error per Parameter (sorted)",
                 fontsize=12, fontweight="bold")
    ax.axhline(5,  color="white", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(10, color=_C[2],  ls="--", lw=0.8, alpha=0.5)
    for j, (m, s) in enumerate(zip(means[order], stds[order])):
        ax.text(j, m + s + 0.2, f"{m:.1f}%", ha="center", fontsize=7.5)
    ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12, top=0.88)
    _safe_save(fig, plots_dir / "dashboard_mean_error.png")

    # (c) R² per parameter ────────────────────────────────────────────────────
    r2_vals = np.array([
        compute_r2(
            summary_df[f"{p}_true"].values.astype(float),
            summary_df[f"{p}_pred"].values.astype(float),
        )
        for p in params
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = [_C[1] if r > 0.98 else _C[4] if r > 0.95 else _C[2]
                  for r in r2_vals]
    ax.bar(np.arange(len(params)), r2_vals, color=bar_colors,
           alpha=0.85, edgecolor="#30363d")
    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(1.0,  color=_C[2],  ls="--", lw=0.8, label="Perfect R²=1")
    ax.axhline(0.99, color="white", ls=":",  lw=0.8, alpha=0.5, label="R²=0.99")
    lo = max(float(r2_vals.min()) - 0.05, 0.0)
    ax.set_ylim(lo, 1.05)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score per Parameter (across all runs)",
                 fontsize=12, fontweight="bold")
    for j, r in enumerate(r2_vals):
        ax.text(j, float(r) + 0.003, f"{r:.4f}", ha="center", fontsize=7.5)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12, top=0.88)
    _safe_save(fig, plots_dir / "dashboard_r2.png")

    # (d) Per-run mean error & accuracy ───────────────────────────────────────
    run_mean_errors = [
        float(summary_df[summary_df["RunID"] == rid][[f"{p}_error_pct" for p in params]].values.mean())
        for rid in run_ids
    ]
    run_accuracy = [100.0 - e for e in run_mean_errors]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per-Run Summary Statistics", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar([f"Run {r}" for r in run_ids], run_mean_errors,
           color=[_C[i % len(_C)] for i in range(n_runs)], alpha=0.85)
    ax.set_ylabel("Mean % Error (all params)")
    ax.set_title("Mean Error per Run")
    ax.grid(True, axis="y")
    for j, v in enumerate(run_mean_errors):
        ax.text(j, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8)

    ax = axes[1]
    ax.bar([f"Run {r}" for r in run_ids], run_accuracy,
           color=[_C[i % len(_C)] for i in range(n_runs)], alpha=0.85)
    lo_acc = max(min(run_accuracy) - 2, 85)
    ax.set_ylim(lo_acc, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per Run  (100 − mean error %)")
    ax.grid(True, axis="y")
    for j, v in enumerate(run_accuracy):
        ax.text(j, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8)

    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.3)
    _safe_save(fig, plots_dir / "dashboard_per_run.png")

    print(f"  Dashboard plots saved → {plots_dir}")


# ── Markdown training report ──────────────────────────────────────────────────
def save_training_report(summary_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    """
    Write results/mdof_pinn/stage2_fixed/training_report.md
    Contains: parameter-level table (mean/std/min/max/R²) + per-run table.
    """
    params  = list(PARAM_META.keys())
    labels  = {p: PARAM_META[p][0] for p in params}
    run_ids = sorted(summary_df["RunID"].tolist())

    lines = [
        "# Stage 2 PINN — Training Report\n",
        f"**Runs completed:** {len(run_ids)}  ",
        f"**Epochs (Adam):** {CONFIG['training']['num_epochs']}  ",
        f"**Warm-up epochs:** {CONFIG['training']['warm_epochs']}  ",
        f"**LBFGS refinement:** {CONFIG['training']['use_lbfgs']}  \n",
        "---\n",
        "## Parameter-Level Summary\n",
        "| Parameter | Mean Err% | Std% | Min% | Max% | R² |",
        "|-----------|-----------|------|------|------|----|",
    ]
    for p in params:
        errs = summary_df[f"{p}_error_pct"].values.astype(float)
        r2   = compute_r2(
            summary_df[f"{p}_true"].values.astype(float),
            summary_df[f"{p}_pred"].values.astype(float),
        )
        lines.append(
            f"| {labels[p]} | {errs.mean():.3f} | {errs.std():.3f} | "
            f"{errs.min():.3f} | {errs.max():.3f} | {r2:.5f} |"
        )

    lines += [
        "\n---\n",
        "## Per-Run Results\n",
        "| Run | " + " | ".join([labels[p] + " err%" for p in params]) + " | Mean err% | Accuracy% |",
        "|-----|" + "|".join(["------" for _ in params]) + "|-----------|-----------|",
    ]
    for rid in run_ids:
        row      = summary_df[summary_df["RunID"] == rid].iloc[0]
        err_vals = [row[f"{p}_error_pct"] for p in params]
        mean_e   = np.mean(err_vals)
        lines.append(
            "| " + " | ".join([str(rid)] + [f"{e:.2f}" for e in err_vals]
                               + [f"{mean_e:.2f}", f"{100-mean_e:.2f}"]) + " |"
        )

    lines += [
        "\n---\n",
        "## Configuration Snapshot\n",
        f"```json\n{json.dumps(CONFIG, indent=2)}\n```",
    ]

    report_path = paths["stage2_root"] / "training_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved → {report_path}")



# TRAINING FUNCTION

def train_stage2_run(run_id: int, cfg: Dict, paths: Dict[str, Path],
                     df: pd.DataFrame) -> bool:
    device = torch.device(cfg["training"]["device"])
    seed   = int(cfg["training"]["seed_base"]) + int(run_id)
    seed_all(seed)

    print(f"\n[Stage 2] RunID {run_id} | device={device}")

    run_dir = paths["runs_dir"] / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Load data ────────────────────────────────────────────────────────
        t_data, F_data, u_meas, v_meas, a_meas, true_par = load_run_data(
            df, run_id, device, return_va=True
        )

        n_colloc = int(cfg["data"]["n_colloc_factor"] * t_data.shape[0])
        t_phys, F_phys = create_collocation_points(t_data, F_data, n_colloc)
        t_phys = t_phys.to(device)
        F_phys = F_phys.to(device)

        # ── Model ────────────────────────────────────────────────────────────
        init_par = perturb_params(true_par, float(cfg["loss"]["perturb_level"]))
        print(
            f"  Init: m1={init_par['m1']:.0f} m2={init_par['m2']:.0f} "
            f"k1={init_par['k1']:.0f} k2={init_par['k2']:.0f}"
        )

        model = ScaledTwoDOFPINN(init_params=init_par).to(device)

        damp_params = _get_damp_params(model)
        nn_params   = list(model.hidden.parameters()) + list(model.out.parameters())

        optimizer = torch.optim.Adam([
            {"params": nn_params,                         "lr": float(cfg["opt"]["lr_nn"])},
            {"params": [model.k1_scale, model.k2_scale], "lr": float(cfg["opt"]["lr_k"])},
            {"params": [model.m1_scale, model.m2_scale], "lr": float(cfg["opt"]["lr_m"])},
            {"params": damp_params,                       "lr": float(cfg["opt"]["lr_damp"])},
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(cfg["sched"]["factor"]),
            patience=int(cfg["sched"]["patience"]),
        )

        epochs         = int(cfg["training"]["num_epochs"])
        warm_epochs    = int(cfg["training"]["warm_epochs"])
        ramp_epochs    = int(cfg["training"]["ramp_epochs"])
        print_every    = int(cfg["training"]["print_every"])
        grad_clip_norm = float(cfg["training"]["grad_clip_norm"])
        eps_norm           = float(cfg["loss"]["eps_norm"])
        w_u                = float(cfg["loss"]["w_u"])
        w_v                = float(cfg["loss"]["w_v"])
        w_a                = float(cfg["loss"]["w_a"])
        w_phys             = float(cfg["loss"]["w_phys"])
        lambda_scale_prior = float(cfg["loss"]["lambda_scale_prior"])

        if warm_epochs >= epochs:
            print(f"  WARNING: warm_epochs({warm_epochs}) >= epochs({epochs}), adjusting.")
            warm_epochs = epochs // 2

        # Freeze physical params during warm-up
        for p in [model.k1_scale, model.k2_scale, model.m1_scale, model.m2_scale] + damp_params:
            p.requires_grad_(False)

        model.train()
        history: List[Dict] = []
        param_history: List[Dict] = []
        Ldata0: Optional[torch.Tensor] = None
        Lphys0: Optional[torch.Tensor] = None

        # ── Adam loop ────────────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)

            t_req  = t_data.clone().detach().requires_grad_(True)
            u_pred = model.forward_u(t_req)
            v_pred = model._d_dt(u_pred, t_req)
            a_pred = model._d_dt(v_pred, t_req)

            loss_data = (
                w_u * torch.mean((u_pred - u_meas) ** 2)
                + w_v * torch.mean((v_pred - v_meas) ** 2)
                + w_a * torch.mean((a_pred - a_meas) ** 2)
            )

            t_phys_req = t_phys.clone().detach().requires_grad_(True)
            residual   = model.compute_residuals(t_phys_req, F_phys)
            loss_phys  = _normalized_residual_mse(residual)

            if epoch == warm_epochs + 1:
                print(f"  Unfreezing parameters at epoch {epoch}")
                for p in [model.k1_scale, model.k2_scale, model.m1_scale, model.m2_scale] + damp_params:
                    p.requires_grad_(True)
                Ldata0 = loss_data.detach().clamp_min(eps_norm)
                Lphys0 = loss_phys.detach().clamp_min(eps_norm)
                print(f"  Baselines: data={Ldata0.item():.2e} phys={Lphys0.item():.2e}")

            if epoch <= warm_epochs:
                ramp = 0.0
                scale_prior = torch.zeros((), device=device)
                loss = loss_data
            else:
                if Ldata0 is None or Lphys0 is None:
                    raise RuntimeError("Normalization baselines not set.")
                ramp = _sigmoid_ramp(epoch, warm_epochs, ramp_epochs)
                scale_prior = (
                    (model.k1_scale - 1.0) ** 2 + (model.k2_scale - 1.0) ** 2
                    + (model.m1_scale - 1.0) ** 2 + (model.m2_scale - 1.0) ** 2
                )
                loss = (
                    (loss_data / Ldata0)
                    + (w_phys * ramp * (loss_phys / Lphys0))
                    + (lambda_scale_prior * scale_prior)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step(loss.item())
            _clamp_scales(model, lo=0.5, hi=2.0)

            history.append({
                "epoch":      epoch,
                "loss_total": float(loss.item()),
                "loss_data":  float(loss_data.item()),
                "loss_phys":  float(loss_phys.item()),
                "ramp":       float(ramp),
            })

            param_history.append({
                "epoch": epoch,
                "m1":    float(model.m1.item()),
                "m2":    float(model.m2.item()),
                "k1":    float(model.k1.item()),
                "k2":    float(model.k2.item()),
            })

            if epoch % print_every == 0 or epoch == epochs:
                print(
                    f"  Ep {epoch:5d} | total={loss.item():.3e} "
                    f"data={loss_data.item():.3e} phys={loss_phys.item():.3e} "
                    f"k1={_safe_pct_err(float(model.k1.item()), true_par['k1']):.1f}% "
                    f"k2={_safe_pct_err(float(model.k2.item()), true_par['k2']):.1f}% "
                    f"m1={_safe_pct_err(float(model.m1.item()), true_par['m1']):.1f}% "
                    f"m2={_safe_pct_err(float(model.m2.item()), true_par['m2']):.1f}%"
                )

        # ── Fallback baselines ────────────────────────────────────────────────
        if Ldata0 is None or Lphys0 is None:
            print("  Baselines not set during Adam; computing now...")
            t_req  = t_data.clone().detach().requires_grad_(True)
            u_p    = model.forward_u(t_req)
            v_p    = model._d_dt(u_p, t_req)
            a_p    = model._d_dt(v_p, t_req)
            ld_tmp = (w_u * torch.mean((u_p - u_meas) ** 2)
                      + w_v * torch.mean((v_p - v_meas) ** 2)
                      + w_a * torch.mean((a_p - a_meas) ** 2))
            t_phys_req = t_phys.clone().detach().requires_grad_(True)
            lp_tmp = _normalized_residual_mse(model.compute_residuals(t_phys_req, F_phys))
            Ldata0 = ld_tmp.detach().clamp_min(eps_norm)
            Lphys0 = lp_tmp.detach().clamp_min(eps_norm)

        # ── LBFGS refinement ─────────────────────────────────────────────────
        if bool(cfg["training"]["use_lbfgs"]):
            print("  LBFGS refinement...")
            for p in model.parameters():
                p.requires_grad_(True)

            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=float(cfg["training"]["lbfgs_lr"]),
                max_iter=int(cfg["training"]["lbfgs_max_iter"]),
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                t_req = t_data.clone().detach().requires_grad_(True)
                u_p   = model.forward_u(t_req)
                v_p   = model._d_dt(u_p, t_req)
                a_p   = model._d_dt(v_p, t_req)
                ld    = (w_u * torch.mean((u_p - u_meas) ** 2)
                         + w_v * torch.mean((v_p - v_meas) ** 2)
                         + w_a * torch.mean((a_p - a_meas) ** 2))
                t_phys_req = t_phys.clone().detach().requires_grad_(True)
                lp = _normalized_residual_mse(model.compute_residuals(t_phys_req, F_phys))
                sp = ((model.k1_scale - 1.0) ** 2 + (model.k2_scale - 1.0) ** 2
                      + (model.m1_scale - 1.0) ** 2 + (model.m2_scale - 1.0) ** 2)
                total = (ld / Ldata0) + w_phys * (lp / Lphys0) + lambda_scale_prior * sp
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                return total

            lbfgs.step(closure)
            _clamp_scales(model, lo=0.5, hi=2.0)

        # ── Final metrics ─────────────────────────────────────────────────────
        model.eval()
        alpha_p, beta_p = _alpha_beta_vals(model)
        with torch.no_grad():
            omega_t, zeta_t = model.modal_properties()
            omega1_pred = float(omega_t[0].item())
            omega2_pred = float(omega_t[1].item())
            zeta1_pred  = float(zeta_t[0].item())
            zeta2_pred  = float(zeta_t[1].item())

        results = {
            "RunID": int(run_id), "stage": "stage2",

            "m1_true":      float(true_par["m1"]),
            "m1_init":      float(init_par["m1"]),
            "m1_pred":      float(model.m1.item()),
            "m1_error_pct": _safe_pct_err(float(model.m1.item()), float(true_par["m1"])),

            "m2_true":      float(true_par["m2"]),
            "m2_init":      float(init_par["m2"]),
            "m2_pred":      float(model.m2.item()),
            "m2_error_pct": _safe_pct_err(float(model.m2.item()), float(true_par["m2"])),

            "k1_true":      float(true_par["k1"]),
            "k1_init":      float(init_par["k1"]),
            "k1_pred":      float(model.k1.item()),
            "k1_error_pct": _safe_pct_err(float(model.k1.item()), float(true_par["k1"])),

            "k2_true":      float(true_par["k2"]),
            "k2_init":      float(init_par["k2"]),
            "k2_pred":      float(model.k2.item()),
            "k2_error_pct": _safe_pct_err(float(model.k2.item()), float(true_par["k2"])),

            "alpha_true":      float(true_par["alpha"]),
            "alpha_init":      float(init_par["alpha"]),
            "alpha_pred":      float(alpha_p.item()),
            "alpha_error_pct": _safe_pct_err(float(alpha_p.item()), float(true_par["alpha"])),

            "beta_true":      float(true_par["beta"]),
            "beta_init":      float(init_par["beta"]),
            "beta_pred":      float(beta_p.item()),
            "beta_error_pct": _safe_pct_err(float(beta_p.item()), float(true_par["beta"])),

            "omega1_true":      float(true_par["omega1"]),
            "omega1_pred":      omega1_pred,
            "omega1_error_pct": _safe_pct_err(omega1_pred, float(true_par["omega1"])),

            "omega2_true":      float(true_par["omega2"]),
            "omega2_pred":      omega2_pred,
            "omega2_error_pct": _safe_pct_err(omega2_pred, float(true_par["omega2"])),

            "zeta1_true":      float(true_par["zeta1"]),
            "zeta1_pred":      zeta1_pred,
            "zeta1_error_pct": _safe_pct_err(zeta1_pred, float(true_par["zeta1"])),

            "zeta2_true":      float(true_par["zeta2"]),
            "zeta2_pred":      zeta2_pred,
            "zeta2_error_pct": _safe_pct_err(zeta2_pred, float(true_par["zeta2"])),
        }

        # ── Save model + CSVs + config ────────────────────────────────────────
        torch.save(model.state_dict(), run_dir / "model.pt")
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        pd.DataFrame(param_history).to_csv(run_dir / "param_history.csv", index=False)
        pd.DataFrame([results]).to_csv(run_dir / "metrics.csv", index=False)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": int(run_id), "stage": "stage2",
                "seed": seed, "device": str(device),
                "config": cfg,
                "true_parameters": {k: float(v) for k, v in true_par.items()},
                "init_parameters": {k: float(v) for k, v in init_par.items()},
            }, f, indent=2, cls=NumpyEncoder)

        # ── Per-run plots  ← NEW ──────────────────────────────────────────────
        run_plots_dir = run_dir / "plots"
        run_plots_dir.mkdir(exist_ok=True)

        plot_training_history(
            history_df, run_id,
            warm_epochs=int(cfg["training"]["warm_epochs"]),
            save_path=run_plots_dir / "training_history.png",
        )
        print(f"  Plot → {run_plots_dir}/training_history.png")

        plot_parameter_comparison(
            results, run_id,
            save_path=run_plots_dir / "parameter_comparison.png",
        )
        print(f"  Plot → {run_plots_dir}/parameter_comparison.png")

        plot_true_vs_pred_scatter(
            results, run_id,
            save_path=run_plots_dir / "true_vs_pred_scatter.png",
        )
        print(f"  Plot → {run_plots_dir}/true_vs_pred_scatter.png")

        # ── Evaluate model on training time points for response plots ─────────
        model.eval()
        with torch.no_grad():
            t_req_eval = t_data.clone().detach().requires_grad_(True)
            u_pred_eval = model.forward_u(t_req_eval)

        # Need grad for v and a
        t_req_grad = t_data.clone().detach().requires_grad_(True)
        u_p_eval = model.forward_u(t_req_grad)
        v_p_eval = model._d_dt(u_p_eval, t_req_grad)
        a_p_eval = model._d_dt(v_p_eval, t_req_grad)

        t_np       = t_data.detach().cpu().numpy().squeeze()
        u_meas_np  = u_meas.detach().cpu().numpy()
        v_meas_np  = v_meas.detach().cpu().numpy()
        a_meas_np  = a_meas.detach().cpu().numpy()
        u_pred_np  = u_p_eval.detach().cpu().numpy()
        v_pred_np  = v_p_eval.detach().cpu().numpy()
        a_pred_np  = a_p_eval.detach().cpu().numpy()

        plot_response_comparison(
            t_np, u_meas_np, v_meas_np, a_meas_np,
            u_pred_np, v_pred_np, a_pred_np,
            run_id, save_path=run_plots_dir / "response_comparison.png",
        )
        print(f"  Plot → {run_plots_dir}/response_comparison.png")

        plot_psd_comparison(
            t_np, u_meas_np, u_pred_np, run_id,
            save_path=run_plots_dir / "psd_comparison.png",
            omega1_true=results.get("omega1_true"),
            omega2_true=results.get("omega2_true"),
            omega1_pred=results.get("omega1_pred"),
            omega2_pred=results.get("omega2_pred"),
        )
        print(f"  Plot → {run_plots_dir}/psd_comparison.png")

        plot_training_history_detailed(
            history_df, run_id,
            warm_epochs=int(cfg["training"]["warm_epochs"]),
            true_par=true_par,
            init_par=init_par,
            param_history=param_history,
            save_path=run_plots_dir / "training_history_detailed.png",
        )
        print(f"  Plot → {run_plots_dir}/training_history_detailed.png")

        # ── Summary CSV ───────────────────────────────────────────────────────
        out_csv  = paths["summary_csv"]
        new_row  = pd.DataFrame([results])
        if out_csv.exists():
            old    = pd.read_csv(out_csv)
            merged = pd.concat([old[old["RunID"] != run_id], new_row], ignore_index=True)
        else:
            merged = new_row
        merged = merged.sort_values("RunID").reset_index(drop=True)
        merged.to_csv(out_csv, index=False)

        print(f"  ✓ RunID {run_id} complete → {run_dir}")
        print(
            f"  Physical : m1={results['m1_error_pct']:.1f}% m2={results['m2_error_pct']:.1f}% "
            f"k1={results['k1_error_pct']:.1f}% k2={results['k2_error_pct']:.1f}% "
            f"α={results['alpha_error_pct']:.1f}% β={results['beta_error_pct']:.1f}%"
        )
        print(
            f"  Modal    : ω1={results['omega1_error_pct']:.1f}% ω2={results['omega2_error_pct']:.1f}% "
            f"ζ1={results['zeta1_error_pct']:.1f}% ζ2={results['zeta2_error_pct']:.1f}%"
        )
        return True

    except Exception as e:
        print(f"  ✗ Error in RunID {run_id}: {e}")
        import traceback; traceback.print_exc()
        return False



# MAIN

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Blind 2DOF PINN Training")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Re-run completed runs")
    args = parser.parse_args()

    if args.start > args.end or args.start < 1:
        print("Error: Invalid run range"); return 1

    paths = get_stage2_paths(project_root)
    for p in [paths["stage2_root"], paths["runs_dir"], paths["logs_dir"], paths["plots_dir"]]:
        p.mkdir(parents=True, exist_ok=True)

    df        = load_full_csv(CONFIG["data"]["csv_path"])
    completed = get_completed_runs(paths["runs_dir"])
    run_list  = [r for r in range(args.start, args.end + 1)
                 if args.force or r not in completed]

    print("=" * 80)
    print("STAGE 2: Fully-Blind PINN Parameter Identification (F2≠0)")
    print("=" * 80)
    print(f"Device  : {CONFIG['training']['device']}")
    print(f"Runs    : {args.start}–{args.end}  ({len(run_list)} pending)")
    print(
        f"Config  : warm={CONFIG['training']['warm_epochs']}  "
        f"ramp={CONFIG['training']['ramp_epochs']}  "
        f"epochs={CONFIG['training']['num_epochs']}"
    )
    print(f"Results : {paths['stage2_root']}")
    print("=" * 80)

    if not run_list:
        print("\n✓ All runs already completed. Use --force to re-run."); return 0

    successful: List[int] = []
    failed:     List[int] = []

    for idx, run_id in enumerate(run_list, 1):
        print(f"\n[{idx}/{len(run_list)}] Training RunID {run_id}")
        print("-" * 80)
        if train_stage2_run(run_id, CONFIG, paths, df):
            successful.append(run_id)
        else:
            failed.append(run_id)

    # ── Multi-run dashboard + report (after all runs) ← NEW ──────────────────
    if paths["summary_csv"].exists():
        summary_df = pd.read_csv(paths["summary_csv"])
        if len(summary_df) > 0:
            print("\n[Dashboard] Generating multi-run plots and Markdown report...")
            plot_multi_run_dashboard(
                summary_df, plots_dir=paths["plots_dir"],
                warm_epochs=int(CONFIG["training"]["warm_epochs"]),
            )
            save_training_report(summary_df, paths)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Successful : {len(successful)}  {successful}")
    if failed:
        print(f"Failed     : {len(failed)}  {failed}")
    print(f"Summary CSV   : {paths['summary_csv']}")
    print(f"Dashboard     : {paths['plots_dir']}/")
    print(f"Per-run plots : runs/run_XXX/plots/")
    print(f"Report        : {paths['stage2_root']}/training_report.md")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
