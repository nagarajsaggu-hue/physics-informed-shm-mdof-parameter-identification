"""
trainer_stage1.py — Stage 1: Baseline 2DOF PINN  (multi-run, fully instrumented)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

current_script_path = Path(__file__).resolve().parent
# Works whether trainer_stage1.py is in mdof_2dof_pinn/ OR mdof_2dof_pinn/stage1/
# We walk up until we find the directory that contains mdof_2dof_pinn as a package
_candidate = current_script_path
for _ in range(3):
    if (_candidate / "mdof_2dof_pinn" / "__init__.py").exists():
        project_root = _candidate
        break
    if (_candidate / "__init__.py").exists():
        _candidate = _candidate.parent
    else:
        project_root = _candidate
        break
else:
    project_root = current_script_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from mdof_2dof_pinn.data_mdof  import load_full_csv, load_run_data, create_collocation_points
    from mdof_2dof_pinn.model_mdof import TwoDOFPINN
except ImportError:
    print("Critical Error: Could not import 'mdof_2dof_pinn'.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG  (used when no YAML supplied)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data": {
        "csv_path":        "Data/mdof_2dof_216runs_pinn.csv",
        "n_colloc_factor": 4.0,
    },
    "training": {
        "device":        "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs":    4000,
        "print_every":   500,
        "warm_epochs":   500,
        "seed_base":     1000,
        "learning_rate": 1e-3,
        "use_lbfgs":     True,
        "lbfgs_max_iter": 400,
        "lbfgs_lr":      0.5,
    },
    "loss": {
        "w_data":        2.0,
        "w_phys":        1e-4,   # scaled down — raw residuals are O(1e4)
        "w_param":       0.01,
        "w_modal":       0.5,
        "w_v":           0.5,
        "w_a":           1.5,
        "use_va_loss":   True,
        "perturb_level": 0.10,
    },
    "paths": {
        "results_root": "results/mdof_pinn/stage1",
    },
}

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

# ──────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ──────────────────────────────────────────────────────────────────────────────
_DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#c9d1d9", "grid.color":      "#21262d",
    "grid.linestyle":   "--",      "grid.alpha":       0.6,
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "font.size": 10,
}
_C = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657",
      "#79c0ff","#56d364","#ff7b72","#bc8cff","#e3b341"]

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
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # deep-merge defaults so missing keys are filled in
    merged = {**DEFAULT_CONFIG}
    for section, vals in raw.items():
        merged[section] = {**merged.get(section, {}), **vals}
    return merged

def _seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _has_log_damping(model: nn.Module) -> bool:
    return (hasattr(model, "alpha_log") and hasattr(model, "beta_log")
            and callable(getattr(model, "alpha_val", None)))

def _alpha_beta(model: nn.Module) -> Tuple[float, float]:
    if _has_log_damping(model):
        return float(model.alpha_val().item()), float(model.beta_val().item())
    return float(model.alpha.item()), float(model.beta.item())

def _damp_params(model: nn.Module):
    return ([model.alpha_log, model.beta_log] if _has_log_damping(model)
            else [model.alpha, model.beta])

def _get_paths(cfg: dict) -> dict:
    root = Path(cfg.get("paths", {}).get("results_root", "results/mdof_pinn/stage1"))
    return {
        "stage1_root": root,
        "runs_dir":    root / "runs",
        "plots_dir":   root / "plots",
        "logs_dir":    root / "logs",
        "summary_csv": root / "identified_parameters.csv",
    }

def _ensure_paths(paths: dict):

    for key, p in paths.items():
        if not isinstance(p, Path):
            continue
        if key == "summary_csv":

            if p.exists() and p.is_dir():
                import shutil
                shutil.rmtree(p)
                print(f"  WARNING: Removed accidental directory at {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

def perturb_params(true_par: Dict[str, float], level: float) -> Dict[str, float]:
    init = {k: float(v) for k, v in true_par.items()}
    if level > 0.0:
        for key in ["m1", "m2", "k1", "k2", "alpha", "beta"]:
            if key in init:
                init[key] *= 1.0 + (np.random.rand() * 2 - 1) * level
    return init

def pct_err(pred: float, true: float) -> float:
    return abs(pred - true) / (abs(true) + 1e-12) * 100.0

def get_completed_runs(runs_dir: Path) -> Set[int]:
    completed = set()
    if not runs_dir.exists(): return completed
    for rf in runs_dir.glob("run_*"):
        if rf.is_dir() and (rf/"model.pt").exists() and (rf/"metrics.csv").exists():
            try:
                sz = (rf/"model.pt").stat().st_size
                if sz > 1000:   # must be non-empty
                    completed.add(int(rf.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return completed

def atomic_save(obj, path: Path):
    """Write to a temp file first, then rename — prevents 0-byte model.pt on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try: os.remove(tmp_path)
        except OSError: pass
        raise

# ──────────────────────────────────────────────────────────────────────────────
# PER-RUN PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_history_detailed(
    history_df: pd.DataFrame, param_history: List[Dict],
    run_id: int, true_par: dict, save_path: Path,
):
    """4-panel: total loss | data vs physics loss | smoothed loss | param % error convergence."""
    _style()
    ep = history_df["epoch"].values
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(f"Detailed Training History — Stage 1 Run {run_id}",
                 fontsize=14, fontweight="bold")

    axes[0].semilogy(ep, history_df["loss_total"], color=_C[0], lw=1.4, label="Total")
    axes[0].set_ylabel("Loss (log)"); axes[0].set_title("Total Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].semilogy(ep, history_df["loss_data"],  color=_C[1], lw=1.3, label="Data")
    axes[1].semilogy(ep, history_df["loss_phys"],  color=_C[2], lw=1.3, label="Physics")
    axes[1].semilogy(ep, history_df["loss_param"], color=_C[4], lw=1.0, label="Param reg")
    if "loss_modal" in history_df.columns:
        axes[1].semilogy(ep, history_df["loss_modal"], color=_C[3], lw=1.0, label="Modal")
    axes[1].set_ylabel("Loss (log)"); axes[1].set_title("Loss Components")
    axes[1].legend(fontsize=7, ncol=4); axes[1].grid(True)

    smooth = pd.Series(history_df["loss_total"].values).rolling(50, min_periods=1).mean().values
    axes[2].semilogy(ep, smooth, color=_C[8] if len(_C)>8 else _C[3], lw=1.2,
                     label="Smoothed total (rolling-50)")
    axes[2].set_ylabel("Loss (log)"); axes[2].set_title("Convergence Proxy (Smoothed Total Loss)")
    axes[2].legend(fontsize=8); axes[2].grid(True)

    ax4 = axes[3]
    if param_history:
        ph_df = pd.DataFrame(param_history)
        for pkey, col in [("m1",_C[0]),("m2",_C[1]),("k1",_C[2]),("k2",_C[4])]:
            if pkey in ph_df.columns and pkey in true_par:
                err = np.abs(ph_df[pkey].values - true_par[pkey]) / (abs(true_par[pkey])+1e-12)*100
                ax4.plot(ph_df["epoch"].values, err, color=col, lw=1.2, label=f"{pkey} err%")
        ax4.axhline(5,  color="white", ls=":", lw=0.8, alpha=0.5)
        ax4.axhline(10, color=_C[2],   ls=":", lw=0.8, alpha=0.5)
        ax4.set_ylabel("% Error"); ax4.set_title("Physical Parameter Convergence")
        ax4.legend(fontsize=8, ncol=4); ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, "param_history not recorded",
                 ha="center", va="center", transform=ax4.transAxes)

    ax4.set_xlabel("Epoch", fontsize=9)
    fig.subplots_adjust(hspace=0.38, top=0.94)
    _safe_save(fig, save_path)


def plot_parameter_comparison(results: dict, run_id: int, save_path: Path):
    """Bar chart: true / init / predicted for physical + modal parameters."""
    _style()
    params = list(PARAM_META.keys())
    n = len(params)
    x = np.arange(n)
    w = 0.26

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Parameter Comparison — Stage 1 Run {run_id}",
                 fontsize=13, fontweight="bold")

    # normalise each param to its true value for display
    true_v  = np.array([results.get(f"{p}_true", 1.0) for p in params])
    norm    = np.abs(true_v) + 1e-12
    true_n  = true_v / norm
    init_v  = np.array([results.get(f"{p}_init", results.get(f"{p}_true", 1.0)) for p in params])
    pred_v  = np.array([results.get(f"{p}_pred", results.get(f"{p}_true", 1.0)) for p in params])
    init_n  = init_v / norm
    pred_n  = pred_v / norm

    ax.bar(x - w, true_n, w, label="True",      color=_C[1], alpha=0.85)
    ax.bar(x,     init_n, w, label="Init",      color=_C[4], alpha=0.75)
    ax.bar(x + w, pred_n, w, label="Predicted", color=_C[0], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_META[p][0] for p in params], fontsize=9)
    ax.set_ylabel("Normalised value (pred / true)")
    ax.axhline(1.0, color="white", ls="--", lw=0.8, alpha=0.6)
    ax.legend(fontsize=9); ax.grid(True, axis="y")

    for j, p in enumerate(params):
        err = results.get(f"{p}_error_pct", 0.0)
        ax.text(j + w, pred_n[j] + 0.02, f"{err:.1f}%", ha="center", fontsize=6.5,
                color=_C[2] if err > 10 else _C[1] if err < 5 else _C[4])

    fig.subplots_adjust(bottom=0.12, top=0.90)
    _safe_save(fig, save_path)


def plot_true_vs_pred_scatter(results: dict, run_id: int, save_path: Path):
    """Scatter: true vs predicted for all 10 parameters (log scale)."""
    _style()
    params = list(PARAM_META.keys())
    tv = [abs(results.get(f"{p}_true", 1.0)) + 1e-12 for p in params]
    pv = [abs(results.get(f"{p}_pred", results.get(f"{p}_true", 1.0))) + 1e-12 for p in params]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f"True vs Predicted — Stage 1 Run {run_id}",
                 fontsize=13, fontweight="bold")
    lo, hi = min(min(tv), min(pv)) * 0.8, max(max(tv), max(pv)) * 1.2
    ax.plot([lo, hi], [lo, hi], color="white", ls="--", lw=0.8, alpha=0.5, label="Perfect fit")
    for j, p in enumerate(params):
        ax.scatter(tv[j], pv[j], color=_C[j % len(_C)], s=80, zorder=3,
                   label=f"{PARAM_META[p][0]}  {results.get(f'{p}_error_pct',0):.1f}%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("True value"); ax.set_ylabel("Predicted value")
    ax.legend(fontsize=7, ncol=2); ax.grid(True)
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
    fig.suptitle(f"Response: Measured vs PINN — Stage 1 Run {run_id}",
                 fontsize=14, fontweight="bold")

    for row, (rlbl, meas, pred) in enumerate(zip(row_labels, meas_data, pred_data)):
        for col in range(2):
            ax = axes[row, col]
            m  = meas[:, col] if meas.ndim == 2 else meas
            p  = pred[:, col] if pred.ndim == 2 else pred
            ax.plot(t_np, m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.85)
            ax.plot(t_np, p, color=_C[0], lw=1.0, label="PINN pred")
            ax.set_ylabel(rlbl, fontsize=8)
            if row == 0:   ax.set_title(f"DOF {col+1}", fontsize=10, fontweight="bold")
            if row == 2:   ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True)
            if row == 0 and col == 1: ax.legend(fontsize=7)

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    _safe_save(fig, save_path)


def plot_psd_comparison(
    t_np, u_meas_np, u_pred_np, run_id: int, save_path: Path,
    omega1_true=None, omega2_true=None,
    omega1_pred=None, omega2_pred=None,
):
    """Welch PSD of measured vs predicted displacement + natural frequency lines."""
    from scipy.signal import welch as sp_welch
    _style()
    dt = float(np.mean(np.diff(t_np))); fs = 1.0 / dt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Power Spectral Density (Displacement) — Stage 1 Run {run_id}",
                 fontsize=13, fontweight="bold")

    for col, (dof_lbl, ax) in enumerate(zip(["DOF 1", "DOF 2"], axes)):
        m = u_meas_np[:, col] if u_meas_np.ndim == 2 else u_meas_np
        p = u_pred_np[:, col] if u_pred_np.ndim == 2 else u_pred_np
        f_m, psd_m = sp_welch(m, fs=fs, nperseg=min(256, len(m)//4))
        f_p, psd_p = sp_welch(p, fs=fs, nperseg=min(256, len(p)//4))
        ax.semilogy(f_m, psd_m, color=_C[2], lw=1.2, ls="--", label="Measured", alpha=0.85)
        ax.semilogy(f_p, psd_p, color=_C[0], lw=1.0, label="PINN pred")
        if col == 0 and omega1_true:
            ax.axvline(omega1_true/(2*np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₁ true ({omega1_true/(2*np.pi):.2f} Hz)")
        if col == 0 and omega1_pred:
            ax.axvline(omega1_pred/(2*np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₁ pred ({omega1_pred/(2*np.pi):.2f} Hz)")
        if col == 1 and omega2_true:
            ax.axvline(omega2_true/(2*np.pi), color=_C[1], ls=":", lw=1.2,
                       label=f"ω₂ true ({omega2_true/(2*np.pi):.2f} Hz)")
        if col == 1 and omega2_pred:
            ax.axvline(omega2_pred/(2*np.pi), color=_C[4], ls=":", lw=1.2,
                       label=f"ω₂ pred ({omega2_pred/(2*np.pi):.2f} Hz)")
        ax.set_xlabel("Frequency (Hz)", fontsize=9); ax.set_ylabel("PSD (m²/Hz)", fontsize=9)
        ax.set_title(dof_lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True)

    fig.subplots_adjust(wspace=0.3, top=0.88)
    _safe_save(fig, save_path)


def plot_eval_heatmap(results: dict, run_id: int, save_path: Path):
    """Single-row colour heatmap of % error for all 10 parameters."""
    _style()
    params   = list(PARAM_META.keys())
    pnames   = [PARAM_META[p][0] for p in params]
    pct_errs = np.array([results.get(f"{p}_error_pct", 0.0) for p in params]).reshape(1,-1)

    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.suptitle(f"Parameter % Error Heatmap — Stage 1 Run {run_id}",
                 fontsize=12, fontweight="bold")
    im = ax.imshow(pct_errs, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=15)
    ax.set_xticks(range(len(pnames))); ax.set_xticklabels(pnames, fontsize=9)
    ax.set_yticks([])
    for j, v in enumerate(pct_errs[0]):
        ax.text(j, 0, f"{v:.2f}%", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if v > 10 else "black")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, fraction=0.05)
    cbar.set_label("% Error  (green ≤ 5% | yellow 5–10% | red > 10%)", fontsize=8)
    fig.subplots_adjust(top=0.78)
    _safe_save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-RUN DASHBOARD PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_dashboard_error_bars(summary_df: pd.DataFrame, plots_dir: Path):
    """Grouped bar: % error per parameter per run."""
    _style()
    params  = list(PARAM_META.keys())
    run_ids = sorted(summary_df["RunID"].tolist())
    n_runs  = len(run_ids)
    x = np.arange(len(params)); w = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Parameter % Error — All Stage 1 Runs", fontsize=13, fontweight="bold")

    for i, rid in enumerate(run_ids):
        row  = summary_df[summary_df["RunID"]==rid].iloc[0]
        vals = [row.get(f"{p}_error_pct", 0.0) for p in params]
        ax.bar(x + (i - n_runs/2 + 0.5)*w, vals, w,
               label=f"Run {rid}", color=_C[i % len(_C)], alpha=0.85)

    ax.axhline(5,  color="white", ls=":",  lw=0.8, alpha=0.5, label="5% threshold")
    ax.axhline(10, color=_C[2],   ls="--", lw=0.8, alpha=0.5, label="10% threshold")
    ax.set_xticks(x); ax.set_xticklabels([PARAM_META[p][0] for p in params], fontsize=9)
    ax.set_ylabel("% Error"); ax.legend(fontsize=7, ncol=n_runs+2); ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.12, top=0.90)
    _safe_save(fig, plots_dir / "dashboard_error_bars.png")


def plot_dashboard_loss_curves(runs_dir: Path, run_ids: List[int], plots_dir: Path):
    """Line plot: total loss history for all runs overlaid."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Total Loss History — All Stage 1 Runs", fontsize=13, fontweight="bold")

    for i, rid in enumerate(run_ids):
        h_path = runs_dir / f"run_{rid:03d}" / "history.csv"
        if not h_path.exists(): continue
        hdf = pd.read_csv(h_path)
        ax.semilogy(hdf["epoch"], hdf["loss_total"],
                    color=_C[i % len(_C)], lw=1.2, label=f"Run {rid}")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss (log)")
    ax.legend(fontsize=8); ax.grid(True)
    fig.subplots_adjust(top=0.88)
    _safe_save(fig, plots_dir / "dashboard_loss_curves.png")


def save_markdown_report(summary_df: pd.DataFrame, stage1_root: Path):
    params   = list(PARAM_META.keys())
    run_ids  = sorted(summary_df["RunID"].tolist())
    pnames   = [PARAM_META[p][0] for p in params]

    lines = [
        "# Stage 1 PINN — Training & Parameter Identification Report\n",
        f"**Runs completed:** {len(run_ids)}\n",
        "---\n",
        "## Parameter % Errors\n",
        "| Run | " + " | ".join(pnames) + " | Mean err% |",
        "|-----|" + "|".join(["------"]*len(params)) + "|-----------|",
    ]
    for rid in run_ids:
        row  = summary_df[summary_df["RunID"]==rid].iloc[0]
        vals = [f"{row.get(f'{p}_error_pct',0):.2f}%" for p in params]
        mean = np.mean([row.get(f"{p}_error_pct", 0.0) for p in params])
        lines.append("| " + " | ".join([str(rid)] + vals + [f"{mean:.2f}%"]) + " |")

    lines += [
        "\n## Modal Properties\n",
        "| Run | ω₁ err% | ω₂ err% | ζ₁ err% | ζ₂ err% |",
        "|-----|---------|---------|---------|---------|",
    ]
    for rid in run_ids:
        row = summary_df[summary_df["RunID"]==rid].iloc[0]
        lines.append(
            f"| {rid} | {row.get('omega1_error_pct',0):.2f}% "
            f"| {row.get('omega2_error_pct',0):.2f}% "
            f"| {row.get('zeta1_error_pct',0):.2f}% "
            f"| {row.get('zeta2_error_pct',0):.2f}% |"
        )

    rpt = stage1_root / "training_report.md"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report → {rpt}")


# ──────────────────────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def train_stage1_run(run_id: int, cfg: Dict, df: pd.DataFrame) -> bool:
    paths   = _get_paths(cfg)
    _ensure_paths(paths)   # re-check every run in case of partial cleanup
    run_dir   = paths["runs_dir"] / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    seed = int(cfg["training"].get("seed_base", 1000)) + run_id
    _seed_all(seed)

    requested = cfg["training"]["device"]
    device    = torch.device(
        requested if (requested == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"\n[Stage 1] RunID {run_id} | device={device} | seed={seed}")

    try:
        # ── Load data ──────────────────────────────────────────────────────
        use_va = bool(cfg["loss"].get("use_va_loss", True))
        if use_va:
            t_data, F_data, u_meas, v_meas, a_meas, true_par = load_run_data(
                df, run_id, device, return_va=True)
        else:
            t_data, F_data, u_meas, true_par = load_run_data(df, run_id, device)
            v_meas = a_meas = None

        print(f"  [data] N={t_data.shape[0]} | m1={true_par['m1']:.0f} "
              f"m2={true_par['m2']:.0f} k1={true_par['k1']:.0f} k2={true_par['k2']:.0f} "
              f"| alpha={true_par['alpha']:.4f} beta={true_par['beta']:.2e}")

        n_colloc = int(cfg["data"]["n_colloc_factor"] * t_data.shape[0])
        t_phys, F_phys = create_collocation_points(t_data, F_data, n_colloc)
        t_phys = t_phys.to(device); F_phys = F_phys.to(device)

        F_rms = torch.sqrt(torch.mean(F_phys**2)).detach() + 1e-8

        # Better normalisation: scale residuals by estimated force magnitude
        # M*a ~ m_ref * max(|u|)/T^2  gives a characteristic residual scale
        m_ref   = float(true_par["m1"] + true_par["m2"]) / 2.0
        T_total = float(t_data[-1].item() - t_data[0].item()) + 1e-8
        u_scale = float(u_meas.abs().max().item()) + 1e-8
        a_scale = u_scale / (T_total ** 2 + 1e-8)
        residual_ref = torch.tensor(m_ref * a_scale + 1e-8,
                                    dtype=torch.float32, device=device).detach()

        # ── Init params ────────────────────────────────────────────────────
        perturb_level = float(cfg["loss"].get("perturb_level", 0.10))
        init_par = perturb_params(true_par, perturb_level)
        print(f"  Init: m1={init_par['m1']:.0f} m2={init_par['m2']:.0f} "
              f"k1={init_par['k1']:.0f} k2={init_par['k2']:.0f}")

        model = TwoDOFPINN(init_params=init_par).to(device)

        # ── Optimiser ──────────────────────────────────────────────────────
        lr       = float(cfg["training"]["learning_rate"])
        nn_params = list(model.hidden.parameters()) + list(model.out.parameters())
        mk_params = [model.m1, model.m2, model.k1, model.k2]
        ab_params = _damp_params(model)

        optimizer = torch.optim.Adam([
            {"params": nn_params, "lr": lr},
            {"params": mk_params, "lr": lr * 0.50},
            {"params": ab_params, "lr": lr * 0.01},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=300)

        # ── Loss weights ───────────────────────────────────────────────────
        w_data  = float(cfg["loss"]["w_data"])
        w_phys  = float(cfg["loss"]["w_phys"])
        w_param = float(cfg["loss"]["w_param"])
        w_modal = float(cfg["loss"].get("w_modal", 0.5))
        w_v     = float(cfg["loss"].get("w_v", 0.5))
        w_a     = float(cfg["loss"].get("w_a", 1.5))

        num_epochs  = int(cfg["training"]["num_epochs"])
        print_every = int(cfg["training"]["print_every"])

        history:       List[Dict] = []
        param_history: List[Dict] = []

        # ── Training loop ──────────────────────────────────────────────────
        warm_epochs = int(cfg["training"].get("warm_epochs", 500))
        print(f"  Warm-up: physics loss frozen for first {warm_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Data loss
            t_req  = t_data.clone().detach().requires_grad_(True)
            u_pred = model.forward_u(t_req)
            loss_data = w_data * torch.mean((u_pred - u_meas) ** 2)

            loss_v_term = torch.zeros((), device=device)
            loss_a_term = torch.zeros((), device=device)
            if use_va:
                v_pred = model._d_dt(u_pred, t_req)
                if w_v > 0 and v_meas is not None:
                    loss_v_term = w_v * torch.mean((v_pred - v_meas) ** 2)
                if w_a > 0 and a_meas is not None:
                    a_pred_tr = model._d_dt(v_pred, t_req)
                    loss_a_term = w_a * torch.mean((a_pred_tr - a_meas) ** 2)
            loss_data = loss_data + loss_v_term + loss_a_term

            # Physics loss — ramped in after warm-up (avoids dominating early training)
            t_phys_req  = t_phys.clone().detach().requires_grad_(True)
            residual    = model.compute_residuals(t_phys_req, F_phys)
            phys_ramp   = 0.0 if epoch <= warm_epochs else min(1.0, (epoch - warm_epochs) / 500.0)
            loss_phys   = w_phys * phys_ramp * torch.mean((residual / residual_ref) ** 2)

            # Parameter regularisation
            loss_param = torch.zeros((), device=device)
            if w_param > 0:
                for name in ["m1","m2","k1","k2"]:
                    iv = init_par[name]
                    p  = getattr(model, name)
                    loss_param = loss_param + ((p - iv) / (abs(iv) + 1e-12)) ** 2
                if _has_log_damping(model):
                    loss_param = loss_param + model.alpha_log**2 + model.beta_log**2
                else:
                    for name in ["alpha","beta"]:
                        iv = init_par[name]
                        p  = getattr(model, name)
                        loss_param = loss_param + ((p - iv) / (abs(iv) + 1e-12)) ** 2
                loss_param = w_param * (loss_param / 6.0)

            # Modal supervision (Stage 1 feature)
            loss_modal = torch.zeros((), device=device)
            if w_modal > 0.0:
                omega_pred, zeta_pred = model.modal_properties()
                omega_true = torch.tensor(
                    [true_par["omega1"], true_par["omega2"]],
                    device=device, dtype=omega_pred.dtype)
                zeta_true = torch.tensor(
                    [true_par["zeta1"], true_par["zeta2"]],
                    device=device, dtype=zeta_pred.dtype)
                loss_omega = torch.mean(
                    ((omega_pred - omega_true) / (torch.abs(omega_true) + 1e-12)) ** 2)
                loss_zeta  = torch.mean(
                    ((zeta_pred - zeta_true) / (torch.abs(zeta_true) + 1e-12)) ** 2)
                loss_modal = w_modal * (loss_omega + loss_zeta)

            loss = loss_data + loss_phys + loss_param + loss_modal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step(loss.item())

            with torch.no_grad():
                model.m1.clamp_(min=1e-6); model.m2.clamp_(min=1e-6)
                model.k1.clamp_(min=1e-6); model.k2.clamp_(min=1e-6)

            history.append({
                "epoch":       epoch,
                "loss_total":  float(loss.item()),
                "loss_data":   float(loss_data.item()),
                "loss_phys":   float(loss_phys.item()),
                "loss_param":  float(loss_param.item()),
                "loss_modal":  float(loss_modal.item()),
            })
            param_history.append({
                "epoch": epoch,
                "m1":    float(model.m1.item()),
                "m2":    float(model.m2.item()),
                "k1":    float(model.k1.item()),
                "k2":    float(model.k2.item()),
            })

            if epoch % print_every == 0 or epoch == num_epochs:
                print(f"  Ep {epoch:5d} | total={loss.item():.3e} "
                      f"data={loss_data.item():.3e} phys={loss_phys.item():.3e} "
                      f"modal={loss_modal.item():.3e} "
                      f"k1={pct_err(model.k1.item(),true_par['k1']):.1f}% "
                      f"k2={pct_err(model.k2.item(),true_par['k2']):.1f}% "
                      f"m1={pct_err(model.m1.item(),true_par['m1']):.1f}% "
                      f"m2={pct_err(model.m2.item(),true_par['m2']):.1f}%")

        # ── Optional LBFGS refinement ──────────────────────────────────────
        if cfg["training"].get("use_lbfgs", True):
            print("  LBFGS refinement...")
            for p in model.parameters(): p.requires_grad_(True)
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=float(cfg["training"].get("lbfgs_lr", 0.5)),
                max_iter=int(cfg["training"].get("lbfgs_max_iter", 400)),
                history_size=50, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                t_r  = t_data.clone().detach().requires_grad_(True)
                u_p  = model.forward_u(t_r)
                v_p  = model._d_dt(u_p, t_r)
                a_p  = model._d_dt(v_p, t_r)
                ld   = (w_data * torch.mean((u_p - u_meas)**2)
                        + w_v * torch.mean((v_p - v_meas)**2) if v_meas is not None else
                        w_data * torch.mean((u_p - u_meas)**2))
                t_pr = t_phys.clone().detach().requires_grad_(True)
                lp   = w_phys * torch.mean((model.compute_residuals(t_pr, F_phys)/residual_ref)**2)
                total = ld + lp
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                return total
            lbfgs.step(closure)
            with torch.no_grad():
                model.m1.clamp_(min=1e-6); model.m2.clamp_(min=1e-6)
                model.k1.clamp_(min=1e-6); model.k2.clamp_(min=1e-6)

        # ── Final metrics ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            alpha_p, beta_p = _alpha_beta(model)
            omega_t, zeta_t = model.modal_properties()

        results = {
            "RunID": int(run_id), "stage": "stage1",
            "m1_true":  float(true_par["m1"]),  "m1_init":  float(init_par["m1"]),
            "m1_pred":  float(model.m1.item()),  "m1_error_pct":  pct_err(model.m1.item(), true_par["m1"]),
            "m2_true":  float(true_par["m2"]),  "m2_init":  float(init_par["m2"]),
            "m2_pred":  float(model.m2.item()),  "m2_error_pct":  pct_err(model.m2.item(), true_par["m2"]),
            "k1_true":  float(true_par["k1"]),  "k1_init":  float(init_par["k1"]),
            "k1_pred":  float(model.k1.item()),  "k1_error_pct":  pct_err(model.k1.item(), true_par["k1"]),
            "k2_true":  float(true_par["k2"]),  "k2_init":  float(init_par["k2"]),
            "k2_pred":  float(model.k2.item()),  "k2_error_pct":  pct_err(model.k2.item(), true_par["k2"]),
            "alpha_true": float(true_par["alpha"]), "alpha_init": float(init_par["alpha"]),
            "alpha_pred": alpha_p, "alpha_error_pct": pct_err(alpha_p, true_par["alpha"]),
            "beta_true":  float(true_par["beta"]),  "beta_init":  float(init_par["beta"]),
            "beta_pred":  beta_p,  "beta_error_pct":  pct_err(beta_p,  true_par["beta"]),
            "omega1_true": float(true_par["omega1"]), "omega1_pred": float(omega_t[0].item()),
            "omega1_error_pct": pct_err(float(omega_t[0].item()), true_par["omega1"]),
            "omega2_true": float(true_par["omega2"]), "omega2_pred": float(omega_t[1].item()),
            "omega2_error_pct": pct_err(float(omega_t[1].item()), true_par["omega2"]),
            "zeta1_true": float(true_par["zeta1"]), "zeta1_pred": float(zeta_t[0].item()),
            "zeta1_error_pct": pct_err(float(zeta_t[0].item()), true_par["zeta1"]),
            "zeta2_true": float(true_par["zeta2"]), "zeta2_pred": float(zeta_t[1].item()),
            "zeta2_error_pct": pct_err(float(zeta_t[1].item()), true_par["zeta2"]),
        }

        # ── Save artifacts (atomic model.pt save) ─────────────────────────
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        pd.DataFrame(param_history).to_csv(run_dir / "param_history.csv", index=False)
        pd.DataFrame([results]).to_csv(run_dir / "metrics.csv", index=False)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id, "seed": seed, "device": str(device), "config": cfg,
                "true_parameters": {k: float(v) for k, v in true_par.items()},
                "init_parameters": {k: float(v) for k, v in init_par.items()},
            }, f, indent=2, cls=NumpyEncoder)

        # Atomic save — prevents 0-byte model.pt on WSL crash
        atomic_save(model.state_dict(), run_dir / "model.pt")
        print(f"  ✓ model.pt saved ({(run_dir/'model.pt').stat().st_size/1024:.1f} KB)")

        # Update global summary CSV
        out_csv = paths["summary_csv"]
        new_row = pd.DataFrame([results])
        if out_csv.exists():
            old    = pd.read_csv(out_csv)
            merged = pd.concat([old[old["RunID"] != run_id], new_row], ignore_index=True)
        else:
            merged = new_row
        merged.sort_values("RunID").reset_index(drop=True).to_csv(out_csv, index=False)

        # ── Per-run plots ──────────────────────────────────────────────────
        plot_training_history_detailed(
            history_df, param_history, run_id, true_par,
            plots_dir / "training_history_detailed.png")
        print(f"  Plot → {plots_dir}/training_history_detailed.png")

        plot_parameter_comparison(results, run_id, plots_dir / "parameter_comparison.png")
        print(f"  Plot → {plots_dir}/parameter_comparison.png")

        plot_true_vs_pred_scatter(results, run_id, plots_dir / "true_vs_pred_scatter.png")
        print(f"  Plot → {plots_dir}/true_vs_pred_scatter.png")

        plot_eval_heatmap(results, run_id, plots_dir / "eval_summary_heatmap.png")
        print(f"  Plot → {plots_dir}/eval_summary_heatmap.png")

        # Response & PSD require forward pass
        t_req_eval = t_data.clone().detach().requires_grad_(True)
        u_pe = model.forward_u(t_req_eval)
        v_pe = model._d_dt(u_pe, t_req_eval)
        a_pe = model._d_dt(v_pe, t_req_eval)
        t_np      = t_data.detach().cpu().numpy().squeeze()
        u_meas_np = u_meas.detach().cpu().numpy()
        v_meas_np = (v_meas.detach().cpu().numpy() if v_meas is not None
                     else np.zeros_like(u_meas_np))
        a_meas_np = (a_meas.detach().cpu().numpy() if a_meas is not None
                     else np.zeros_like(u_meas_np))
        u_pred_np = u_pe.detach().cpu().numpy()
        v_pred_np = v_pe.detach().cpu().numpy()
        a_pred_np = a_pe.detach().cpu().numpy()

        plot_response_comparison(
            t_np, u_meas_np, v_meas_np, a_meas_np,
            u_pred_np, v_pred_np, a_pred_np,
            run_id, plots_dir / "response_comparison.png")
        print(f"  Plot → {plots_dir}/response_comparison.png")

        plot_psd_comparison(
            t_np, u_meas_np, u_pred_np, run_id,
            plots_dir / "psd_comparison.png",
            omega1_true=true_par.get("omega1"), omega2_true=true_par.get("omega2"),
            omega1_pred=results["omega1_pred"],  omega2_pred=results["omega2_pred"],
        )
        print(f"  Plot → {plots_dir}/psd_comparison.png")

        print(f"\n  ✓ RunID {run_id} complete → {run_dir}")
        print(f"  Physical : m1={results['m1_error_pct']:.1f}% "
              f"m2={results['m2_error_pct']:.1f}% "
              f"k1={results['k1_error_pct']:.1f}% k2={results['k2_error_pct']:.1f}% "
              f"α={results['alpha_error_pct']:.1f}% β={results['beta_error_pct']:.1f}%")
        print(f"  Modal    : ω1={results['omega1_error_pct']:.1f}% "
              f"ω2={results['omega2_error_pct']:.1f}% "
              f"ζ1={results['zeta1_error_pct']:.1f}% ζ2={results['zeta2_error_pct']:.1f}%")
        return True

    except Exception as e:
        print(f"\n  [ERROR] RunID {run_id}: {e}")
        import traceback; traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 1: Baseline 2DOF PINN (multi-run)")
    parser.add_argument("--start",  type=int, default=1)
    parser.add_argument("--end",    type=int, default=5)
    parser.add_argument("--force",  action="store_true", help="Re-run completed runs")
    parser.add_argument("--config", type=str,
                        default="mdof_2dof_pinn/config_mdof.yaml",
                        help="Path to YAML config (defaults built-in if missing)")
    args = parser.parse_args()

    if args.start > args.end or args.start < 1:
        print("Error: Invalid run range."); return 1

    # Load config (fall back to DEFAULT_CONFIG if file missing)
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(str(cfg_path))
        print(f"Config loaded from {cfg_path}")
    else:
        cfg = DEFAULT_CONFIG
        print(f"Config file {cfg_path} not found — using built-in defaults.")

    paths = _get_paths(cfg)
    _ensure_paths(paths)

    df = load_full_csv(cfg["data"]["csv_path"])

    completed = get_completed_runs(paths["runs_dir"])
    run_list  = [r for r in range(args.start, args.end + 1)
                 if args.force or r not in completed]

    print("=" * 80)
    print("STAGE 1 — Baseline PINN Parameter Identification")
    print("=" * 80)
    print(f"Device  : {cfg['training']['device']}")
    print(f"Runs    : {args.start}–{args.end}  ({len(run_list)} pending)")
    print(f"Epochs  : {cfg['training']['num_epochs']}  "
          f"(warm {cfg['training'].get('warm_epochs','N/A')}  LBFGS "
          f"{'ON' if cfg['training'].get('use_lbfgs') else 'OFF'})")
    print(f"Results : {paths['stage1_root']}")
    print("=" * 80)

    if not run_list:
        print("All runs already completed. Use --force to re-run."); return 0

    successful, failed = [], []
    for idx, run_id in enumerate(run_list, 1):
        print(f"\n[{idx}/{len(run_list)}] Training RunID {run_id}")
        print("-" * 80)
        if train_stage1_run(run_id, cfg, df):
            successful.append(run_id)
        else:
            failed.append(run_id)

    # Dashboard + report (only if ≥ 2 runs done)
    out_csv = paths["summary_csv"]
    if out_csv.exists() and len(successful) >= 1:
        summary_df = pd.read_csv(out_csv)
        if len(summary_df) >= 2:
            print("\n[Dashboard] Generating multi-run plots and Markdown report...")
            plot_dashboard_error_bars(summary_df, paths["plots_dir"])
            plot_dashboard_loss_curves(
                paths["runs_dir"], summary_df["RunID"].tolist(), paths["plots_dir"])
            save_markdown_report(summary_df, paths["stage1_root"])

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Successful : {len(successful)}  {successful}")
    if failed: print(f"Failed     : {len(failed)}  {failed}")
    print(f"Summary CSV: {out_csv}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
