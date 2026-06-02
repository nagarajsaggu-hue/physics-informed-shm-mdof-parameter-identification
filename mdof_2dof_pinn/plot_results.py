"""
plot_stage2.py  —  Full plotting script for Stage-2 2DOF PINN

Features
--------
- Works for any selected subset of runs
- Supports:
    --start / --end
    --run-ids
- Skips missing runs safely
- Skips existing plots unless --force is used
- Generates:
    * Parameter identification plots
    * Modal plots
    * Training-history plots
    * Run-ranking plots
    * Heatmaps
    * Improvement plots
- Uses only summary/eval/history files, so it is robust and lightweight
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None



# DEFAULTS


DEFAULT_STAGE_ROOT = Path("results/mdof_pinn/stage2_fixed")

PARAMS_PHYSICAL = ["m1", "m2", "k1", "k2", "alpha", "beta"]
PARAMS_MODAL = ["omega1", "omega2", "zeta1", "zeta2"]
PARAMS_ALL = PARAMS_PHYSICAL + PARAMS_MODAL

GROUPS = {
    "mass": ["m1", "m2"],
    "stiffness": ["k1", "k2"],
    "damping": ["alpha", "beta"],
    "modal_frequency": ["omega1", "omega2"],
    "modal_damping": ["zeta1", "zeta2"],
    "overall": PARAMS_ALL,
}


# HELPERS

def load_yaml_config(path: Path) -> Dict:
    if yaml is None:
        raise ImportError("PyYAML is required for --config but is not installed.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(
    config_path: Optional[Path],
    stage_root: Optional[Path],
    summary_csv: Optional[Path],
    eval_dir: Optional[Path],
    plots_dir: Optional[Path],
) -> Dict[str, Path]:
    cfg = {}
    if config_path is not None:
        cfg = load_yaml_config(config_path)

    cfg_paths = cfg.get("paths", {})

    if stage_root is None:
        if "results_root" in cfg_paths:
            stage_root = Path(cfg_paths["results_root"])
        else:
            stage_root = DEFAULT_STAGE_ROOT

    if summary_csv is None:
        if "summary_csv" in cfg_paths:
            summary_csv = Path(cfg_paths["summary_csv"])
        else:
            summary_csv = stage_root / "identified_parameters.csv"

    if eval_dir is None:
        if "eval_dir" in cfg_paths:
            eval_dir = Path(cfg_paths["eval_dir"])
        else:
            eval_dir = stage_root / "eval"

    if plots_dir is None:
        if "plots_dir" in cfg_paths:
            plots_dir = Path(cfg_paths["plots_dir"])
        else:
            plots_dir = stage_root / "plots"

    return {
        "stage_root": stage_root,
        "summary_csv": summary_csv,
        "runs_dir": stage_root / "runs",
        "eval_dir": eval_dir,
        "plots_dir": plots_dir,
    }


def parse_run_selection(df: pd.DataFrame, start: Optional[int], end: Optional[int], run_ids: Optional[List[int]]) -> Tuple[List[int], List[int]]:
    existing = sorted(df["RunID"].astype(int).unique().tolist())

    if run_ids:
        requested = sorted(set(int(x) for x in run_ids))
    elif start is not None and end is not None:
        requested = list(range(int(start), int(end) + 1))
    else:
        requested = existing

    missing = [r for r in requested if r not in existing]
    selected = [r for r in requested if r in existing]
    return selected, missing


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def should_skip(path: Path, force: bool) -> bool:
    return path.exists() and not force


def err_col(param: str) -> str:
    return f"{param}_error_pct"


def true_col(param: str) -> str:
    return f"{param}_true"


def pred_col(param: str) -> str:
    return f"{param}_pred"


def init_col(param: str) -> str:
    return f"{param}_init"


def read_history_for_runs(runs_dir: Path, selected_runs: List[int]) -> Dict[int, pd.DataFrame]:
    histories = {}
    for run_id in selected_runs:
        hp = runs_dir / f"run_{run_id:03d}" / "history.csv"
        if hp.exists():
            try:
                histories[run_id] = pd.read_csv(hp)
            except Exception:
                pass
    return histories


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def maybe_eval_csv(eval_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    p = eval_dir / filename
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None



# PLOTTING FUNCTIONS


def plot_true_vs_pred_scatter(df: pd.DataFrame, params: List[str], out_dir: Path, force: bool) -> List[Path]:
    generated = []
    ensure_dir(out_dir)

    for param in params:
        tc = true_col(param)
        pc = pred_col(param)
        if tc not in df.columns or pc not in df.columns:
            continue

        out_path = out_dir / f"{param}_true_vs_pred.png"
        if should_skip(out_path, force):
            continue

        x = df[tc].to_numpy(dtype=float)
        y = df[pc].to_numpy(dtype=float)

        plt.figure(figsize=(6, 5))
        plt.scatter(x, y)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"{param} true")
        plt.ylabel(f"{param} predicted")
        plt.title(f"True vs Predicted: {param}")
        plt.grid(True, alpha=0.3)
        save_fig(out_path)
        generated.append(out_path)

    return generated


def plot_mean_error_bar(df: pd.DataFrame, params: List[str], out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None

    labels, means = [], []
    for p in params:
        c = err_col(p)
        if c in df.columns:
            labels.append(p)
            means.append(float(df[c].mean()))

    if not labels:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(labels, means)
    plt.ylabel("Mean error (%)")
    plt.title("Mean Percentage Error by Parameter")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_error_boxplot(df: pd.DataFrame, params: List[str], out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None

    labels, data = [], []
    for p in params:
        c = err_col(p)
        if c in df.columns:
            labels.append(p)
            data.append(df[c].to_numpy(dtype=float))

    if not labels:
        return None

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Error (%)")
    plt.title("Error Distribution by Parameter")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_modal_error_summary(df: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None

    modal_params = ["omega1", "omega2", "zeta1", "zeta2"]
    labels, means = [], []
    for p in modal_params:
        c = err_col(p)
        if c in df.columns:
            labels.append(p)
            means.append(float(df[c].mean()))

    if not labels:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.ylabel("Mean error (%)")
    plt.title("Modal Error Summary")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_runwise_overall_error(df: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None

    err_cols = [err_col(p) for p in PARAMS_ALL if err_col(p) in df.columns]
    if not err_cols:
        return None

    tmp = df.copy()
    tmp["overall_mean_error_pct"] = tmp[err_cols].mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.bar(tmp["RunID"].astype(str), tmp["overall_mean_error_pct"].to_numpy(dtype=float))
    plt.xlabel("RunID")
    plt.ylabel("Mean error (%)")
    plt.title("Overall Mean Error by Run")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_run_parameter_heatmap(df: pd.DataFrame, params: List[str], out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None

    err_cols = [err_col(p) for p in params if err_col(p) in df.columns]
    if not err_cols:
        return None

    matrix = df[err_cols].to_numpy(dtype=float)

    plt.figure(figsize=(12, max(4, 0.6 * len(df))))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Error (%)")
    plt.xticks(range(len(err_cols)), [c.replace("_error_pct", "") for c in err_cols], rotation=45, ha="right")
    plt.yticks(range(len(df)), [str(x) for x in df["RunID"].tolist()])
    plt.xlabel("Parameter")
    plt.ylabel("RunID")
    plt.title("Run × Parameter Error Heatmap")
    save_fig(out_path)
    return out_path


def plot_history_curves(histories: Dict[int, pd.DataFrame], out_dir: Path, force: bool) -> List[Path]:
    generated = []
    ensure_dir(out_dir)

    for run_id, h in histories.items():
        out_path = out_dir / f"history_run_{run_id:03d}.png"
        if should_skip(out_path, force):
            continue
        if h.empty or "epoch" not in h.columns:
            continue

        plt.figure(figsize=(9, 5))
        for col in ["loss_total", "loss_data", "loss_phys"]:
            if col in h.columns:
                plt.plot(h["epoch"], h[col], label=col)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training History — Run {run_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_fig(out_path)
        generated.append(out_path)

    return generated


def plot_physics_loss_reduction(history_summary: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None
    if history_summary is None or history_summary.empty:
        return None
    if "loss_phys_reduction_pct" not in history_summary.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(history_summary["RunID"].astype(str), history_summary["loss_phys_reduction_pct"].to_numpy(dtype=float))
    plt.xlabel("RunID")
    plt.ylabel("Physics loss reduction (%)")
    plt.title("Physics Loss Reduction by Run")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_final_loss_components(history_summary: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None
    if history_summary is None or history_summary.empty:
        return None

    required = ["RunID", "loss_total_last", "loss_data_last", "loss_phys_last"]
    if not all(c in history_summary.columns for c in required):
        return None

    x = np.arange(len(history_summary))
    width = 0.25

    plt.figure(figsize=(11, 5))
    plt.bar(x - width, history_summary["loss_total_last"].to_numpy(dtype=float), width=width, label="loss_total_last")
    plt.bar(x, history_summary["loss_data_last"].to_numpy(dtype=float), width=width, label="loss_data_last")
    plt.bar(x + width, history_summary["loss_phys_last"].to_numpy(dtype=float), width=width, label="loss_phys_last")

    plt.xticks(x, history_summary["RunID"].astype(str).tolist())
    plt.yscale("log")
    plt.xlabel("RunID")
    plt.ylabel("Final loss")
    plt.title("Final Loss Components by Run")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_improvement_summary(improvement_summary: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None
    if improvement_summary is None or improvement_summary.empty:
        return None
    if not {"parameter", "avg_relative_improvement_pct"}.issubset(improvement_summary.columns):
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(
        improvement_summary["parameter"].tolist(),
        improvement_summary["avg_relative_improvement_pct"].to_numpy(dtype=float),
    )
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Average relative improvement (%)")
    plt.title("Prediction Improvement Over Initialization")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path


def plot_group_summary(grouped_summary: pd.DataFrame, out_path: Path, force: bool) -> Optional[Path]:
    if should_skip(out_path, force):
        return None
    if grouped_summary is None or grouped_summary.empty:
        return None
    if not {"group", "mean_error_pct"}.issubset(grouped_summary.columns):
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(grouped_summary["group"].tolist(), grouped_summary["mean_error_pct"].to_numpy(dtype=float))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean group error (%)")
    plt.title("Grouped Summary Error")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)
    return out_path

# MAIN


def main():
    parser = argparse.ArgumentParser(description="Plot Stage-2 PINN results")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    parser.add_argument("--stage-root", type=str, default=None, help="Stage root directory")
    parser.add_argument("--summary-csv", type=str, default=None, help="identified_parameters.csv path")
    parser.add_argument("--eval-dir", type=str, default=None, help="Evaluation directory")
    parser.add_argument("--plots-dir", type=str, default=None, help="Plot output directory")

    parser.add_argument("--start", type=int, default=None, help="Start RunID")
    parser.add_argument("--end", type=int, default=None, help="End RunID")
    parser.add_argument("--run-ids", type=int, nargs="*", default=None, help="Specific RunIDs")

    parser.add_argument("--force", action="store_true", help="Re-generate existing plots")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    stage_root = Path(args.stage_root) if args.stage_root else None
    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    plots_dir = Path(args.plots_dir) if args.plots_dir else None

    paths = resolve_paths(config_path, stage_root, summary_csv, eval_dir, plots_dir)

    if not paths["summary_csv"].exists():
        raise FileNotFoundError(f"Summary CSV not found: {paths['summary_csv']}")

    df = pd.read_csv(paths["summary_csv"])
    if "RunID" not in df.columns:
        raise ValueError("Summary CSV must contain a 'RunID' column.")

    selected_runs, missing_runs = parse_run_selection(df, args.start, args.end, args.run_ids)
    if not selected_runs:
        print("No selected runs found.")
        if missing_runs:
            print(f"Missing requested runs: {missing_runs}")
        return

    filtered_df = df[df["RunID"].isin(selected_runs)].copy()
    filtered_df = filtered_df.sort_values("RunID").reset_index(drop=True)

    # Load optional eval outputs
    grouped_summary = maybe_eval_csv(paths["eval_dir"], "grouped_summary.csv")
    parameter_summary = maybe_eval_csv(paths["eval_dir"], "parameter_summary.csv")
    improvement_summary = maybe_eval_csv(paths["eval_dir"], "improvement_summary.csv")
    history_summary = maybe_eval_csv(paths["eval_dir"], "history_summary.csv")

    # History raw files
    histories = read_history_for_runs(paths["runs_dir"], selected_runs)

    # Plot folders
    summary_dir = paths["plots_dir"] / "summary"
    modal_dir = paths["plots_dir"] / "modal"
    history_dir = paths["plots_dir"] / "history"
    ranking_dir = paths["plots_dir"] / "ranking"
    improve_dir = paths["plots_dir"] / "improvement"

    for d in [summary_dir, modal_dir, history_dir, ranking_dir, improve_dir]:
        ensure_dir(d)

    generated = []

    # Parameter + modal scatters
    generated += plot_true_vs_pred_scatter(filtered_df, PARAMS_PHYSICAL, summary_dir, args.force)
    generated += plot_true_vs_pred_scatter(filtered_df, ["omega1", "omega2"], modal_dir, args.force)
    generated += plot_true_vs_pred_scatter(filtered_df, ["zeta1", "zeta2"], modal_dir, args.force)

    # Summary plots
    p = plot_mean_error_bar(filtered_df, PARAMS_ALL, summary_dir / "mean_error_by_parameter.png", args.force)
    if p: generated.append(p)

    p = plot_error_boxplot(filtered_df, PARAMS_ALL, summary_dir / "error_boxplot.png", args.force)
    if p: generated.append(p)

    p = plot_modal_error_summary(filtered_df, modal_dir / "modal_error_summary.png", args.force)
    if p: generated.append(p)

    p = plot_runwise_overall_error(filtered_df, ranking_dir / "runwise_overall_error.png", args.force)
    if p: generated.append(p)

    p = plot_run_parameter_heatmap(filtered_df, PARAMS_ALL, ranking_dir / "run_parameter_error_heatmap.png", args.force)
    if p: generated.append(p)

    # History plots
    generated += plot_history_curves(histories, history_dir, args.force)

    p = plot_physics_loss_reduction(history_summary, history_dir / "physics_loss_reduction.png", args.force)
    if p: generated.append(p)

    p = plot_final_loss_components(history_summary, history_dir / "final_loss_components.png", args.force)
    if p: generated.append(p)

    # Improvement / grouped summary
    p = plot_improvement_summary(improvement_summary, improve_dir / "improvement_summary.png", args.force)
    if p: generated.append(p)

    p = plot_group_summary(grouped_summary, summary_dir / "grouped_summary_error.png", args.force)
    if p: generated.append(p)

    print("=" * 80)
    print("STAGE-2 PLOTTING COMPLETED")
    print("=" * 80)
    print(f"Stage root     : {paths['stage_root']}")
    print(f"Summary CSV    : {paths['summary_csv']}")
    print(f"Selected runs  : {selected_runs}")
    if missing_runs:
        print(f"Missing runs   : {missing_runs}")
    print(f"Plots dir      : {paths['plots_dir']}")
    print(f"Generated plots: {len(generated)}")
    for gp in generated:
        print(f"  - {gp}")
    print("=" * 80)


if __name__ == "__main__":
    main()