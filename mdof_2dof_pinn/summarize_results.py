"""
summarize_results.py

Summarize parameter identification results across runs.

Reads (preferred):
- results/mdof_pinn/eval/eval_summary.csv  (from evaluate.py)

Fallback:
- results/mdof_pinn/runs/identified_parameters.csv

Writes:
- results/mdof_pinn/eval/summary_stats.csv
- results/mdof_pinn/eval/outliers.csv
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: Dict) -> Dict[str, Path]:
    p = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    results_root = Path(p.get("results_root", Path("results") / "mdof_pinn"))
    runs_dir = Path(p.get("runs_dir", results_root / "runs"))
    summary_csv = Path(p.get("summary_csv", runs_dir / "identified_parameters.csv"))
    eval_dir = Path(p.get("eval_dir", results_root / "eval"))
    return {
        "results_root": results_root,
        "runs_dir": runs_dir,
        "summary_csv": summary_csv,
        "eval_dir": eval_dir,
    }


def rel_err(true: pd.Series, pred: pd.Series) -> pd.Series:
    den = true.abs().replace(0, np.nan)
    return (pred - true).abs() / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--topk", type=int, default=10, help="Top-K outliers per parameter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    paths["eval_dir"].mkdir(parents=True, exist_ok=True)

    eval_summary = paths["eval_dir"] / "eval_summary.csv"
    if eval_summary.exists():
        df = pd.read_csv(eval_summary)
        print(f"[summarize] Using {eval_summary}")
    else:
        df = pd.read_csv(paths["summary_csv"])
        print(f"[summarize] eval_summary.csv not found. Using {paths['summary_csv']}")

    # Parameters to summarize
    pairs = [
        ("m1_true", "m1_pred"),
        ("m2_true", "m2_pred"),
        ("k1_true", "k1_pred"),
        ("k2_true", "k2_pred"),
        ("alpha_true", "alpha_pred"),
        ("beta_true", "beta_pred"),
        ("omega1_true", "omega1_pred"),
        ("omega2_true", "omega2_pred"),
        ("zeta1_true", "zeta1_pred"),
        ("zeta2_true", "zeta2_pred"),
    ]

    stats_rows = []
    outlier_rows = []

    for tcol, pcol in pairs:
        if tcol not in df.columns or pcol not in df.columns:
            continue

        err = rel_err(df[tcol], df[pcol])
        df[f"{pcol}_relerr"] = err

        stats_rows.append({
            "metric": f"{pcol}_relerr",
            "mean_%": float(np.nanmean(err) * 100.0),
            "std_%": float(np.nanstd(err) * 100.0),
            "median_%": float(np.nanmedian(err) * 100.0),
            "max_%": float(np.nanmax(err) * 100.0),
            "n": int(np.sum(~np.isnan(err))),
        })

        # Top-K outliers
        tmp = df[["RunID", tcol, pcol, f"{pcol}_relerr"]].copy()
        tmp = tmp.sort_values(f"{pcol}_relerr", ascending=False).head(args.topk)
        tmp["parameter"] = pcol
        outlier_rows.append(tmp)

    stats_df = pd.DataFrame(stats_rows).sort_values("mean_%", ascending=False)
    stats_path = paths["eval_dir"] / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[summarize]  Wrote: {stats_path}")

    if outlier_rows:
        out_df = pd.concat(outlier_rows, ignore_index=True)
        out_path = paths["eval_dir"] / "outliers.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[summarize] Wrote: {out_path}")


if __name__ == "__main__":
    main()
