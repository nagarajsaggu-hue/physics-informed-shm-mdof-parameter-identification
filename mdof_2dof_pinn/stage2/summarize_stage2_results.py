"""
summarize_stage2_results.py

Reads Stage-2 summary CSV (default: results/stage2_blind/stage2_summary.csv)
Computes:
- MAPE (%) per parameter
- Mean/Std/Median/Max error (%)
- Success counts under a threshold (default 5%)
Writes:
- results/stage2_blind/summary_stats.csv
- results/stage2_blind/outliers.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PARAMS = ["m1", "m2", "k1", "k2", "alpha", "beta"]


def default_stage2_csv() -> Path:
    # project root relative to this file: mdof_2dof_pinn/stage2/ -> go up 2
    root = Path(__file__).resolve().parents[2]
    return root / "results" / "stage2_blind" / "stage2_summary.csv"


def rel_err_pct(true: pd.Series, pred: pd.Series) -> pd.Series:
    den = true.abs().replace(0, np.nan)
    return ((pred - true).abs() / den) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-csv", type=str, default=str(default_stage2_csv()))
    ap.add_argument("--success-thresh", type=float, default=5.0, help="success if abs % error < threshold")
    args = ap.parse_args()

    stage2_csv = Path(args.stage2_csv)
    if not stage2_csv.exists():
        raise FileNotFoundError(f"Stage-2 CSV not found: {stage2_csv}")

    df = pd.read_csv(stage2_csv)
    if "RunID" not in df.columns:
        raise ValueError("Expected 'RunID' column in Stage-2 summary CSV")

    out_dir = stage2_csv.parent
    stats_rows = []
    outlier_rows = []

    for p in PARAMS:
        tcol = f"{p}_true"
        pred_col = f"{p}_pred"
        if tcol not in df.columns or pred_col not in df.columns:
            continue

        e = rel_err_pct(df[tcol], df[pred_col])

        stats_rows.append({
            "param": p,
            "runs": int(e.notna().sum()),
            "mape_%": float(np.nanmean(e)),
            "mean_%": float(np.nanmean(e)),
            "std_%": float(np.nanstd(e)),
            "median_%": float(np.nanmedian(e)),
            "max_%": float(np.nanmax(e)),
            "success_count": int((e < float(args.success_thresh)).sum()),
            "success_rate_%": float((e < float(args.success_thresh)).mean() * 100.0),
        })

        tmp = df[["RunID", tcol, pred_col]].copy()
        tmp["abs_err_%"] = e
        tmp["param"] = p
        tmp = tmp.sort_values("abs_err_%", ascending=False).head(10)
        outlier_rows.append(tmp)

    stats_df = pd.DataFrame(stats_rows).sort_values("mape_%", ascending=True)
    stats_path = out_dir / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    if outlier_rows:
        out_df = pd.concat(outlier_rows, ignore_index=True)
        out_path = out_dir / "outliers.csv"
        out_df.to_csv(out_path, index=False)

    print("\n Stage-2 Summary Saved")
    print(f"  stats : {stats_path}")
    if outlier_rows:
        print(f"  outliers: {out_dir / 'outliers.csv'}")
    print("\nPreview:\n", stats_df)


if __name__ == "__main__":
    main()
