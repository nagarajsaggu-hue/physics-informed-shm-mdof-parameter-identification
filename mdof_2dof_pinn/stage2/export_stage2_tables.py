"""
export_stage2_tables.py

Exports thesis tables for Stage-2 results:
- summary_table.csv
- summary_table.tex (LaTeX)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PARAMS = ["m1", "m2", "k1", "k2", "alpha", "beta"]


def default_stage2_csv() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "results" / "stage2_blind" / "stage2_summary.csv"


def rel_err_pct(true: pd.Series, pred: pd.Series) -> pd.Series:
    den = true.abs().replace(0, np.nan)
    return ((pred - true).abs() / den) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-csv", type=str, default=str(default_stage2_csv()))
    ap.add_argument("--caption", type=str, default="Stage-2 identification error statistics.")
    ap.add_argument("--label", type=str, default="tab:stage2_errors")
    args = ap.parse_args()

    stage2_csv = Path(args.stage2_csv)
    if not stage2_csv.exists():
        raise FileNotFoundError(f"Stage-2 CSV not found: {stage2_csv}")

    df = pd.read_csv(stage2_csv)
    out_dir = stage2_csv.parent

    rows = []
    for p in PARAMS:
        tcol = f"{p}_true"
        pcol = f"{p}_pred"
        if tcol not in df.columns or pcol not in df.columns:
            continue
        e = rel_err_pct(df[tcol], df[pcol])
        rows.append({
            "Parameter": p,
            "Runs": int(e.notna().sum()),
            "MeanError(%)": float(np.nanmean(e)),
            "StdError(%)": float(np.nanstd(e)),
            "MedianError(%)": float(np.nanmedian(e)),
            "MaxError(%)": float(np.nanmax(e)),
        })

    table = pd.DataFrame(rows)
    csv_out = out_dir / "summary_table.csv"
    table.to_csv(csv_out, index=False)

    # LaTeX export (simple, clean)
    tex_out = out_dir / "summary_table.tex"
    latex = table.to_latex(index=False, float_format="%.3f")
    latex = (
        "\\begin{table}[t]\n\\centering\n"
        + latex
        + f"\\caption{{{args.caption}}}\n"
        + f"\\label{{{args.label}}}\n"
        + "\\end{table}\n"
    )
    tex_out.write_text(latex, encoding="utf-8")

    print(f" Wrote: {csv_out}")
    print(f" Wrote: {tex_out}")


if __name__ == "__main__":
    main()
