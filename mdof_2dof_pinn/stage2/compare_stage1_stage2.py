"""
compare_stage1_stage2.py

Compare Stage-1 vs Stage-2 parameter identification statistics.

Stage-1 CSV expected columns:
  m1_true,m1_pred,k1_true,k1_pred,... alpha_true,alpha_pred,beta_true,beta_pred

Stage-2 CSV expected columns:
  m1_true,m1_pred,k1_true,k1_pred,... alpha_true,alpha_pred,beta_true,beta_pred
(Your Stage-2 file also has *_init columns; ignored here.)

Outputs:
- results/stage2_blind/compare_stage1_stage2.csv
- results/stage2_blind/plots/compare_mape.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARAMS = ["m1", "m2", "k1", "k2", "alpha", "beta"]


def default_stage2_csv() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "results" / "stage2_blind" / "stage2_summary.csv"


def default_stage1_csv() -> Path:
    root = Path(__file__).resolve().parents[2]
    # common Stage-1 location from earlier structure
    return root / "results" / "mdof_pinn" / "runs" / "identified_parameters.csv"


def rel_err_pct(true: pd.Series, pred: pd.Series) -> pd.Series:
    den = true.abs().replace(0, np.nan)
    return ((pred - true).abs() / den) * 100.0


def stats_for(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for p in PARAMS:
        tcol = f"{p}_true"
        pcol = f"{p}_pred"
        if tcol not in df.columns or pcol not in df.columns:
            continue
        e = rel_err_pct(df[tcol], df[pcol])
        rows.append({
            "stage": tag,
            "param": p,
            "runs": int(e.notna().sum()),
            "mape_%": float(np.nanmean(e)),
            "mean_%": float(np.nanmean(e)),
            "std_%": float(np.nanstd(e)),
            "max_%": float(np.nanmax(e)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-csv", type=str, default=str(default_stage1_csv()))
    ap.add_argument("--stage2-csv", type=str, default=str(default_stage2_csv()))
    args = ap.parse_args()

    s1 = Path(args.stage1_csv)
    s2 = Path(args.stage2_csv)

    if not s2.exists():
        raise FileNotFoundError(f"Stage-2 CSV not found: {s2}")
    if not s1.exists():
        raise FileNotFoundError(f"Stage-1 CSV not found: {s1} (pass --stage1-csv)")

    df1 = pd.read_csv(s1)
    df2 = pd.read_csv(s2)

    st1 = stats_for(df1, "stage1")
    st2 = stats_for(df2, "stage2")

    comp = pd.concat([st1, st2], ignore_index=True)
    out_dir = s2.parent
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "compare_stage1_stage2.csv"
    comp.to_csv(out_csv, index=False)

    # Plot compare MAPE
    piv = comp.pivot(index="param", columns="stage", values="mape_%").fillna(np.nan)
    plt.figure()
    x = np.arange(len(piv.index))
    w = 0.35
    plt.bar(x - w/2, piv.get("stage1", pd.Series(index=piv.index)).values, width=w, label="Stage 1")
    plt.bar(x + w/2, piv.get("stage2", pd.Series(index=piv.index)).values, width=w, label="Stage 2")
    plt.xticks(x, piv.index.tolist())
    plt.ylabel("MAPE (%)")
    plt.title("Stage-1 vs Stage-2 Parameter Error")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "compare_mape.png", dpi=200)
    plt.close()

    print(f" Wrote: {out_csv}")
    print(f" Plot:  {out_dir / 'plots' / 'compare_mape.png'}")


if __name__ == "__main__":
    main()
