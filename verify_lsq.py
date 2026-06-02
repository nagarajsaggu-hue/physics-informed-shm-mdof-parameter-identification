import numpy as np
import pandas as pd

df = pd.read_csv("Data/mdof_2dof_216runs_pinn.csv")

for run_id in [1, 2, 3]:
    sub = df[df["RunID"] == run_id]
    t   = sub["time"].values
    dt  = t[1] - t[0]

    # u in mm -> convert to metres
    u1 = sub["u1"].values * 1e-3
    u2 = sub["u2"].values * 1e-3
    F1 = sub["F1"].values
    F2 = sub["F2"].values

    true_m1  = sub["m1"].iloc[0]
    true_m2  = sub["m2"].iloc[0]
    true_k1  = sub["k1"].iloc[0]
    true_k2  = sub["k2"].iloc[0]
    true_alp = sub["beta_ray"].iloc[0]
    true_bet = sub["alpha"].iloc[0]

    # FD derivatives on physical time [s]
    v1 = np.gradient(u1, dt)
    v2 = np.gradient(u2, dt)
    a1 = np.gradient(v1, dt)
    a2 = np.gradient(v2, dt)

    N = len(u1)

    # STEP 1: solve m1, m2, k1, k2 (ignore damping)
    A_mk = np.zeros((2*N, 4))
    b_mk = np.zeros(2*N)
    A_mk[:N, 0] = a1;   A_mk[:N, 2] = u1 - u2;           b_mk[:N] = F1
    A_mk[N:, 1] = a2;   A_mk[N:, 2] = -u1 + u2
    A_mk[N:, 3] = u2;   b_mk[N:]    = F2

    theta_mk, _, _, _ = np.linalg.lstsq(A_mk, b_mk, rcond=None)
    m1_p, m2_p, k1_p, k2_p = theta_mk

    print(f"\nRun {run_id} — LSQ FD (u in metres after *1e-3):")
    print(f"  m1: true={true_m1:.0f}  pred={m1_p:.1f}  err={abs(m1_p-true_m1)/true_m1*100:.2f}%")
    print(f"  m2: true={true_m2:.0f}  pred={m2_p:.1f}  err={abs(m2_p-true_m2)/true_m2*100:.2f}%")
    print(f"  k1: true={true_k1:.0f}  pred={k1_p:.1f}  err={abs(k1_p-true_k1)/true_k1*100:.2f}%")
    print(f"  k2: true={true_k2:.0f}  pred={k2_p:.1f}  err={abs(k2_p-true_k2)/true_k2*100:.2f}%")

    # STEP 2: solve alpha, beta
    r1 = F1 - m1_p*a1 - k1_p*(u1 - u2)
    r2 = F2 - m2_p*a2 - (-k1_p*u1 + (k1_p+k2_p)*u2)

    A_ab = np.zeros((2*N, 2))
    b_ab = np.zeros(2*N)
    A_ab[:N, 0] = m1_p*v1;   A_ab[:N, 1] = k1_p*(v1-v2);             b_ab[:N] = r1
    A_ab[N:, 0] = m2_p*v2;   A_ab[N:, 1] = -k1_p*v1+(k1_p+k2_p)*v2; b_ab[N:] = r2

    theta_ab, _, _, _ = np.linalg.lstsq(A_ab, b_ab, rcond=None)
    alp_p, bet_p = theta_ab

    print(f"  alpha: true={true_alp:.4f}  pred={alp_p:.4f}  err={abs(alp_p-true_alp)/true_alp*100:.2f}%")
    print(f"  beta:  true={true_bet:.3e}  pred={bet_p:.3e}  err={abs(bet_p-true_bet)/true_bet*100:.2f}%")
