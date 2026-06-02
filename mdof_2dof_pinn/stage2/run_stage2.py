import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import sys
import os


current_script_path = Path(__file__).resolve().parent
project_root = current_script_path
if (current_script_path / "mdof_2dof_pinn").exists():
    project_root = current_script_path
elif (current_script_path.parent / "mdof_2dof_pinn").exists():
    project_root = current_script_path.parent
else:
    project_root = Path(os.getcwd())

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import shared core modules
try:
    from mdof_2dof_pinn.data_mdof import load_full_csv, load_run_data, create_collocation_points
    from mdof_2dof_pinn.model_mdof import TwoDOFPINN
except ImportError:
    print(" Error: Could not import 'mdof_2dof_pinn'. Check your folder structure.")
    sys.exit(1)

# --- 2. CONFIGURATION (Tweaked for Stability) ---
CONFIG = {
    "data": {
        "csv_path": "Data/mdof_2dof_216runs_pinn.csv",
        "n_colloc_factor": 2.0
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": 5000,
        "print_every": 500,
        "runs_start": 1,
        "runs_end": 5
    },
    "loss": {
        "w_data": 1.0,
        "w_phys": 2.0,  # Physics Weight
        "perturb_level": 0.15  # +/- 15% Blind Noise
    }
}


class ScaledTwoDOFPINN(TwoDOFPINN):
    """
    k_effective = k_initial_guess * scale_factor
    scale_factor starts at 1.0.
    """

    def __init__(self, init_params: Dict[str, float]):
        super().__init__(init_params)

        # 1. Store the initial guess as a fixed reference
        self.register_buffer('m1_ref', torch.tensor(init_params['m1'], dtype=torch.float32))
        self.register_buffer('m2_ref', torch.tensor(init_params['m2'], dtype=torch.float32))
        self.register_buffer('k1_ref', torch.tensor(init_params['k1'], dtype=torch.float32))
        self.register_buffer('k2_ref', torch.tensor(init_params['k2'], dtype=torch.float32))

        # 2. Remove the original parameters
        for name in ['m1', 'm2', 'k1', 'k2']:
            if name in self._parameters: del self._parameters[name]
            if hasattr(self, name): delattr(self, name)

        # 3. Define the new Learnable Scaling Factors (init at 1.0)
        self.m1_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.m2_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.k1_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.k2_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    # 4. Override properties
    @property
    def m1(self):
        return self.m1_ref * self.m1_scale

    @property
    def m2(self):
        return self.m2_ref * self.m2_scale

    @property
    def k1(self):
        return self.k1_ref * self.k1_scale

    @property
    def k2(self):
        return self.k2_ref * self.k2_scale


def perturb_params(true_p: Dict[str, float], level: float) -> Dict[str, float]:
    distorted = {}
    targets = ["m1", "m2", "k1", "k2", "alpha", "beta"]
    for k, v in true_p.items():
        if k in targets:
            noise = (np.random.rand() * 2 - 1) * level
            distorted[k] = v * (1.0 + noise)
        else:
            distorted[k] = v
    return distorted


def save_local_results(run_id, model, true_par, init_par):
    out_dir = project_root / "results" / "stage2_blind"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle alpha/beta (direct or log)
    if hasattr(model, "alpha_val"):
        alpha_pred = float(model.alpha_val().item())
        beta_pred = float(model.beta_val().item())
    else:
        alpha_pred = float(model.alpha.item())
        beta_pred = float(model.beta.item())

    row = {
        "RunID": run_id,
        "k1_true": true_par["k1"], "k1_init": init_par["k1"], "k1_pred": float(model.k1.item()),
        "k2_true": true_par["k2"], "k2_init": init_par["k2"], "k2_pred": float(model.k2.item()),
        "m1_true": true_par["m1"], "m1_init": init_par["m1"], "m1_pred": float(model.m1.item()),
        "m2_true": true_par["m2"], "m2_init": init_par["m2"], "m2_pred": float(model.m2.item()),
        "alpha_true": true_par["alpha"], "alpha_pred": alpha_pred,
        "beta_true": true_par["beta"], "beta_pred": beta_pred,
    }

    csv_path = out_dir / "stage2_summary.csv"
    df_new = pd.DataFrame([row])

    if csv_path.exists():
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    print(f"    Saved Run {run_id} to {csv_path.name}")


def train_stage2_run(run_id: int, cfg: Dict):
    device = torch.device(cfg["training"]["device"])

    try:
        df = load_full_csv(cfg["data"]["csv_path"])
        t_data, F_data, u_meas, true_par = load_run_data(df, run_id, device)
    except Exception as e:
        print(f" Error loading Run {run_id}: {e}")
        return

    n_colloc = int(cfg["data"]["n_colloc_factor"] * t_data.shape[0])
    t_phys, F_phys = create_collocation_points(t_data, F_data, n_colloc)

    perturb_level = float(cfg["loss"]["perturb_level"])
    init_par = perturb_params(true_par, perturb_level)

    print(f"\n[Stage 2] Run {run_id} Init (+/- {perturb_level:.0%}):")
    print(f"  k1: True={true_par['k1']:.0f} -> Init={init_par['k1']:.0f}")

    model = ScaledTwoDOFPINN(init_params=init_par).to(device)

    # --- TUNED OPTIMIZER ---
    if hasattr(model, 'alpha_log'):
        damp_params = [model.alpha_log, model.beta_log]
    else:
        damp_params = [model.alpha, model.beta]

    nn_params = list(model.hidden.parameters()) + list(model.out.parameters())

    optimizer = torch.optim.Adam([
        {'params': nn_params, 'lr': 1e-3},
        {'params': [model.k1_scale, model.k2_scale], 'lr': 5e-3},
        {'params': [model.m1_scale, model.m2_scale], 'lr': 5e-4},
        {'params': damp_params, 'lr': 1e-3}
    ])


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500,
                                                           verbose=True)

    epochs = int(cfg["training"]["num_epochs"])
    w_data = float(cfg["loss"]["w_data"])
    w_phys = float(cfg["loss"]["w_phys"])

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        u_pred = model.forward_u(t_data)
        loss_data = torch.mean((u_pred - u_meas) ** 2)

        res = model.compute_residuals(t_phys, F_phys)
        loss_phys = torch.mean(res ** 2)

        loss = w_data * loss_data + w_phys * loss_phys
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step(loss)

        # --- TIGHTER CONSTRAINTS ---
        with torch.no_grad():
            model.m1_scale.clamp_(min=0.5, max=2.0)
            model.m2_scale.clamp_(min=0.5, max=2.0)
            model.k1_scale.clamp_(min=0.5, max=2.0)
            model.k2_scale.clamp_(min=0.5, max=2.0)

        if epoch % cfg["training"]["print_every"] == 0:
            k1_err = abs(model.k1.item() - true_par['k1']) / true_par['k1'] * 100
            print(f"Ep {epoch} | Loss: {loss.item():.2e} | k1 Err: {k1_err:.1f}% | Scale: {model.k1_scale.item():.3f}")

    save_local_results(run_id, model, true_par, init_par)


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Starting Stage 2 Blind ID (Device: {CONFIG['training']['device']})")

    # Reset result file for a clean start
    result_file = project_root / "results/stage2_blind/stage2_summary.csv"
    if result_file.exists(): result_file.unlink()

    for r_id in range(CONFIG["training"]["runs_start"], CONFIG["training"]["runs_end"] + 1):
        train_stage2_run(r_id, CONFIG)