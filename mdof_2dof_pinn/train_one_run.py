"""
train_one_run.py — Backward-compatibility shim
================================================
Imports from trainer_stage1.py (same folder) so any existing code that does:
  from mdof_2dof_pinn.train_one_run import load_config, train_one_run
keeps working without changes.
"""
from mdof_2dof_pinn.trainer_stage1 import (
    load_config,
    train_stage1_run as train_one_run,
    DEFAULT_CONFIG,
    pct_err,
    perturb_params,
)

__all__ = ["load_config", "train_one_run", "DEFAULT_CONFIG", "pct_err", "perturb_params"]