"""
run_all_runs.py — Multi-run launcher for Stage 1 Baseline 2DOF PINN
====================================================================
Delegates to trainer_stage1.py which sits directly in mdof_2dof_pinn/.

Usage:
  python -m mdof_2dof_pinn.run_all_runs --start 1 --end 5
  python -m mdof_2dof_pinn.run_all_runs --start 1 --end 5 --force
  python -m mdof_2dof_pinn.run_all_runs --start 3 --end 3
"""

import sys
from pathlib import Path

_here      = Path(__file__).resolve().parent
_proj_root = _here.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from mdof_2dof_pinn.trainer_stage1 import main

if __name__ == "__main__":
    sys.exit(main())