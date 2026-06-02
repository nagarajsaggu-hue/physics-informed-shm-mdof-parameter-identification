#!/usr/bin/env python3
"""
mdof_2dof_pinn/utils_seed.py
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[utils_seed] Global seed set to {seed}")