"""
model_mdof.py — 2DOF PINN
M u'' + C u' + K u = F(t) [F1, F2 both active]

Rayleigh damping: C = alpha*M + beta*K
Trainable: m1, m2, k1, k2, alpha_log, beta_log
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class TwoDOFPINN(nn.Module):

    def __init__(
        self,
        init_params: Dict[str, float],
        hidden_layers: int = 3,
        hidden_units:  int = 64,
    ):
        super().__init__()

        # ── Neural network: t → [u1, u2] ──────────────────────────────
        layers = []
        in_dim = 1
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            in_dim = hidden_units
        self.hidden = nn.ModuleList(layers)
        self.out    = nn.Linear(hidden_units, 2)

        # ── Trainable physical parameters ─────────────────────────────
        self.m1 = nn.Parameter(torch.tensor(init_params["m1"], dtype=torch.float32))
        self.m2 = nn.Parameter(torch.tensor(init_params["m2"], dtype=torch.float32))
        self.k1 = nn.Parameter(torch.tensor(init_params["k1"], dtype=torch.float32))
        self.k2 = nn.Parameter(torch.tensor(init_params["k2"], dtype=torch.float32))

        # ── Rayleigh damping (log-parameterised for positivity) ────────
        self.alpha0    = float(init_params["alpha"])
        self.beta0     = float(init_params["beta"])
        self.alpha_log = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta_log  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    # ── Damping accessors ──────────────────────────────────────────────
    def alpha_val(self) -> torch.Tensor:
        return torch.tensor(self.alpha0, device=self.alpha_log.device,
                            dtype=torch.float32) * torch.exp(self.alpha_log)

    def beta_val(self) -> torch.Tensor:
        return torch.tensor(self.beta0, device=self.beta_log.device,
                            dtype=torch.float32) * torch.exp(self.beta_log)

    # ── Forward pass ───────────────────────────────────────────────────
    def forward_u(self, t: torch.Tensor) -> torch.Tensor:
        """t: [N,1] → u: [N,2]"""
        x = t
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return self.out(x)

    # ── Automatic differentiation helper ──────────────────────────────
    @staticmethod
    def _d_dt(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """dy/dt for y=[N,2], t=[N,1] → returns [N,2]"""
        grads = []
        for j in range(y.shape[1]):
            yj  = y[:, j:j+1]
            dyj = torch.autograd.grad(
                yj, t,
                grad_outputs=torch.ones_like(yj),
                create_graph=True,
                retain_graph=True,
            )[0]
            grads.append(dyj)
        return torch.cat(grads, dim=1)

    # ── System matrices ────────────────────────────────────────────────
    def _MCK(self, device: torch.device, dtype: torch.dtype):
        """Build mass M, damping C, stiffness K matrices."""
        z = torch.zeros((), device=device, dtype=dtype)

        M = torch.stack([
            torch.stack([self.m1.to(dtype=dtype),  z                          ]),
            torch.stack([z,                         self.m2.to(dtype=dtype)   ]),
        ])
        K = torch.stack([
            torch.stack([self.k1.to(dtype=dtype),              (-self.k1).to(dtype=dtype)             ]),
            torch.stack([(-self.k1).to(dtype=dtype),           (self.k1 + self.k2).to(dtype=dtype)   ]),
        ])

        alpha = self.alpha_val().to(dtype=dtype)
        beta  = self.beta_val().to(dtype=dtype)
        C     = alpha * M + beta * K
        return M, C, K

    # ── Physics residual ───────────────────────────────────────────────
    def compute_residuals(self, t: torch.Tensor, F_vec: torch.Tensor) -> torch.Tensor:
        """r(t) = M*a + C*v + K*u - F, shape [N,2]

        IMPORTANT: t must arrive with requires_grad=True from the caller.
        Do NOT detach t here — that breaks the gradient path to parameters.
        The caller (train_one_run.py) is responsible for:
            t_phys_req = t_phys.clone().detach().requires_grad_(True)
            residual   = model.compute_residuals(t_phys_req, F_phys)
        """
        #  use t directly — do NOT clone().detach() here
        # The old line was:  t_req = t.clone().detach().requires_grad_(True)
        # That created a disconnected leaf, breaking gradients to m1/m2/k1/k2/alpha/beta
        u = self.forward_u(t)
        v = self._d_dt(u, t)
        a = self._d_dt(v, t)
        M, C, K = self._MCK(t.device, t.dtype)
        return (a @ M.T) + (v @ C.T) + (u @ K.T) - F_vec

    # ── Modal properties ───────────────────────────────────────────────
    def modal_properties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns natural frequencies omega [rad/s] and damping ratios zeta.
        omega = sqrt(eigenvalues(M^{-1/2} K M^{-1/2}))
        zeta  = 0.5 * (alpha/omega + beta*omega)  ← correct Rayleigh formula
        """
        device = self.m1.device
        dtype  = self.m1.dtype
        z      = torch.zeros((), device=device, dtype=dtype)

        M = torch.stack([
            torch.stack([self.m1.to(dtype=dtype), z                       ]),
            torch.stack([z,                        self.m2.to(dtype=dtype) ]),
        ])
        K = torch.stack([
            torch.stack([self.k1.to(dtype=dtype),            (-self.k1).to(dtype=dtype)           ]),
            torch.stack([(-self.k1).to(dtype=dtype),         (self.k1 + self.k2).to(dtype=dtype)  ]),
        ])

        alpha = self.alpha_val().to(dtype=dtype)
        beta  = self.beta_val().to(dtype=dtype)

        m_diag      = torch.diag(M)
        inv_sqrt_m  = 1.0 / torch.sqrt(torch.clamp(m_diag, min=1e-12))
        Minv_sqrt   = torch.diag(inv_sqrt_m)
        A           = Minv_sqrt @ K @ Minv_sqrt
        lam         = torch.linalg.eigvalsh(A)
        lam         = torch.clamp(lam, min=1e-12)
        omega       = torch.sqrt(lam)
        zeta        = 0.5 * (alpha / omega + beta * omega)

        return omega, zeta
