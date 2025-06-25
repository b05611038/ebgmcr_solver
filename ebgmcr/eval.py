from collections import deque

import torch

__all__ = ['StreamingMetrics', 'ConvergenceChecker']

class StreamingMetrics:
    def __init__(self, dim: int,
                 mode: str = "dense",       # 'dense' | 'mask' | 'threshold'
                 eps:  float = 1e-3):       # threshold for 'threshold' mode

        self.dim      = dim
        self.mode     = mode
        self.eps      = eps

        # running totals
        self.ssr      = 0.0   # Σ‖residual‖²
        self.sst_num  = 0.0   # Σ‖x‖²   (for dense / threshold modes)
        self.mu_sum   = torch.zeros(dim)  # Σ x_j
        self.n        = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor, X_hat: torch.Tensor):
        """
        Accumulate batch statistics for NMSE and R².

        Parameters
        ----------
        X, X_hat : (B, d) tensors on *any* device.
        """
        B = X.shape[0]

        if self.mode == "threshold":
            mask = (X.abs() >= self.eps) | (X_hat.abs() >= self.eps)
            X    = X * mask
            X_hat = X_hat * mask

        if self.mode == "mask":
            mask = X != 0
            # residuals on active channels only
            res_sq = ((X_hat - X)[mask] ** 2).sum().detach().cpu().item()
            sig_sq = (X[mask] ** 2).sum().detach().cpu().item()
        else:  # 'dense' or 'threshold'
            res_sq = ((X_hat - X) ** 2).sum().detach().cpu().item()
            sig_sq = (X ** 2).sum().detach().cpu().item()

        # accumulate
        self.ssr     += res_sq
        self.sst_num += sig_sq
        self.mu_sum  += X.sum(dim = 0).detach().cpu()
        self.n       += B
        return None

    def compute(self):
        """
        Returns (nmse, r2_centered) over all data seen so far.
        """
        nmse = self.ssr / self.sst_num

        # compute SST0 = Σ (x_ij - x̄_j)²
        x_bar = self.mu_sum / self.n                       # (d,)
        sst0  = self.sst_num - (x_bar.pow(2).sum().item() * self.n)

        r2_centered = 1.0 - (self.ssr / sst0)

        return nmse, r2_centered


