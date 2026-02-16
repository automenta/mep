import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

class MuonOptimizer(Optimizer):
    """
    SMEP Optimizer: Pure Muon (Newton-Schulz) on all 2D matrices.
    Assumes external Spectral Normalization or soft constraints.
    """
    def __init__(self, params, lr=0.02, momentum=0.9, wd=0.0005, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, wd=wd, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def newton_schulz(self, G, steps):
        """Standard Muon Orthogonalization"""
        if G.ndim != 2: return G
        r, c = G.shape

        # Handle rectangular matrices by transposing if needed
        transposed = False
        if r < c:
            transposed = True
            G = G.T
            r, c = c, r

        # Pre-normalize to ensure convergence (Frobenius)
        X = G.clone()
        norm = X.norm() + 1e-6
        X.div_(norm)

        # NS Iteration: X = 0.5 * X * (3I - X^T X)
        I = torch.eye(c, device=G.device)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * I - A)

        res = X * norm # Restore scale

        if transposed:
            return res.T
        return res

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                # Get Grad
                g = p.grad.data

                # Orthogonalize (Muon)
                if p.ndim >= 2:
                    # Reshape for Conv: (C_out, C_in, K, K) -> (C_out, C_in*K*K)
                    orig_shape = g.shape
                    if p.ndim > 2:
                        g_flat = g.view(g.shape[0], -1)
                    else:
                        g_flat = g

                    update_flat = self.newton_schulz(g_flat, group['ns_steps'])

                    if p.ndim > 2:
                        update = update_flat.view(orig_shape)
                    else:
                        update = update_flat
                else:
                    update = g

                # Momentum
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(update)

                # Apply (with weight decay)
                p.data.mul_(1 - group['lr'] * group['wd'])
                p.data.add_(buf, alpha=-group['lr'])

class SDMEPOptimizer(Optimizer):
    """
    SDMEP Optimizer:
    1. Hybrid Dion (Low-Rank) / Muon (Newton-Schulz).
    2. Error Feedback (Continual Learning).
    3. Strict Spectral Constraint (L < 1).
    """
    def __init__(self, params, lr=0.02, momentum=0.9, wd=0.0005,
                 gamma=0.95, rank_frac=0.2, error_beta=0.9, dion_thresh=100000):
        defaults = dict(lr=lr, momentum=momentum, wd=wd, gamma=gamma,
                        rank_frac=rank_frac, error_beta=error_beta, dion_thresh=dion_thresh)
        super().__init__(params, defaults)

    @torch.no_grad()
    def get_spectral_norm(self, W, u, v, iter=3):
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)

        if u is None: u = torch.randn(W.shape[0], device=W.device)
        if v is None: v = torch.randn(W.shape[1], device=W.device)
        for _ in range(iter):
            v = F.normalize(torch.mv(W.T, u), dim=0, eps=1e-8)
            u = F.normalize(torch.mv(W, v), dim=0, eps=1e-8)
        sigma = torch.dot(u, torch.mv(W, v))
        return sigma, u, v

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]

                # Init State
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['error_buffer'] = torch.zeros_like(p)
                    if p.ndim >= 2:
                        state['u_spec'] = None
                        state['v_spec'] = None

                # 1. Retrieve Gradient + Error Feedback
                g = p.grad.data
                # Add residual from previous low-rank approx
                g_aug = g + group['error_beta'] * state['error_buffer']

                update = g_aug.clone()

                # 2. Hybrid Dion / Muon
                if p.ndim >= 2:
                    # Flatten for processing
                    orig_shape = p.shape
                    if p.ndim > 2:
                        g_flat = g_aug.view(p.shape[0], -1)
                        p_flat_shape = (p.shape[0], -1)
                    else:
                        g_flat = g_aug
                        p_flat_shape = p.shape

                    if p.numel() > group['dion_thresh']:
                        # --- DION (Low Rank) ---
                        # Project update onto top-k components
                        k = max(1, int(min(p_flat_shape) * group['rank_frac']))
                        # SVD is expensive in python, but simulates O(Nr) logic
                        # In C++/CUDA this would be iterative QR
                        U, S, Vh = torch.linalg.svd(g_flat, full_matrices=False)
                        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
                        update_lowrank = U @ (torch.diag(S) @ Vh)

                        # Store Error (Residual)
                        if p.ndim > 2:
                            state['error_buffer'] = g_aug - update_lowrank.view(orig_shape)
                            update = update_lowrank.view(orig_shape)
                        else:
                            state['error_buffer'] = g_aug - update_lowrank
                            update = update_lowrank
                    else:
                        # --- MUON (Full Rank Ortho) ---
                        # Simplified NS for POC
                        norm = g_flat.norm() + 1e-6
                        X = g_flat / norm
                        I = torch.eye(X.shape[1], device=X.device)
                        for _ in range(3):
                            X = 0.5 * X @ (3*I - X.T @ X)
                        update_flat = X * norm
                        state['error_buffer'].zero_() # No error

                        if p.ndim > 2:
                            update = update_flat.view(orig_shape)
                        else:
                            update = update_flat

                # 3. Momentum & Apply
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(update)

                p.data.mul_(1 - group['lr'] * group['wd'])
                p.data.add_(buf, alpha=-group['lr'])

                # 4. Enforce Spectral Constraint
                if p.ndim >= 2:
                    sigma, u, v = self.get_spectral_norm(p.data, state['u_spec'], state['v_spec'])
                    state['u_spec'], state['v_spec'] = u, v

                    if sigma > group['gamma']:
                        p.data.mul_(group['gamma'] / sigma)
