import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==============================================================================
# 2. ENERGY MODEL (Common for SMEP/SDMEP)
# ==============================================================================

class EPNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.act = nn.Hardtanh()
        self.states = []

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i > 0: h = self.act(h)
            h = layer(h)
        return h

    def energy(self, x, states, y_target=None, beta=0.0):
        """
        Calculates Primitive Energy for EP.
        E = 0.5 sum(s^2) - sum(s_next * W * s_prev)
        """
        E = 0.0
        # Input layer is clamped to x
        prev_act = x

        for i, layer in enumerate(self.layers):
            s = states[i] # State of layer i+1

            # State decay term: 0.5 * ||s||^2
            E += 0.5 * torch.sum(s**2)

            # Interaction term: - s * (W @ prev_act + b)
            # This generates dynamics: ds/dt = -s + W@prev + b

            # Apply activation to previous state if hidden
            if i == 0:
                inp = prev_act
            else:
                inp = self.act(prev_act)

            drive = layer(inp)

            if i < len(self.layers)-1:
                # Hidden layers
                E -= torch.sum(self.act(s) * drive)
                prev_act = s
            else:
                # Output layer (linear state usually)
                E -= torch.sum(s * drive)

        # Nudge Cost
        if y_target is not None:
            # Output state is the last state
            out = states[-1] # Linear output
            # MSE Cost scaled by Beta
            cost = 0.5 * torch.sum((out - y_target)**2)
            E += beta * cost

        return E / x.shape[0] # Average over batch

    def settle(self, x, y_target=None, beta=0.0, steps=15,
               params_lazy=False, sparsity=0.0):
        """
        Relaxation Dynamics.
        params_lazy: If True, stop early if settled (SDMEP feature).
        sparsity: If > 0, enforce sparsity (SDMEP feature).
        """
        # Init states (cold start)
        self.states = []
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.layers):
                # Initialize near forward pass for speed
                if i > 0: h = self.act(h)
                h = layer(h)
                self.states.append(h.detach().requires_grad_(True))

        # Optimizer for State Dynamics (Fast Settle)
        # Using SGD with momentum as the physical dynamic solver
        state_opt = optim.SGD(self.states, lr=0.05, momentum=0.5)

        for t in range(steps):
            state_opt.zero_grad()
            E = self.energy(x, self.states, y_target, beta)
            E.backward()
            state_opt.step()

            # Apply SDMEP specific features
            if sparsity > 0 and t % 3 == 0:
                with torch.no_grad():
                    for s in self.states[:-1]: # Don't sparsify output
                        # Spatial/Channel Sparsity
                        # For Conv: (B, C, H, W) -> topk on C (dim 1)
                        if s.ndim > 1 and s.shape[1] > 1:
                            k = max(1, int(s.shape[1] * (1.0 - sparsity)))
                            vals, _ = torch.topk(s.abs(), k, dim=1)
                            # Get threshold (k-th value)
                            # vals[:, -1] gives (B, H, W). unsqueeze(1) -> (B, 1, H, W)
                            thresh = vals[:, -1].unsqueeze(1)

                            mask = (s.abs() >= thresh).float()
                            s.data.mul_(mask)

            # Lazy Exit (SDMEP)
            if params_lazy and t > 5:
                # Check gradient magnitude (proxy for movement)
                grad_norm = sum([s.grad.norm() for s in self.states])
                if grad_norm < 0.05:
                    break

        return [s.detach().clone() for s in self.states]

    def compute_ep_gradients(self, x, y, beta=0.5, sdmp_config=None):
        """Autograd-based EP Contrastive Phase"""
        # 1. Free Phase
        is_sdmep = sdmp_config is not None
        sparsity = sdmp_config['sparsity'] if is_sdmep else 0.0
        lazy = is_sdmep

        s_free = self.settle(x, None, 0.0, steps=20, params_lazy=lazy, sparsity=sparsity)

        # Compute Free Energy Gradients w.r.t Params
        # We need to re-compute Energy with detached states to get param grads
        # s_free contains detached tensors.
        # But we need graph connected to parameters.
        # Calling self.energy(x, s_free) uses self.layers (parameters).
        E_free = self.energy(x, s_free, y_target=None, beta=0.0)

        # 2. Nudged Phase
        s_nudged = self.settle(x, y, beta, steps=12, params_lazy=lazy, sparsity=sparsity)
        E_nudged = self.energy(x, s_nudged, y_target=y, beta=beta)

        # 3. Compute Gradients
        # Formula: grad_p = (grad_nudged - grad_free) / beta

        grads_free = torch.autograd.grad(E_free, self.parameters(), allow_unused=True)
        grads_nudged = torch.autograd.grad(E_nudged, self.parameters(), allow_unused=True)

        for p, gf, gn in zip(self.parameters(), grads_free, grads_nudged):
            if gf is None or gn is None: continue

            ep_grad = (gn - gf) / beta

            # Set gradients manually
            if p.grad is None:
                p.grad = ep_grad
            else:
                # Accumulate or overwrite?
                # Usually we zero_grad() before step.
                # If we use batches, we might accumulate?
                # But here we do one batch per step.
                p.grad = ep_grad

class EPMLP(EPNetwork):
    """Legacy wrapper for MLP construction"""
    def __init__(self, dims):
        layers = []
        for i in range(len(dims)-1):
            l = nn.Linear(dims[i], dims[i+1], bias=True)
            nn.init.orthogonal_(l.weight)
            nn.init.zeros_(l.bias)
            layers.append(l)
        super().__init__(layers)
