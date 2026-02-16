import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import datasets, transforms
import time
import math

# ==============================================================================
# 1. OPTIMIZERS
# ==============================================================================

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
        state_opt = torch.optim.SGD(self.states, lr=0.05, momentum=0.5)
        
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

# ==============================================================================
# 3. RUNNER
# ==============================================================================

def train(mode="Backprop", model_type="MLP"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training Mode: {mode} ({model_type}) ---")
    
    # Setup Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Subset for speed in POC
    train_set = torch.utils.data.Subset(train_set, range(4000))
    loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    
    # Setup Model
    if model_type == "MLP":
        # 784 -> 1000 -> 10
        model = EPMLP([784, 1000, 10]).to(device)
    else:
        # Simple CNN: Conv(1->16, 3x3) -> Conv(16->32, 3x3) -> Flatten -> Linear(32*24*24->10)
        # Note: No pooling to keep it simple for EP state shapes for now
        class Flatten(nn.Module):
            def forward(self, x):
                return x.view(x.size(0), -1)

        layers = [
            nn.Conv2d(1, 16, 3, padding=0, bias=False), # 28->26
            nn.Conv2d(16, 32, 3, padding=0, bias=False), # 26->24
            Flatten(),
            nn.Linear(32*24*24, 10, bias=False)
        ]
        # Init weights orthogonally for stability
        for l in layers:
            if isinstance(l, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(l.weight)
                if l.bias is not None: nn.init.zeros_(l.bias)

        model = EPNetwork(layers).to(device)

    # Setup Optimizer
    if mode == "Backprop":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01 if model_type=="CNN" else 0.05, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    elif mode == "SMEP":
        # Pure Muon
        optimizer = MuonOptimizer(model.parameters(), lr=0.02, ns_steps=4)
    elif mode == "SDMEP":
        # Hybrid Dion-Muon + Spectral + Error Feedback
        # dion_thresh=500000 ensures large linear layers use Dion
        sdmep_lr = 0.005 if model_type == "CNN" else 0.02
        optimizer = SDMEPOptimizer(model.parameters(), lr=sdmep_lr, gamma=0.95,
                                   rank_frac=0.1, error_beta=0.9, dion_thresh=500000)
    
    # Train Loop
    model.train()
    start_time = time.time()
    
    for epoch in range(3):
        correct = 0
        total = 0
        ep_loss = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Flatten for MLP
            if model_type == "MLP":
                x = x.view(x.shape[0], -1)

            y_oh = F.one_hot(y, 10).float()
            
            optimizer.zero_grad()
            
            if mode == "Backprop":
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                # Acc calc
                correct += (pred.argmax(1) == y).sum().item()
                ep_loss += loss.item()
                
            else:
                # EP Modes
                conf = None
                if mode == "SDMEP":
                    conf = {'sparsity': 0.2} # 20% sparsity
                
                # Tune Beta for CNN to avoid explosion
                beta_val = 0.1 if model_type == "CNN" else 0.5
                model.compute_ep_gradients(x, y_oh, beta=beta_val, sdmp_config=conf)
                optimizer.step()
                
                # Inference for stats
                with torch.no_grad():
                    pred = model(x)
                    correct += (pred.argmax(1) == y).sum().item()
                    ep_loss += F.mse_loss(pred, y_oh).item()

        # Stats
        acc = 100 * correct / len(train_set)
        
        # Check Spectral Norm of Layer 0
        l0_w = model.layers[0].weight
        if l0_w.ndim > 2:
             w_mat = l0_w.view(l0_w.shape[0], -1)
        else:
             w_mat = l0_w

        u = torch.randn(w_mat.shape[0], device=device)
        v = torch.randn(w_mat.shape[1], device=device)
        for _ in range(5):
            v = F.normalize(torch.mv(w_mat.t(), u), dim=0)
            u = F.normalize(torch.mv(w_mat, v), dim=0)
        sigma = torch.dot(u, torch.mv(w_mat, v)).item()
        
        print(f"Epoch {epoch+1} | Acc: {acc:.2f}% | Loss: {ep_loss/len(loader):.4f} | L0 Sigma: {sigma:.4f}")

    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")
    return acc, total_time

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("COMPARISON BENCHMARK: MNIST SUBSET (4000 samples)")
    
    print("\n[MLP BASELINE]")
    bp_acc, bp_time = train("Backprop", "MLP")
    sdmep_acc, sdmep_time = train("SDMEP", "MLP")
    
    print("\n[CNN TEST]")
    # 1. Backprop CNN
    cnn_bp_acc, cnn_bp_time = train("Backprop", "CNN")
    
    # 2. SDMEP CNN
    cnn_sdmep_acc, cnn_sdmep_time = train("SDMEP", "CNN")
    
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Method':<15} | {'Acc':<8} | {'Time':<8}")
    print("-" * 50)
    print(f"{'BP (MLP)':<15} | {bp_acc:.2f}%   | {bp_time:.2f}s")
    print(f"{'SDMEP (MLP)':<15} | {sdmep_acc:.2f}%   | {sdmep_time:.2f}s")
    print(f"{'BP (CNN)':<15} | {cnn_bp_acc:.2f}%   | {cnn_bp_time:.2f}s")
    print(f"{'SDMEP (CNN)':<15} | {cnn_sdmep_acc:.2f}%   | {cnn_sdmep_time:.2f}s")
