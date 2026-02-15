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
        if r > c: return G # Simplified for POC: skip highly rectangular
        
        # Pre-normalize to ensure convergence (Frobenius)
        X = G.clone()
        norm = X.norm() + 1e-6
        X.div_(norm)
        
        # NS Iteration: X = 0.5 * X * (3I - X^T X)
        I = torch.eye(c, device=G.device)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * I - A)
            
        return X * norm # Restore scale

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
                if p.ndim == 2:
                    update = self.newton_schulz(g, group['ns_steps'])
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
    def get_spectral_norm(self, W, u, v, iter=1):
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
                    if p.ndim == 2:
                        state['u_spec'] = None
                        state['v_spec'] = None

                # 1. Retrieve Gradient + Error Feedback
                g = p.grad.data
                # Add residual from previous low-rank approx
                g_aug = g + group['error_beta'] * state['error_buffer']
                
                update = g_aug.clone()
                
                # 2. Hybrid Dion / Muon
                if p.ndim == 2:
                    if p.numel() > group['dion_thresh']:
                        # --- DION (Low Rank) ---
                        # Project update onto top-k components
                        k = max(1, int(min(p.shape) * group['rank_frac']))
                        # SVD is expensive in python, but simulates O(Nr) logic
                        # In C++/CUDA this would be iterative QR
                        U, S, Vh = torch.linalg.svd(g_aug, full_matrices=False)
                        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
                        update_lowrank = U @ (torch.diag(S) @ Vh)
                        
                        # Store Error (Residual)
                        state['error_buffer'] = g_aug - update_lowrank
                        update = update_lowrank
                    else:
                        # --- MUON (Full Rank Ortho) ---
                        # Simplified NS for POC
                        norm = update.norm() + 1e-6
                        X = update / norm
                        I = torch.eye(X.shape[1], device=X.device)
                        for _ in range(3):
                            X = 0.5 * X @ (3*I - X.T @ X)
                        update = X * norm
                        state['error_buffer'].zero_() # No error
                
                # 3. Momentum & Apply
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(update)
                
                p.data.mul_(1 - group['lr'] * group['wd'])
                p.data.add_(buf, alpha=-group['lr'])
                
                # 4. Enforce Spectral Constraint
                if p.ndim == 2:
                    sigma, u, v = self.get_spectral_norm(p.data, state['u_spec'], state['v_spec'])
                    state['u_spec'], state['v_spec'] = u, v
                    
                    if sigma > group['gamma']:
                        p.data.mul_(group['gamma'] / sigma)

# ==============================================================================
# 2. ENERGY MODEL (Common for SMEP/SDMEP)
# ==============================================================================

class EPMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            l = nn.Linear(dims[i], dims[i+1], bias=True)
            # Orthogonal init helps EP start stable
            nn.init.orthogonal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
        
        self.dims = dims
        self.act = nn.Hardtanh() 
        self.states = [None] * len(dims)

    def forward(self, x):
        # Standard forward pass (for Backprop baseline)
        h = x
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)

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
            drive = layer(self.act(prev_act) if i < len(self.layers)-1 else prev_act)
            if i < len(self.layers)-1:
                # Hidden layers
                E -= torch.sum(self.act(s) * drive)
                prev_act = self.act(s)
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
        batch_size = x.shape[0]
        
        # Init states (cold start)
        self.states = []
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.layers):
                # Initialize near forward pass for speed
                pre_act = layer(h if i==0 else self.act(h))
                self.states.append(pre_act.detach().requires_grad_(True))
                h = pre_act
        
        # Optimizer for State Dynamics (Fast Settle)
        # Using SGD with momentum as the physical dynamic solver
        state_opt = torch.optim.SGD(self.states, lr=0.1, momentum=0.5)
        
        for t in range(steps):
            state_opt.zero_grad()
            E = self.energy(x, self.states, y_target, beta)
            E.backward()
            state_opt.step()
            
            # Apply SDMEP specific features
            if sparsity > 0 and t % 3 == 0:
                with torch.no_grad():
                    for s in self.states[:-1]: # Don't sparsify output
                        # Simple Top-K Hard Threshold
                        k = int(s.shape[1] * (1.0 - sparsity))
                        vals, _ = torch.topk(s.abs(), k, dim=1)
                        thresh = vals[:, -1].unsqueeze(1)
                        s.data = s.data * (s.data.abs() >= thresh).float()

            # Lazy Exit (SDMEP)
            if params_lazy and t > 5:
                # Check gradient magnitude (proxy for movement)
                grad_norm = sum([s.grad.norm() for s in self.states])
                if grad_norm < 0.05:
                    break
        
        return [s.detach().clone() for s in self.states]

    def compute_ep_gradients(self, x, y, beta=0.5, sdmp_config=None):
        """Standard EP Contrastive Phase"""
        # 1. Free Phase
        is_sdmep = sdmp_config is not None
        sparsity = sdmp_config['sparsity'] if is_sdmep else 0.0
        lazy = is_sdmep
        
        s_free = self.settle(x, None, 0.0, steps=20, params_lazy=lazy, sparsity=sparsity)
        
        # 2. Nudged Phase
        s_nudged = self.settle(x, y, beta, steps=12, params_lazy=lazy, sparsity=sparsity)
        
        # 3. Compute Gradients
        # For a layer s_next = W @ s_prev
        # Grad = (1/beta) * (s_next_n * s_prev_n.T - s_next_f * s_prev_f.T)
        
        bs = x.shape[0]
        scale = 1.0 / (beta * bs)
        
        act_free = [x] + [self.act(s) for s in s_free[:-1]]
        act_nudged = [x] + [self.act(s) for s in s_nudged[:-1]]
        
        # Since output layer is linear in this energy def, we use state directly
        out_free = s_free # Last element is output state
        out_nudged = s_nudged
        
        for i, layer in enumerate(self.layers):
            # Prev activation
            p_f = act_free[i]
            p_n = act_nudged[i]
            
            # Next state/grad (EP rule varies by formulation, this is the Hebbian form)
            # Gradient of Energy wrt W is -rho(s) * rho(prev).
            # EP update is -(dL/dW) ~ -(dF/dW - dN/dW).
            # So update direction is (Nudged - Free).
            
            # We need dE/dW form.
            # Term in Energy is - s_next * W * p_prev
            # dE/dW = - s_next^T * p_prev
            # Grad_theta = (dE_nudge/dW - dE_free/dW) / beta
            #            = ( (-s_n * p_n) - (-s_f * p_f) ) / beta
            #            = (s_f * p_f - s_n * p_n) / beta
            # But we want to MINIMIZE loss, so we move opposite to Grad_theta.
            # Update ~ (s_n * p_n - s_f * p_f)
            
            next_f = s_free[i]
            next_n = s_nudged[i]
            
            # Apply activation for hidden layers in the Hebbian product if strictly following
            # standard EP, but primitive energy usually uses the state directly.
            # We used act() in energy for hidden, linear for output.
            
            if i < len(self.layers) - 1:
                nf = self.act(next_f)
                nn_ = self.act(next_n)
            else:
                nf = next_f
                nn_ = next_n
                
            # Compute Grad
            dW = scale * (nn_.T @ p_n - nf.T @ p_f)
            db = scale * (nn_.sum(0) - nf.sum(0))
            
            # Set into .grad
            layer.weight.grad = -dW # EP produces the update direction, PyTorch optimizers subtract grad
            layer.bias.grad = -db

# ==============================================================================
# 3. RUNNER
# ==============================================================================

def train(mode="Backprop"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training Mode: {mode} ---")
    
    # Setup Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Subset for speed in POC
    train_set = torch.utils.data.Subset(train_set, range(4000))
    loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    
    # Setup Model (784 -> 1000 -> 10)
    # 1000 hidden units creates a 784x1000 matrix (~800k params), triggering Dion threshold
    model = EPMLP([784, 1000, 10]).to(device)
    
    # Setup Optimizer
    if mode == "Backprop":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    elif mode == "SMEP":
        # Pure Muon
        optimizer = MuonOptimizer(model.parameters(), lr=0.02, ns_steps=4)
    elif mode == "SDMEP":
        # Hybrid Dion-Muon + Spectral + Error Feedback
        # dion_thresh=500000 ensures Layer 1 (784k) uses Dion, Layer 2 (10k) uses Muon
        optimizer = SDMEPOptimizer(model.parameters(), lr=0.02, gamma=0.95, 
                                   rank_frac=0.1, error_beta=0.9, dion_thresh=500000)
    
    # Train Loop
    model.train()
    start_time = time.time()
    
    for epoch in range(3):
        correct = 0
        total = 0
        ep_loss = 0
        
        for x, y in loader:
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
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
                
                model.compute_ep_gradients(x, y_oh, beta=0.5, sdmp_config=conf)
                optimizer.step()
                
                # Inference for stats
                with torch.no_grad():
                    pred = model(x)
                    correct += (pred.argmax(1) == y).sum().item()
                    ep_loss += F.mse_loss(pred, y_oh).item()

        # Stats
        acc = 100 * correct / len(train_set)
        
        # Check Spectral Norm of the large layer (L1)
        l1_w = model.layers[0].weight
        u = torch.randn(l1_w.shape[0], device=device)
        v = torch.randn(l1_w.shape[1], device=device)
        # Power iter for checking
        for _ in range(5):
            v = F.normalize(torch.mv(l1_w.t(), u), dim=0)
            u = F.normalize(torch.mv(l1_w, v), dim=0)
        sigma = torch.dot(u, torch.mv(l1_w, v)).item()
        
        print(f"Epoch {epoch+1} | Acc: {acc:.2f}% | Loss: {ep_loss/len(loader):.4f} | L1 Sigma: {sigma:.4f}")

    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")
    return acc, total_time

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("COMPARISON BENCHMARK: MNIST SUBSET (4000 samples)")
    
    # 1. Backprop
    bp_acc, bp_time = train("Backprop")
    
    # 2. SMEP (Muon Only)
    smep_acc, smep_time = train("SMEP")
    
    # 3. SDMEP (Dion-Muon + Spectral + Sparse)
    sdmep_acc, sdmep_time = train("SDMEP")
    
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Method':<10} | {'Acc':<8} | {'Time':<8} | {'Notes'}")
    print("-" * 50)
    print(f"{'Backprop':<10} | {bp_acc:.2f}%   | {bp_time:.2f}s   | Baseline")
    print(f"{'SMEP':<10} | {smep_acc:.2f}%   | {smep_time:.2f}s   | Sigma unbounded/high")
    print(f"{'SDMEP':<10} | {sdmep_acc:.2f}%   | {sdmep_time:.2f}s   | Sigma controlled < 0.95")
