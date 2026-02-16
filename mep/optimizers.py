import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Tuple, List, Iterable

class EPWrapper:
    """Wraps the model to add automatic free-phase settling in EP mode."""
    def __init__(self, model: nn.Module, optimizer: 'SMEPOptimizer'):
        self.model = model
        self.optimizer = optimizer
        self.original_forward = model.forward
        self.free_states = []
        self.last_input = None

    def forward(self, x, phase='free', target=None, **kwargs):
        # If optimizer is in backprop mode, just pass through
        if self.optimizer.defaults['mode'] == 'backprop':
            return self.original_forward(x, **kwargs)

        if phase == 'free':
            self.last_input = x
            # Run settling
            states = self.optimizer._settle(
                self.model, x, target=None, beta=0.0,
                forward_fn=self.original_forward
            )
            self.free_states = states
            return states[-1] # Return output

        elif phase == 'nudged':
            # Nudged phase
            states = self.optimizer._settle(
                self.model, x, target=target, beta=self.optimizer.defaults['beta'],
                forward_fn=self.original_forward
            )
            return states

        else:
            return self.original_forward(x, **kwargs)

class SMEPOptimizer(Optimizer):
    """
    SMEP Optimizer: Spectral Muon Equilibrium Propagation
    
    A self-contained optimizer that:
    - Computes gradients via Equilibrium Propagation OR standard backprop
    - Applies Muon (Newton-Schulz) orthogonalization to weight updates
    - Optional: Error Feedback (continual learning)
    - Optional: Spectral Constraints (Lipschitz control)
    - Works with arbitrary PyTorch models (drop-in replacement for SGD/Adam)
    """
    
    # Constants for numerical stability
    EPSILON_NORM = 1e-6  # Small value to prevent division by zero
    EPSILON_SPECTRAL = 1e-6  # Epsilon for spectral norm power iteration
    SETTLING_MOMENTUM = 0.5  # Momentum for state settling optimizer
    SPECTRAL_POWER_ITER = 3  # Number of power iterations for spectral norm
    def __init__(
        self, 
        params: Iterable, 
        model: Optional[nn.Module] = None,
        lr: float = 0.02, 
        momentum: float = 0.9, 
        wd: float = 0.0005,
        mode: str = 'backprop',
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        ns_steps: int = 5,
        use_error_feedback: bool = True,
        error_beta: float = 0.9,
        use_spectral_constraint: bool = True,
        gamma: float = 0.95
    ):
        """
        Initialize SMEPOptimizer.

        Args:
            params: Iterable of parameters.
            model: Optional model instance. Required for new EP API (mode='ep').
            lr: Learning rate.
            momentum: Momentum factor.
            wd: Weight decay.
            mode: 'backprop' or 'ep'.
            beta: Nudging strength for EP.
            settle_steps: Number of settling steps for EP.
            settle_lr: Learning rate for settling optimization.
            ns_steps: Newton-Schulz iterations.
            use_error_feedback: Enable error feedback.
            error_beta: Error feedback decay.
            use_spectral_constraint: Enable spectral constraint.
            gamma: Spectral norm bound.
        """
        defaults = dict(
            lr=lr, momentum=momentum, wd=wd, ns_steps=ns_steps,
            mode=mode, beta=beta, settle_steps=settle_steps, settle_lr=settle_lr,
            use_error_feedback=use_error_feedback, error_beta=error_beta,
            use_spectral_constraint=use_spectral_constraint, gamma=gamma
        )
        super().__init__(params, defaults)
        
        # Validate mode
        if mode not in ['backprop', 'ep']:
            raise ValueError(f"mode must be 'backprop' or 'ep', got {mode}")
        
        # Cache for model structure to avoid repeated introspection
        self._model_structure_cache = {}

        self.model = model
        self.ep_wrapper = None

        # Check if already wrapped and unwrap if so (to avoid nesting and ensure correct optimizer ref)
        if self.model is not None:
            if hasattr(self.model.forward, '__self__') and isinstance(self.model.forward.__self__, EPWrapper):
                old_wrapper = self.model.forward.__self__
                self.model.forward = old_wrapper.original_forward

        if mode == 'ep' and self.model is not None:
            # One-time wrap
            self.ep_wrapper = EPWrapper(self.model, self)
            self.model.forward = self.ep_wrapper.forward

    @torch.no_grad()
    def newton_schulz(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Newton-Schulz orthogonalization (Muon update).
        
        Args:
            G: Gradient tensor (must be 2D for orthogonalization)
            steps: Number of Newton-Schulz iterations
            
        Returns:
            Orthogonalized gradient tensor
        """
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
        norm = X.norm().clamp(min=1e-4, max=1e4) + self.EPSILON_NORM
        X.div_(norm)

        # NS Iteration: X = 0.5 * X * (3I - X^T X)
        I = torch.eye(c, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * I - A)

        res = X * norm # Restore scale

        if transposed:
            return res.T
        return res
        
    @torch.no_grad()
    def get_spectral_norm(
        self, 
        W: torch.Tensor, 
        u: Optional[torch.Tensor], 
        v: Optional[torch.Tensor], 
        iter: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spectral norm via power iteration.
        
        Args:
            W: Weight matrix
            u: Left singular vector (cached)
            v: Right singular vector (cached)
            iter: Number of power iterations (defaults to class constant)
            
        Returns:
            Tuple of (spectral_norm, updated_u, updated_v)
        """
        if iter is None:
            iter = self.SPECTRAL_POWER_ITER
            
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)
        
        h, w = W.shape
        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u /= u.norm()
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v /= v.norm()
        
        for _ in range(iter):
            v = W.T @ u
            v /= v.norm() + self.EPSILON_SPECTRAL
            u = W @ v
            u /= u.norm() + self.EPSILON_SPECTRAL
        
        sigma = (u @ W @ v).abs()
        return sigma, u, v

    def _prepare_target(self, target: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
        """Convert target to one-hot if needed, matching dtype."""
        if target.dim() == 1:
            one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=dtype)
            return one_hot
        return target.to(dtype=dtype)
    
    def _inspect_model(self, model: nn.Module) -> List[dict]:
        """Extract sequence of layers and activations (cached)."""
        model_id = id(model)
        if model_id in self._model_structure_cache:
            return self._model_structure_cache[model_id]
        
        structure = []
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                structure.append({'type': 'layer', 'module': m})
            elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax)):
                structure.append({'type': 'act', 'module': m})
        
        self._model_structure_cache[model_id] = structure
        return structure

    def _compute_energy(self, model, x, states, structure, target_vec=None, beta=0.0) -> torch.Tensor:
        """
        Compute total energy: E = E_int + E_ext
        E_int = 0.5 * mean_over_batch sum_over_features || s_i - f_i(s_{i-1}) ||^2
        E_ext = beta * Loss(s_last, target)
        
        Note: Energy is normalized by batch size to ensure batch-size invariance.
        """
        batch_size = x.shape[0]
        E = 0.0
        prev = x
        state_idx = 0
        
        # Iterate through structure to reconstruct graph
        for item in structure:
            if item['type'] == 'layer':
                if state_idx >= len(states): break
                
                module = item['module']
                state = states[state_idx]
                
                # Prediction from previous state
                h = module(prev)
                
                # Energy mismatch (mean over batch for stability)
                E = E + 0.5 * ((h - state)**2).sum() / batch_size
                
                # Next input base is current state
                prev = state
                state_idx += 1
                
            elif item['type'] == 'act':
                # Apply activation to current flow
                prev = item['module'](prev)
        
        # Nudge term (consistent reduction with E_int)
        if target_vec is not None and beta > 0:
            output = prev
            # Use reduction='sum' then divide by batch_size for consistency
            E = E + beta * F.mse_loss(output, target_vec, reduction='sum') / batch_size
            
        return E
        
    def _settle(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        target: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        forward_fn = None
    ) -> List[torch.Tensor]:
        """
        Settle network activations to minimize energy.
        """
        if forward_fn is None:
            forward_fn = model

        # Introspect model
        structure = self._inspect_model(model)
        
        # Capture initial states via forward pass
        states = []
        handles = []
        
        def capture_hook(module, input, output):
            states.append(output.detach().clone().requires_grad_(True))
        
        for item in structure:
            if item['type'] == 'layer':
                handles.append(item['module'].register_forward_hook(capture_hook))
        
        try:
            with torch.no_grad():
                forward_fn(x)
        finally:
            for h in handles:
                h.remove()
        
        if not states:
            # Provide helpful debugging information
            layer_count = sum(1 for item in structure if item['type'] == 'layer')
            raise RuntimeError(
                f"No activations captured during settling. "
                f"Expected {layer_count} layer(s) but got 0. \n"
                f"Model type: {type(model).__name__}\n"
                f"Model structure contains {len(structure)} items: "
                f"{', '.join(item['type'] for item in structure[:5])}{'...' if len(structure) > 5 else ''}\n"
                f"Ensure model contains nn.Linear or nn.Conv2d layers."
            )
            
        # Optimization loop
        # Use SGD to relax states (lr now configurable)
        state_optimizer = torch.optim.SGD(
            states, 
            lr=self.defaults['settle_lr'], 
            momentum=self.SETTLING_MOMENTUM
        )
        
        # Prepare target vector if needed
        target_vec = None
        if target is not None:
            # Use states[-1] dtype to ensure target matches computation precision (e.g. FP16)
            target_vec = self._prepare_target(target, states[-1].shape[-1], dtype=states[-1].dtype)
        
        for _ in range(self.defaults['settle_steps']):
            state_optimizer.zero_grad()
            with torch.enable_grad():
                E = self._compute_energy(model, x, states, structure, target_vec, beta)
                grads = torch.autograd.grad(E, states, retain_graph=False)

            for state, g in zip(states, grads):
                if state.grad is None:
                    state.grad = g.detach()
                else:
                    state.grad.copy_(g.detach())
            state_optimizer.step()
            
        return [s.detach() for s in states]

    def _apply_ep_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: List[dict]
    ):
        """
        Compute and accumulate EP gradients given free and nudged states.
        """
        # Prepare target (use helper, matching free state precision)
        target_vec = self._prepare_target(target, states_free[-1].shape[-1], dtype=states_free[-1].dtype)

        # E_free
        E_free = self._compute_energy(model, x, states_free, structure, target_vec=None, beta=0.0)
        
        # E_nudged
        E_nudged = self._compute_energy(model, x, states_nudged, structure, target_vec, beta=self.defaults['beta'])
        
        grads_free = torch.autograd.grad(E_free, model.parameters(), retain_graph=False, allow_unused=True)
        grads_nudged = torch.autograd.grad(E_nudged, model.parameters(), retain_graph=False, allow_unused=True)
        
        for p, g_free, g_nudged in zip(model.parameters(), grads_free, grads_nudged):
            if g_free is None or g_nudged is None: continue
            ep_grad = (g_nudged - g_free) / self.defaults['beta']
            if p.grad is None:
                p.grad = ep_grad.detach()
            else:
                p.grad.add_(ep_grad.detach())

    def _compute_ep_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor
    ):
        # 1. Inspect
        structure = self._inspect_model(model)

        # 2. Free Phase
        states_free = self._settle(model, x, target=None, beta=0.0)

        # 3. Nudged Phase
        states_nudged = self._settle(model, x, target=target, beta=self.defaults['beta'])

        # 4. Compute Gradients via Contrast
        self._apply_ep_gradients(model, x, target, states_free, states_nudged, structure)

    @torch.no_grad()
    def step(
        self, 
        x: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        closure=None
    ):
        """
        Perform optimization step.

        New EP API:
            optimizer = SMEPOptimizer(..., model=model, mode='ep')
            output = model(x)  # Triggers free phase settling
            optimizer.step(target=y)  # Triggers nudged phase & update

        Legacy EP API:
            optimizer = SMEPOptimizer(..., mode='ep')
            optimizer.step(x=x, target=y, model=model)

        Backprop API:
            optimizer = SMEPOptimizer(..., mode='backprop')
            loss.backward()
            optimizer.step()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # If mode='ep', compute gradients via EP
        mode = self.defaults['mode']
        if mode == 'ep':
            if x is None:
                # New mode: check if model wrapped and target present
                if self.model is None or self.ep_wrapper is None:
                     # Fallback check: maybe user forgot x but didn't wrap model?
                     raise ValueError("For EP mode, pass model=model at optimizer creation OR pass x, target, model to step()")

                if target is None:
                     raise ValueError("EP mode requires target=y in step(target=y)")

                x_input = self.ep_wrapper.last_input
                if x_input is None:
                     raise RuntimeError("In EP mode with wrapped model, you must call model(x) before optimizer.step(target=y)")

                with torch.enable_grad():
                    # Nudged phase
                    states_nudged = self.ep_wrapper.forward(x_input, phase='nudged', target=target)
                    states_free = self.ep_wrapper.free_states

                    # Compute gradients
                    structure = self._inspect_model(self.model)
                    self._apply_ep_gradients(self.model, x_input, target, states_free, states_nudged, structure)

            else:
                # Legacy mode
                if target is None or model is None:
                    # If model was passed in init, use it as default
                    model = model or self.model
                    if model is None or target is None:
                        raise ValueError("mode='ep' requires x, target, and model arguments")

                # Temporarily enable gradients for EP computation
                with torch.enable_grad():
                    self._compute_ep_gradients(model, x, target)
        
        # Apply Muon updates (works for both backprop and EP gradients)
        for group in self.param_groups:
            # Initialize momentum buffers
            for p in group['params']:
                if p.grad is None: continue
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                if 'error_buffer' not in state:
                    state['error_buffer'] = torch.zeros_like(p)
                if 'u_spec' not in state:
                    state['u_spec'] = None
                if 'v_spec' not in state:
                    state['v_spec'] = None
                
                g = p.grad.data
                
                # --- Error Feedback ---
                if group['use_error_feedback']:
                    g_aug = g + group['error_beta'] * state['error_buffer']
                else:
                    g_aug = g
                    # Clear buffer if unused
                    state['error_buffer'].zero_()

                update = g_aug.clone()
                
                # --- Update Calculation (Muon / Dion) ---
                if p.ndim >= 2:
                    orig_shape = p.shape
                    if p.ndim > 2:
                        g_flat = g_aug.view(g_aug.shape[0], -1)
                    else:
                        g_flat = g_aug
                    
                    # Override hook for subclasses (e.g. Dion)
                    update_flat = self._compute_update(p, g_flat, group, state, g_aug, orig_shape)
                    
                    if p.ndim > 2:
                        update = update_flat.view(orig_shape)
                    else:
                        update = update_flat
                else:
                    update = g
                
                # --- Momentum ---
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(update)
                
                # --- Weight Decay + Apply Update ---
                p.data.mul_(1 - group['lr'] * group['wd'])
                p.data.add_(buf, alpha=-group['lr'])
                
                # --- Spectral Constraint ---
                if group['use_spectral_constraint'] and p.ndim >= 2:
                    sigma, u, v = self.get_spectral_norm(p.data, state['u_spec'], state['v_spec'])
                    state['u_spec'] = u
                    state['v_spec'] = v
                    if sigma > group['gamma']:
                        p.data.mul_(group['gamma'] / sigma)
        
        return loss

    def _compute_update(self, p, g_flat, group, state, g_aug, orig_shape):
        """
        Default update strategy: Pure Muon (Newton-Schulz).
        Subclasses (SDMEP) can override this to add Dion logic.
        """
        # Default: Muon for everything
        update_flat = self.newton_schulz(g_flat, group['ns_steps'])
        
        # When using pure Muon, error buffer is zeroed (no residual)
        # unless we want to keep it? Usually Muon doesn't use error feedback 
        # unless we are approximating (like Dion).
        # But if Error Feedback is enabled globally, we should respect it.
        # But Muon is full rank, so there is no "residual" in the low-rank sense.
        # So we zero it out for full rank updates.
        state['error_buffer'].zero_()
        
        return update_flat


class SDMEPOptimizer(SMEPOptimizer):
    """
    SDMEP Optimizer: Spectral Dion-Muon Equilibrium Propagation
    
    This is SMEP but with Dion (Low-Rank) updates for large matrices.
    """
    def __init__(
        self, 
        params: Iterable, 
        model: Optional[nn.Module] = None,
        lr: float = 0.02, 
        momentum: float = 0.9, 
        wd: float = 0.0005,
        gamma: float = 0.95, 
        rank_frac: float = 0.2, 
        error_beta: float = 0.9, 
        dion_thresh: int = 100000,
        mode: str = 'backprop',
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        ns_steps: int = 5,
        use_error_feedback: bool = True,
        use_spectral_constraint: bool = True
    ):
        # Call parent (SMEPOptimizer) constructor
        super().__init__(
            params=params,
            model=model,
            lr=lr,
            momentum=momentum,
            wd=wd,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            ns_steps=ns_steps,
            use_error_feedback=use_error_feedback,
            error_beta=error_beta,
            use_spectral_constraint=use_spectral_constraint,
            gamma=gamma
        )
        
        # Add SDMEP-specific parameters to defaults
        for group in self.param_groups:
            group['rank_frac'] = rank_frac
            group['dion_thresh'] = dion_thresh

    def _compute_update(self, p, g_flat, group, state, g_aug, orig_shape):
        """
        Override: Use Dion for large matrices, Muon for small ones.
        """
        if p.numel() > group['dion_thresh']:
            # --- DION (Low-rank SVD) ---
            rank = max(1, int(min(g_flat.shape) * group['rank_frac']))
            # U: (M, r), S: (r,), V: (N, r)
            U, S, V = torch.svd_lowrank(g_flat, q=rank)
            
            # Reconstruct: U @ diag(S) @ V.T
            # Efficiently: (U * S) @ V.T
            update_lowrank = (U * S.unsqueeze(0)) @ V.T
            
            # Error Feedback: Update buffer with residual
            state['error_buffer'] = g_aug - update_lowrank.view(orig_shape)
            
            return update_lowrank.view(g_flat.shape)
        else:
            # --- MUON (Newton-Schulz) ---
            # Use parent implementation
            update_flat = self.newton_schulz(g_flat, group['ns_steps'])
            
            # Full rank -> no error
            state['error_buffer'].zero_()
            
            return update_flat
