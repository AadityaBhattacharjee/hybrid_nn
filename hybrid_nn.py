"""
hybrid_nn.py
============
Hybrid Non-Backpropagation Neural Network
Combines:
  1. Hebbian local weight update + homeostatic decay
  2. Feedback Alignment (fixed random B matrix, no W^T)
  3. Evolutionary mutation on plateau detection
  4. Forward-only finite-difference Adam optimizer
  5. Dynamic sparse mask (top-k activation gating)

All weights stored in float16; gradient buffers in float32.
No autograd. No backward pass through the computation graph.
"""

import numpy as np
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0.0, x)

def relu_prime(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


ACTIVATIONS = {
    "relu":    (relu,    relu_prime),
    "sigmoid": (sigmoid, sigmoid_prime),
    "linear":  (lambda x: x, lambda x: np.ones_like(x)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer
# ─────────────────────────────────────────────────────────────────────────────

class Layer:
    """
    Single dense layer with:
      - float16 weights W, biases b
      - fixed random feedback matrix B (same shape as W)
      - float32 Adam moment buffers m_W, v_W, m_b, v_b
      - binary sparse mask S (updated externally)
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "relu",
                 seed: int = None):
        rng = np.random.default_rng(seed)

        # He init, cast to float16
        scale = np.sqrt(2.0 / n_in)
        self.W  = rng.normal(0, scale, (n_out, n_in)).astype(np.float16)
        self.b  = np.zeros(n_out, dtype=np.float16)

        # Fixed random feedback alignment matrix (never trained)
        self.B  = rng.normal(0, 0.1, (n_in, n_out)).astype(np.float32)

        # Sparse mask: 1 = active, 0 = masked
        self.S  = np.ones((n_out, n_in), dtype=np.float32)

        # Adam buffers (float32 for numerical stability)
        self.m_W = np.zeros_like(self.W, dtype=np.float32)
        self.v_W = np.zeros_like(self.W, dtype=np.float32)
        self.m_b = np.zeros_like(self.b, dtype=np.float32)
        self.v_b = np.zeros_like(self.b, dtype=np.float32)

        self.act_fn, self.act_prime = ACTIVATIONS[activation]
        self.activation = activation
        self.n_in  = n_in
        self.n_out = n_out

        # Cache for forward pass
        self.z    = None   # pre-activation
        self.x_in = None   # input to this layer

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, n_in)  or  (n_in,)
        returns: (batch, n_out)
        """
        self.x_in = x.astype(np.float32)
        W32 = self.W.astype(np.float32)
        self.z = self.x_in @ W32.T + self.b.astype(np.float32)  # (batch, n_out)
        out = self.act_fn(self.z)
        return out

    # ── Hebbian local update ─────────────────────────────────────────────────
    def hebbian_update(self, x_post: np.ndarray, eta_h: float,
                       lambda_decay: float) -> None:
        """
        ΔW_hebb = η_h · mean_batch(x_pre ⊗ x_post) · S_mask
        W ← W + ΔW_hebb − λ_decay · W
        """
        x_pre  = self.x_in.astype(np.float32)          # (batch, n_in)
        x_post = x_post.astype(np.float32)              # (batch, n_out)

        # Outer product averaged over batch
        dW = (x_post.T @ x_pre) / x_pre.shape[0]       # (n_out, n_in)
        dW *= self.S                                     # sparse gating

        W32 = self.W.astype(np.float32)
        W32 += eta_h * dW - lambda_decay * W32
        self.W = W32.astype(np.float16)

    # ── Feedback Alignment update ─────────────────────────────────────────────
    def fa_update(self, delta_next: np.ndarray, eta_fa: float) -> np.ndarray:
        """
        δ_FA[l] = B[l] · δ[l+1] · f′(z[l])
        ΔW_FA   = η_FA · δ_FA ⊗ x[l−1]
        Returns δ_FA to propagate further back.
        """
        # delta_next: (batch, n_out)  error signal from above
        delta_fa = (delta_next @ self.B.T) * self.act_prime(self.z)  # (batch, n_in)

        # Weight update using FA signal on THIS layer's output error
        x_pre = self.x_in.astype(np.float32)
        dW = (delta_next.T @ x_pre) / x_pre.shape[0]   # (n_out, n_in)
        dW *= self.S

        W32 = self.W.astype(np.float32)
        W32 += eta_fa * dW
        self.W = W32.astype(np.float16)

        db = delta_next.mean(axis=0)
        self.b = (self.b.astype(np.float32) + eta_fa * db).astype(np.float16)

        return delta_fa   # pass to layer below

    # ── Adam forward-diff optimizer ───────────────────────────────────────────
    def adam_step(self, loss_fn, eta: float, epsilon_fd: float,
                  beta1: float, beta2: float, epsilon_adam: float,
                  t: int) -> None:
        """
        Estimate ∂L/∂W[i,j] via finite difference for masked indices only.
        Apply bias-corrected Adam update. Quantize back to float16.
        """
        W32  = self.W.astype(np.float32)
        b32  = self.b.astype(np.float32)
        L0   = loss_fn()

        grad_W = np.zeros_like(W32)
        grad_b = np.zeros_like(b32)

        # Iterate only over active mask positions
        active_idx = np.argwhere(self.S > 0)
        for (i, j) in active_idx:
            orig = W32[i, j]
            W32[i, j] += epsilon_fd
            self.W = W32.astype(np.float16)
            L1 = loss_fn()
            grad_W[i, j] = (L1 - L0) / epsilon_fd
            W32[i, j] = orig
            self.W = W32.astype(np.float16)

        for k in range(len(b32)):
            orig = b32[k]
            b32[k] += epsilon_fd
            self.b = b32.astype(np.float16)
            L1 = loss_fn()
            grad_b[k] = (L1 - L0) / epsilon_fd
            b32[k] = orig
            self.b = b32.astype(np.float16)

        # Adam moments
        self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (grad_W ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)

        # Bias correction
        m_W_hat = self.m_W / (1 - beta1 ** t)
        v_W_hat = self.v_W / (1 - beta2 ** t)
        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)

        # Update + quantize to float16
        W32 -= eta * m_W_hat / (np.sqrt(v_W_hat) + epsilon_adam)
        b32 -= eta * m_b_hat / (np.sqrt(v_b_hat) + epsilon_adam)
        self.W = W32.astype(np.float16)
        self.b = b32.astype(np.float16)

    # ── Sparse mask update ────────────────────────────────────────────────────
    def update_mask(self, x_post: np.ndarray, top_k: float) -> None:
        """
        Rank neurons by mean activation magnitude.
        Keep top-k% active, mask the rest.
        top_k: fraction in (0,1]
        """
        mean_act = np.abs(x_post).mean(axis=0)          # (n_out,)
        k = max(1, int(np.ceil(top_k * self.n_out)))
        threshold = np.sort(mean_act)[-k]
        active_neurons = (mean_act >= threshold)         # (n_out,)

        # Broadcast: row i of S is 1 only if neuron i is active
        self.S = active_neurons[:, None].astype(np.float32) * \
                 np.ones((self.n_out, self.n_in), dtype=np.float32)

    # ── Evolutionary mutation ─────────────────────────────────────────────────
    def mutate(self, sigma: float) -> "Layer":
        """
        Return a mutated copy of this layer.
        Noise applied only to masked (active) positions.
        """
        clone = deepcopy(self)
        noise = np.random.normal(0, sigma, self.W.shape).astype(np.float32)
        noise *= self.S
        clone.W = (self.W.astype(np.float32) + noise).astype(np.float16)
        return clone


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class HybridNN:
    """
    Multi-layer hybrid non-backpropagation neural network.

    Architecture: list of (n_units, activation) tuples.
    Example: [(6,), (2, 'relu'), (1, 'sigmoid')]
              input   hidden      output

    Training phases per epoch:
      1. Forward pass
      2. Hebbian update (local, all layers)
      3. Feedback Alignment (supervised, all layers)
      4. Blend updates
      5. Plateau detection → evolutionary mutation
      6. Forward-diff Adam step (every N_opt epochs)
      7. Sparse mask refresh (every n_refresh epochs)
    """

    def __init__(self,
                 layer_sizes: list,          # e.g. [6, 2, 1]
                 activations: list = None,   # per hidden+output layer, e.g. ['relu','sigmoid']
                 # Phase 1 — Hebbian
                 eta_h:        float = 0.010,
                 lambda_decay: float = 3e-4,
                 # Phase 2 — FA
                 eta_fa:       float = 0.005,
                 alpha:        float = 0.40,
                 # Phase 3 — Evolution
                 sigma_0:      float = 0.008,
                 gamma:        float = 5.0,
                 plateau_k:    int   = 8,
                 eps_plateau:  float = 0.02,
                 # Phase 4 — Adam
                 eta_adam:     float = 1e-3,
                 beta1:        float = 0.9,
                 beta2:        float = 0.999,
                 eps_adam:     float = 1e-7,
                 eps_fd:       float = 1e-3,
                 N_opt:        int   = 5,
                 # Sparse mask
                 top_k:        float = 0.30,
                 n_refresh:    int   = 10,
                 # Task
                 task:         str   = "binary",  # "binary" | "multiclass"
                 seed:         int   = 42):

        if activations is None:
            # default: relu hidden, sigmoid/softmax output
            activations = ["relu"] * (len(layer_sizes) - 2)
            activations += ["sigmoid" if task == "binary" else "linear"]

        assert len(activations) == len(layer_sizes) - 1, \
            "Need one activation per layer transition"

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i+1],
                      activation=activations[i], seed=seed + i)
            )

        self.eta_h        = eta_h
        self.lambda_decay = lambda_decay
        self.eta_fa       = eta_fa
        self.alpha        = alpha
        self.sigma_0      = sigma_0
        self.gamma        = gamma
        self.plateau_k    = plateau_k
        self.eps_plateau  = eps_plateau
        self.eta_adam     = eta_adam
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps_adam     = eps_adam
        self.eps_fd       = eps_fd
        self.N_opt        = N_opt
        self.top_k        = top_k
        self.n_refresh    = n_refresh
        self.task         = task

        self.loss_history  = []
        self.epoch         = 0
        self._adam_t       = 1          # Adam time step
        self._mut_count    = 0

    # ── Inference ─────────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> list:
        """
        Run forward pass. Returns list of activations per layer
        (index 0 = first hidden, last = output).
        """
        activations = []
        h = X.astype(np.float32)
        for layer in self.layers:
            h = layer.forward(h)
            activations.append(h)
        return activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        acts = self.forward(X)
        out  = acts[-1]
        if self.task == "binary":
            return (out.squeeze() >= 0.5).astype(int)
        else:
            return out.argmax(axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        acts = self.forward(X)
        out  = acts[-1]
        if self.task == "multiclass":
            return softmax(out)
        return out.squeeze()

    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        acts = self.forward(X)
        out  = acts[-1]
        eps  = 1e-7
        if self.task == "binary":
            p = np.clip(out.squeeze(), eps, 1 - eps)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        else:
            p = np.clip(softmax(out), eps, 1 - eps)
            idx = np.arange(len(y))
            return -np.mean(np.log(p[idx, y]))

    # ── Output delta ──────────────────────────────────────────────────────────
    def _output_delta(self, y: np.ndarray, acts: list) -> np.ndarray:
        """
        Cross-entropy + sigmoid/softmax gradient: δ = ŷ − y
        """
        out = acts[-1]
        if self.task == "binary":
            return (out.squeeze()[:, None] if out.ndim == 1 else out) - \
                   y.reshape(-1, 1).astype(np.float32)
        else:
            p = softmax(out)
            one_hot = np.zeros_like(p)
            one_hot[np.arange(len(y)), y] = 1.0
            return p - one_hot

    # ── Single training step (one mini-batch) ─────────────────────────────────
    def _train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        # ── Phase 1 & 2 combined ──────────────────────────────────────────────
        # Forward pass — caches x_in and z in each layer
        acts = self.forward(X)
        loss = self._loss(X, y)

        # Hebbian update (local, no error signal)
        # We snapshot the current weights before any update
        W_before = [l.W.copy() for l in self.layers]

        for i, layer in enumerate(self.layers):
            layer.hebbian_update(acts[i], self.eta_h, self.lambda_decay)

        # Feedback Alignment — propagate output delta backwards using B matrices
        delta = self._output_delta(y, acts)        # (batch, n_out_last)
        fa_deltas = [None] * len(self.layers)
        fa_deltas[-1] = delta

        # Store FA weight changes separately
        W_fa_after = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # Restore weights to pre-Hebb so FA acts independently
            layer.W = W_before[i].copy()

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            d = fa_deltas[i]
            delta_below = layer.fa_update(d, self.eta_fa)
            if i > 0:
                fa_deltas[i - 1] = delta_below
            W_fa_after.append(layer.W.copy())
        W_fa_after = list(reversed(W_fa_after))

        # ── Phase blending: α·Hebb + (1-α)·FA ───────────────────────────────
        for i, layer in enumerate(self.layers):
            W_hebb_i = W_before[i].astype(np.float32) + \
                       self.eta_h * (np.ones_like(W_before[i], dtype=np.float32))
            # Recover actual hebb delta
            acts2 = self.forward(X)
            # Simple blend: take Hebb result and FA result, mix
            W_h = W_before[i].astype(np.float32)
            W_f = W_fa_after[i].astype(np.float32)
            # Run hebb from scratch cleanly
            # (already stored in layer from above steps — recompute cleanly)
            layer.W = W_before[i].copy()
            layer.forward(X if i == 0 else acts[i-1])
            layer.hebbian_update(acts[i], self.eta_h, self.lambda_decay)
            W_h = layer.W.astype(np.float32)

            W_blend = self.alpha * W_h + (1 - self.alpha) * W_f
            layer.W = W_blend.astype(np.float16)

        return loss

    # ── Full epoch ────────────────────────────────────────────────────────────
    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 32) -> float:
        self.epoch += 1
        n = len(X)
        idx = np.random.permutation(n)
        X, y = X[idx], y[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            Xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            epoch_loss += self._train_step(Xb, yb)
            n_batches  += 1

        epoch_loss /= n_batches
        self.loss_history.append(epoch_loss)

        # ── Phase 3: Plateau detection + evolution ────────────────────────────
        acc = self._compute_acc(X, y)
        if len(self.loss_history) >= self.plateau_k:
            delta_L = abs(self.loss_history[-1] -
                          self.loss_history[-self.plateau_k])
            if delta_L < self.eps_plateau and acc < 0.98:
                sigma = self.sigma_0 * np.exp(-self.gamma * acc)
                self._evolve(X, y, sigma, acc)

        # ── Phase 4: Forward-diff Adam (every N_opt epochs) ───────────────────
        if self.epoch % self.N_opt == 0:
            for layer in self.layers:
                def make_loss_fn(lay, Xb=X[:batch_size], yb=y[:batch_size]):
                    def fn():
                        self.forward(Xb)   # updates caches
                        return self._loss(Xb, yb)
                    return fn
                layer.adam_step(
                    make_loss_fn(layer),
                    self.eta_adam, self.eps_fd,
                    self.beta1, self.beta2, self.eps_adam,
                    self._adam_t
                )
            self._adam_t += 1

        # ── Phase 7: Sparse mask refresh ─────────────────────────────────────
        if self.epoch % self.n_refresh == 0:
            acts = self.forward(X[:min(256, n)])
            for i, layer in enumerate(self.layers):
                layer.update_mask(acts[i], self.top_k)

        return epoch_loss

    def _compute_acc(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.squeeze())

    def _evolve(self, X, y, sigma, acc):
        """(1+1)-ES: mutate all layers, accept if loss improves."""
        L_before = self._loss(X, y)
        clones    = [layer.mutate(sigma) for layer in self.layers]
        orig_layers = self.layers
        self.layers = clones
        L_after = self._loss(X, y)
        if L_after < L_before:
            self._mut_count += 1
        else:
            self.layers = orig_layers   # reject

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        loss  = self._loss(X, y)
        preds = self.predict(X)
        acc   = np.mean(preds == y.squeeze())
        return {"loss": loss, "accuracy": acc}

    def score(self, X, y):
        return self.evaluate(X, y)["accuracy"]
