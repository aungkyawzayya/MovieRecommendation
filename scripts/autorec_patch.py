"""Applies mini-batch training to core/models/autorec.py (run from repo root)."""
import pathlib, sys
p = pathlib.Path("core/models/autorec.py")
s = p.read_text()

# --- 1. batch_size parameter ---
old = """    def __init__(self, hidden_dim=100, lr=1e-3, weight_decay=1e-4,
                 epochs=300, patience=20, input_fill="zero", seed=STUDENT_ID):"""
new = """    def __init__(self, hidden_dim=100, lr=1e-3, weight_decay=1e-4,
                 epochs=300, patience=20, input_fill="zero", batch_size=256,
                 seed=STUDENT_ID):"""
assert old in s, "ctor signature not found"
s = s.replace(old, new)

old = """        self.input_fill = input_fill
        self.seed = seed"""
new = """        self.input_fill = input_fill
        # Mini-batching over ITEMS. Full-batch (batch_size=None) takes exactly
        # ONE gradient step per epoch, so epochs=80 meant 80 updates total for
        # a model with up to ~611k parameters — several configs early-stopped
        # after fewer than 10 updates, i.e. before training had begun. The
        # grid was then measuring how far each config happened to get, not its
        # capacity. At batch_size=256 an epoch is ~38 updates, each ~1/38 the
        # cost, so the SAME wall-clock budget buys ~38x more updates.
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1, or None for full batch")
        self.batch_size = batch_size
        self.seed = seed"""
assert old in s, "ctor body not found"
s = s.replace(old, new)

# --- 2. mini-batch training loop ---
old = """        for epoch in range(self.epochs):
            self._net.train()
            optimizer.zero_grad()
            reconstructed = self._net(X_tensor)
            loss = _masked_mse(reconstructed, X_tensor, mask_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())"""
new = """        n_samples = X_tensor.shape[0]
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)

        for epoch in range(self.epochs):
            self._net.train()
            # Shuffle item order each epoch so batches aren't always the same
            # grouping. torch.manual_seed() above makes randperm deterministic,
            # so runs stay reproducible.
            perm = torch.randperm(n_samples)
            epoch_loss_sum, epoch_batches = 0.0, 0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                reconstructed = self._net(X_tensor[idx])
                loss = _masked_mse(reconstructed, X_tensor[idx], mask_tensor[idx])
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
                epoch_batches += 1
            # One point per EPOCH (mean over its batches), so the loss curve
            # stays comparable to the full-batch version's one-point-per-epoch.
            train_losses.append(epoch_loss_sum / max(epoch_batches, 1))"""
assert old in s, "training loop not found"
s = s.replace(old, new)

# --- 3. score_all_items: return a copy, not a view into the cache ---
old = """        scores = self._reconstruction.loc[user_id]
        if clip:"""
new = """        # .copy() so a caller can never mutate the cached reconstruction
        # through the Series it gets back.
        scores = self._reconstruction.loc[user_id].copy()
        if clip:"""
assert old in s, "score_all_items not found"
s = s.replace(old, new)

p.write_text(s)
print("core/models/autorec.py patched: mini-batching + defensive copy")
