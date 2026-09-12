"""
autorec.py
I-AutoRec (Sedhain et al., 2015) — an item-based autoencoder. Each MOVIE is
one training sample: its rating vector across all 610 users is fed through
a bottleneck (encoder -> sigmoid -> k hidden units -> decoder -> identity)
and the network learns to reconstruct it. This is a non-linear alternative
to SVD's linear factorization, trained on the exact same train_matrix.

C# analogy for the split below: _AutoRecNet is a private implementation
detail (like a class only used inside another class's .cs file), and
AutoRecRecommender is the public type other code actually depends on — it
USES torch, it isn't a torch.nn.Module itself (has-a, not is-a), so the
rest of the project never needs to import torch to call recommend_top_n().
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.data.preprocess import STUDENT_ID


class _AutoRecNet(nn.Module):
    """The network itself. Architecture from the paper: no bias term is
    skipped, no activation on the output — ratings are a bounded but
    continuous target, and the paper found identity outperforms sigmoid
    there (sigmoid would need a 0-1 rescale on top for no measured benefit)."""

    def __init__(self, n_users, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(n_users, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_users)

    def forward(self, x):
        hidden = torch.sigmoid(self.encoder(x))
        return self.decoder(hidden)  # identity — no activation on the output


def _masked_mse(pred, target, mask):
    """
    Mean squared error over OBSERVED entries only (mask=True), matching
    the paper's objective. Without the mask, every unrated (user, item)
    pair would silently become a target of 0 (or the fill value), and the
    network would learn "predict low ratings everywhere" instead of
    "reconstruct what you saw" — the exact failure mode 0-filling this
    project's SVD matrix without a mask would also cause.
    """
    diff = (pred - target) * mask
    denom = mask.sum().clamp(min=1)
    return (diff ** 2).sum() / denom


class AutoRecRecommender:
    # AutoRec reconstructs star ratings directly (unlike ContentBasedRecommender's
    # cosine similarity or HybridRecommender's blended z-score), so RMSE/MAE
    # against the 0.5-5.0 rating column is a meaningful comparison.
    produces_ratings = True

    def __init__(self, hidden_dim=100, lr=1e-3, weight_decay=1e-4,
                 epochs=300, patience=20, input_fill="zero", batch_size=None,
                 seed=STUDENT_ID):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        # Parameters (n_users*hidden_dim*2 roughly) can easily outnumber the
        # observed ratings on a dataset this size (see the project notes'
        # capacity-vs-data-size discussion) — early stopping on validation
        # loss is not optional polish here, it is what keeps the model from
        # memorizing the train matrix outright.
        self.patience = patience
        if input_fill not in ("zero", "mean"):
            raise ValueError('input_fill must be "zero" or "mean"')
        self.input_fill = input_fill
        # Mini-batching over ITEMS. Full-batch (batch_size=None) takes exactly
        # ONE gradient step per epoch, so epochs=80 meant 80 updates total for
        # a model with up to ~611k parameters — several configs early-stopped
        # after fewer than 10 updates, i.e. before training had begun. The
        # grid was then measuring how far each config happened to get, not its
        # capacity. At batch_size=256 an epoch is ~38 updates, each ~1/38 the
        # cost, so the SAME wall-clock budget buys ~38x more updates.
        # Measured outcome: at this dataset's scale mini-batching did NOT
        # help (best val RMSE 0.8815 at batch=256/lr=1e-3 vs 0.8616
        # full-batch/lr=1e-2), because 38x more updates at the same lr
        # overshoots into overfitting within 1-3 epochs. Default is None
        # (full batch) for that reason; the option stays so the comparison
        # can be reported as an ablation.
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1, or None for full batch")
        self.batch_size = batch_size
        self.seed = seed

        self._net = None
        self._user_ids = None
        self._movie_ids = None
        self._seen_mask = None       # which (user, movie) pairs were in train
        self._item_support = None    # how many TRAIN ratings each item has
        self._reconstruction = None  # (n_users x n_movies) DataFrame, cached
        self.train_losses_ = None    # exposed for the loss-curve graph
        self.val_losses_ = None
        self.best_epoch_ = None

    def fit(self, train_matrix, val_df=None):
        """
        train_matrix: same (users x movies) grid as SVDRecommender.fit().
        val_df: optional long-form (userId, movieId, rating) frame used
                ONLY to decide when to stop training — never fed into the
                network as input. Feeding val ratings into the input vector
                would let the network "peek" at what it's being scored on
                (the exact 2-way-split leakage this project's Step 1
                already moved away from, reappearing inside one model).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._user_ids = train_matrix.index.to_numpy()
        self._movie_ids = train_matrix.columns.to_numpy()
        self._seen_mask = train_matrix.notna().to_numpy()
        self._item_support = self._seen_mask.sum(axis=0)

        n_users = len(self._user_ids)
        n_items = len(self._movie_ids)

        # Item-based: each MOVIE is a training sample, its 610 ratings are
        # the features. The fill value only matters for what the network
        # SEES as input — the loss below is masked, so it's never asked to
        # reproduce the fill value as a target either way.
        X_dense = train_matrix.to_numpy(dtype=np.float32)
        if self.input_fill == "zero":
            X_dense = np.nan_to_num(X_dense, nan=0.0)
        else:
            # Per-USER mean (row mean), computed from TRAIN ratings only —
            # a per-ITEM mean would leak that item's own rating distribution
            # into the very feature vector being reconstructed for it.
            row_means = np.nanmean(X_dense, axis=1).astype(np.float32)
            nan_rows, nan_cols = np.where(np.isnan(X_dense))
            X_dense[nan_rows, nan_cols] = row_means[nan_rows]

        X = X_dense.T  # (n_items, n_users) — item-based orientation
        mask_train = self._seen_mask.T  # (n_items, n_users)

        X_tensor = torch.from_numpy(X)
        mask_tensor = torch.from_numpy(mask_train)

        # --- Validation targets, same (item x user) grid, positions only ---
        val_target = np.zeros((n_items, n_users), dtype=np.float32)
        val_mask = np.zeros((n_items, n_users), dtype=bool)
        if val_df is not None and len(val_df) > 0:
            item_pos = {mid: i for i, mid in enumerate(self._movie_ids)}
            user_pos = {uid: i for i, uid in enumerate(self._user_ids)}
            for row in val_df.itertuples(index=False):
                i = item_pos.get(row.movieId)
                u = user_pos.get(row.userId)
                if i is not None and u is not None:
                    val_target[i, u] = row.rating
                    val_mask[i, u] = True
        val_target_tensor = torch.from_numpy(val_target)
        val_mask_tensor = torch.from_numpy(val_mask)
        has_val = bool(val_mask.any())

        self._net = _AutoRecNet(n_users, self.hidden_dim)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_losses, val_losses = [], []
        best_val_loss, best_state, best_epoch = float("inf"), None, 0
        epochs_since_improve = 0

        n_samples = X_tensor.shape[0]
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
            train_losses.append(epoch_loss_sum / max(epoch_batches, 1))

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    # SAME train-only input X_tensor — only the TARGET and
                    # MASK are validation-specific. Never train=X, target=X_val
                    # with X_val as input; see the fit() docstring.
                    val_pred = self._net(X_tensor)
                    val_loss = _masked_mse(val_pred, val_target_tensor, val_mask_tensor).item()
                val_losses.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss, best_epoch = val_loss, epoch
                    best_state = copy.deepcopy(self._net.state_dict())
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.patience:
                        break
            else:
                val_losses.append(float("nan"))

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.train_losses_ = train_losses
        self.val_losses_ = val_losses

        # Cache the full (users x items) reconstruction once, like SVD caches
        # its factors — score_all_items() then becomes a plain lookup instead
        # of re-running a forward pass on every call.
        self._net.eval()
        with torch.no_grad():
            full_reconstruction = self._net(X_tensor).numpy().T  # -> (n_users, n_items)
        self._reconstruction = pd.DataFrame(
            full_reconstruction, index=self._user_ids, columns=self._movie_ids
        )
        return self

    @property
    def supported_items(self):
        """movieIds with >=1 TRAIN rating — everything else reconstructs
        from bias terms alone (see the cold-item degeneracy check in Step 9)."""
        if self._item_support is None:
            return None
        return self._movie_ids[self._item_support > 0]

    @property
    def item_ids(self):
        """Every movieId this model was fit on, cold items included."""
        return self._movie_ids

    def seen_items(self, user_id):
        """movieIds this user rated in the matrix passed to fit() — same
        convention as SVDRecommender.seen_items() / ContentBasedRecommender.seen_items()."""
        user_idx = np.where(self._user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([], dtype=self._movie_ids.dtype)
        return self._movie_ids[self._seen_mask[user_idx[0]]]

    def score_all_items(self, user_id, clip=True):
        """
        Reconstructed rating for every movie, for one user — a lookup into
        the cached reconstruction, same shape/contract as
        SVDRecommender.score_all_items().
        clip=True -> bounded to 0.5-5.0, for RMSE/MAE.
        clip=False -> raw reconstruction, for ranking (matches SVD's reasoning:
                      clipping flattens the top of the distribution into ties).
        """
        if user_id not in self._reconstruction.index:
            return None
        # .copy() so a caller can never mutate the cached reconstruction
        # through the Series it gets back.
        scores = self._reconstruction.loc[user_id].copy()
        if clip:
            scores = scores.clip(0.5, 5.0)
        return scores

    def predict(self, user_id, movie_id):
        scores = self.score_all_items(user_id)
        if scores is None or movie_id not in scores.index:
            return None
        return scores.loc[movie_id]

    def recommend_top_n(self, user_id, n=10, exclude_seen=True):
        scores = self.score_all_items(user_id, clip=False)
        if scores is None:
            return pd.Series(dtype=float)

        if exclude_seen:
            seen = self.seen_items(user_id)
            scores = scores[~scores.index.isin(seen)]

        return scores.sort_values(ascending=False, kind="mergesort").head(n)
