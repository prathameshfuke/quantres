"""
Module 2 — Baseline Model Suite
================================
Implements a complete suite of baseline models for FI-2010 LOB mid-price
direction prediction, sharing a common BaseModel interface.

Models
------
  MomentumBaseline      : Last-observed mid-price direction heuristic.
  OLSImbalanceModel     : Logistic regression on OBI + spread features.
  XGBoostLOB            : Gradient-boosted trees with early stopping.
  RandomForestLOB       : sklearn RandomForest (200 trees, balanced).
  DeepLOB               : CNN-Inception + LSTM (Zhang et al., 2019).

All deep models use CrossEntropyLoss + Adam, batch_size=64, lr=1e-4.
Early stopping (patience=5) on validation macro-F1.

Usage:
  python -m src.module2_baselines
"""

import os
import abc
import csv
import time
from tqdm import tqdm
import pathlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — XGBoostLOB will raise if called.")

from src.module1_data_pipeline import LOBDataset

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR = pathlib.Path("images")
CHECKPOINTS_DIR = pathlib.Path("checkpoints")
LOGS_DIR = pathlib.Path("logs")

FEATURE_COLS = [
    "MidPrice", "Spread",
    "OBI_1", "OBI_3", "OBI_5",
    "WeightedMidPrice",
    "TradeIntensity_10", "TradeIntensity_50", "TradeIntensity_100",
    "QueueDepthRatio",
]

# Indices into FEATURE_COLS for the OLS feature subset
OLS_FEATURE_IDX = [1, 2, 3, 4, 5]   # Spread, OBI_1, OBI_3, OBI_5, WeightedMidPrice

RAW_LOB_DIM = 40    # Full 10-level raw LOB columns used by DeepLOB
N_CLASSES   = 3


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _empty_eval_dict() -> Dict[str, Any]:
    return {
        "F1_macro": None,
        "Precision_macro": None,
        "Recall_macro": None,
        "Accuracy": None,
        "execution_weighted_accuracy": None,   # filled in Module 3
    }


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes macro F1, precision, recall, and accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)

    Returns
    -------
    Dict with keys: F1_macro, Precision_macro, Recall_macro, Accuracy,
                    execution_weighted_accuracy (placeholder = None).
    """
    d = _empty_eval_dict()
    d["F1_macro"]         = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    d["Precision_macro"]  = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    d["Recall_macro"]     = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    d["Accuracy"]         = float(accuracy_score(y_true, y_pred))
    return d


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """
    Abstract interface shared by all LOB baseline models.

    All concrete models must implement:
      fit(train_ds, val_ds, **kwargs) — trains/calibrates the model.
      predict(dataset)               — returns np.ndarray of class predictions.
      evaluate(dataset)              — returns classification metric dict.
    """

    def __init__(self, name: str = "BaseModel") -> None:
        self.name = name

    @abc.abstractmethod
    def fit(self, train_ds: LOBDataset, val_ds: LOBDataset, **kwargs) -> None:
        """Train or calibrate the model on train_ds, using val_ds for tuning."""
        ...

    @abc.abstractmethod
    def predict(self, dataset: LOBDataset) -> np.ndarray:
        """
        Returns integer class predictions (0/1/2) for every sample in dataset.
        """
        ...

    def evaluate(self, dataset: LOBDataset) -> Dict[str, Any]:
        """
        Evaluates the model on *dataset*.

        Handles both:
        - Flat models (OLS, RF, XGB) that return one prediction per row
          in dataset.features  → length = N
        - Sequence models (MomentumBaseline, DeepLOB) that return one
          prediction per window → length = N - seq_len + 1

        Returns
        -------
        Dict with keys:
          F1_macro, Precision_macro, Recall_macro, Accuracy,
          execution_weighted_accuracy (None — filled by Module 3).
        """
        y_pred = self.predict(dataset)
        all_labels = dataset.labels.numpy()
        # Align labels to predictions by taking the tail
        if len(y_pred) < len(all_labels):
            y_true = all_labels[len(all_labels) - len(y_pred):]
        else:
            y_true = all_labels
        min_n = min(len(y_true), len(y_pred))
        return compute_classification_metrics(y_true[:min_n], y_pred[:min_n])


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

class MomentumBaseline(BaseModel):
    """
    Predicts the direction of the *last observed* mid-price change.

    If the latest mid-price is higher than the previous -> predict UP (2).
    If lower -> predict DOWN (0).
    If equal -> predict STATIONARY (1).

    No learnable parameters.
    """

    def __init__(self) -> None:
        super().__init__(name="MomentumBaseline")

    def fit(self, train_ds: LOBDataset, val_ds: LOBDataset, **kwargs) -> None:
        """No fitting required."""
        pass

    def predict(self, dataset: LOBDataset) -> np.ndarray:
        """
        Derives predictions from the MidPrice column (index 0) in the
        feature tensor by comparing consecutive values at the *last* step
        of each sequence window.

        Parameters
        ----------
        dataset : LOBDataset

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        n = len(dataset)
        preds = np.ones(n, dtype=np.int64)  # default STATIONARY

        for i in range(n):
            feat_seq, _, _ = dataset[i]
            # feat_seq: (seq_len, F) — compare last two mid-price values
            if feat_seq.shape[0] >= 2:
                diff = feat_seq[-1, 0].item() - feat_seq[-2, 0].item()
            else:
                diff = 0.0
            if diff > 0:
                preds[i] = 2
            elif diff < 0:
                preds[i] = 0
        return preds


class OLSImbalanceModel(BaseModel):
    """
    Logistic regression trained on a 5-feature subset:
    [Spread, OBI_1, OBI_3, OBI_5, WeightedMidPrice].

    Uses sklearn LogisticRegression (solver='lbfgs', max_iter=1000).
    Features are z-score normalised before fitting.
    """

    def __init__(self) -> None:
        super().__init__(name="OLSImbalanceModel")
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000, class_weight="balanced", random_state=42
        )

    def _get_X(self, dataset: LOBDataset) -> np.ndarray:
        """Extracts the OLS feature subset from dataset.features."""
        return dataset.features[:, OLS_FEATURE_IDX].numpy()

    def fit(self, train_ds: LOBDataset, val_ds: LOBDataset, **kwargs) -> None:
        X_train = self.scaler.fit_transform(self._get_X(train_ds))
        y_train = train_ds.labels.numpy()
        self.clf.fit(X_train, y_train)
        # Evaluate on val for logging
        X_val = self.scaler.transform(self._get_X(val_ds))
        val_f1 = f1_score(val_ds.labels.numpy(), self.clf.predict(X_val),
                          average="macro", zero_division=0)
        print(f"[OLSImbalanceModel] Val macro-F1 = {val_f1:.4f}")

    def predict(self, dataset: LOBDataset) -> np.ndarray:
        X = self.scaler.transform(self._get_X(dataset))
        return self.clf.predict(X).astype(np.int64)


# ---------------------------------------------------------------------------
# Tree-based baselines
# ---------------------------------------------------------------------------

class RandomForestLOB(BaseModel):
    """
    scikit-learn RandomForestClassifier with 200 trees trained on the full
    10-feature matrix.  Uses class_weight='balanced' to handle label imbalance.
    """

    def __init__(self, n_estimators: int = 200) -> None:
        super().__init__(name="RandomForestLOB")
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, train_ds: LOBDataset, val_ds: LOBDataset, **kwargs) -> None:
        X_train = train_ds.features.numpy()
        y_train = train_ds.labels.numpy()
        print(f"[RandomForestLOB] Fitting on {X_train.shape[0]:,} samples …")
        self.clf.fit(X_train, y_train)
        X_val = val_ds.features.numpy()
        val_f1 = f1_score(val_ds.labels.numpy(), self.clf.predict(X_val),
                          average="macro", zero_division=0)
        print(f"[RandomForestLOB] Val macro-F1 = {val_f1:.4f}")
        # Feature importances
        imps = self.clf.feature_importances_
        for name, imp in sorted(zip(FEATURE_COLS, imps), key=lambda x: -x[1]):
            print(f"   {name:<25s} {imp:.4f}")

    def predict(self, dataset: LOBDataset) -> np.ndarray:
        return self.clf.predict(dataset.features.numpy()).astype(np.int64)


class XGBoostLOB(BaseModel):
    """
    XGBoost gradient-boosted classifier trained on the full feature matrix
    with early stopping on the validation set.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
    ) -> None:
        super().__init__(name="XGBoostLOB")
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            device=_device,
            early_stopping_rounds=20,
            random_state=42,
            verbosity=0,
        )
        self.clf: Optional[xgb.XGBClassifier] = None

    def fit(self, train_ds: LOBDataset, val_ds: LOBDataset, **kwargs) -> None:
        X_train = train_ds.features.numpy()
        y_train = train_ds.labels.numpy()
        X_val   = val_ds.features.numpy()
        y_val   = val_ds.labels.numpy()

        # class_weight via sample_weight
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y_train)

        self.clf = xgb.XGBClassifier(**self.params)
        self.clf.fit(
            X_train, y_train,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = self.clf.best_iteration
        val_f1 = f1_score(y_val, self.clf.predict(X_val),
                          average="macro", zero_division=0)
        print(f"[XGBoostLOB] Best iteration = {best_iter}  |  Val macro-F1 = {val_f1:.4f}")

        # Feature importances
        imps = self.clf.feature_importances_
        for name, imp in sorted(zip(FEATURE_COLS, imps), key=lambda x: -x[1]):
            print(f"   {name:<25s} {imp:.4f}")

    def predict(self, dataset: LOBDataset) -> np.ndarray:
        return self.clf.predict(dataset.features.numpy()).astype(np.int64)


# ---------------------------------------------------------------------------
# DeepLOB architecture (Zhang et al., 2019)
# ---------------------------------------------------------------------------

class _InceptionBranch(nn.Module):
    """Single branch of the Inception module: Conv2d + BN + LeakyReLU."""

    def __init__(self, kernel_h: int, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(kernel_h, 1),
            padding=(kernel_h // 2, 0),
            bias=False,
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    """
    Three-branch Inception block operating on raw LOB snapshots.

    Input  : (B, 1, T, 40)  — batch × channel × time × LOB features
    Output : (B, 96, T, 40) — three concatenated 32-channel branches
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch1 = _InceptionBranch(kernel_h=1)
        self.branch2 = _InceptionBranch(kernel_h=2)
        self.branch4 = _InceptionBranch(kernel_h=4)

        # 1×1 bottleneck to reduce channels after concat
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        # Trim height dimension to match (kernel_h=2 adds 1 row of padding)
        min_h = min(b1.size(2), b2.size(2), b4.size(2))
        b1, b2, b4 = b1[:,:,:min_h,:], b2[:,:,:min_h,:], b4[:,:,:min_h,:]
        out = torch.cat([b1, b2, b4], dim=1)       # (B, 96, T, 40)
        return self.bottleneck(out)                 # (B, 64, T, 40)


class DeepLOBNet(nn.Module):
    """
    DeepLOB network: Inception CNN + 2-layer LSTM + FC head.

    Follows the architecture in Zhang et al. (2019):
    "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books."

    Parameters
    ----------
    seq_len : int
        Length of the input time window (default 10).
    lob_dim : int
        Raw LOB feature dimension (default 40).
    lstm_hidden : int
        Hidden units in the LSTM (default 64).
    n_classes : int
        Number of output classes (default 3).
    """

    def __init__(
        self,
        seq_len: int = 10,
        lob_dim: int = 40,
        lstm_hidden: int = 64,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len  = seq_len
        self.lob_dim  = lob_dim

        self.inception = InceptionModule()

        # After inception: (B, 64, ≤T, lob_dim)
        # Pool across the lob_dim axis to create a temporal sequence
        self.pool = nn.AvgPool2d(kernel_size=(1, lob_dim))

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, T, lob_dim)   raw LOB sequence

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        # Reshape for Conv2d: (B, 1, T, lob_dim)
        out = x.unsqueeze(1)                               # (B, 1, T, F)
        out = self.inception(out)                          # (B, 64, T', F)
        out = self.pool(out)                               # (B, 64, T', 1)
        out = out.squeeze(-1).permute(0, 2, 1)            # (B, T', 64)
        _, (hn, _) = self.lstm(out)                        # hn: (2, B, H)
        out = hn[-1]                                       # last layer (B, H)
        return self.fc(out)                                # (B, n_classes)


class DeepLOBModel(BaseModel):
    """
    Wrapper around DeepLOBNet implementing the BaseModel interface.

    The model expects LOBDataset samples that expose raw 40-dim LOB
    data as features (or snapshots).  When the dataset provides the
    10-feature engineered matrix we reconstruct it from the snapshot.

    NOTE: DeepLOB is designed to take the *raw* 40-column LOB state.
    We build a minimal raw-feature dataset from the snapshots (20-dim
    bid/ask volumes) expanded to 40 dimensions for compatibility.
    """

    def __init__(
        self,
        seq_len: int = 10,
        lob_dim: int = 40,
        lstm_hidden: int = 64,
        lr: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 5,
        device: Optional[str] = None,
        checkpoint_path: str = "checkpoints/deeplob_ce.pt",
    ) -> None:
        super().__init__(name="DeepLOB_CrossEntropy")
        self.seq_len         = seq_len
        self.lob_dim         = lob_dim
        self.lr              = lr
        self.batch_size      = batch_size
        self.max_epochs      = max_epochs
        self.patience        = patience
        self.checkpoint_path = checkpoint_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[DeepLOBModel] Using device: {self.device}"
              + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))

        self.net = DeepLOBNet(
            seq_len=seq_len, lob_dim=lob_dim,
            lstm_hidden=lstm_hidden, n_classes=N_CLASSES
        ).to(self.device)

        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _build_raw_sequences(
        self, dataset: LOBDataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds (X, y) tensors where X has shape (N, seq_len, lob_dim).

        DeepLOB expects the raw 40-feature LOB input.  Since we store both
        the 10-dim engineered features AND the 20-dim volume snapshot, we
        reconstruct a 20-dim input and zero-pad to lob_dim.

        In practice, callers should supply datasets whose features tensor
        already contains 40 raw LOB columns.  We handle both cases.
        """
        n = len(dataset)
        F = dataset.features.shape[1]
        seq = self.seq_len

        X_list, y_list = [], []
        for i in range(n):
            feat_seq, label, _ = dataset[i]          # (T, F), int, (20,)
            if F >= self.lob_dim:
                x = feat_seq[:, :self.lob_dim]
            else:
                # Pad with zeros to reach lob_dim
                pad = torch.zeros(seq, self.lob_dim - F)
                x   = torch.cat([feat_seq, pad], dim=1)
            X_list.append(x)
            y_list.append(label)

        X = torch.stack(X_list)                       # (N, T, lob_dim)
        y = torch.tensor(y_list, dtype=torch.long)    # (N,)
        return X, y

    def _make_loader(self, dataset: LOBDataset, shuffle: bool) -> DataLoader:
        X, y = self._build_raw_sequences(dataset)
        td = TensorDataset(X, y)
        return DataLoader(td, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=self.device.type == "cuda")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_ds: LOBDataset,
        val_ds: LOBDataset,
        loss_fn: Optional[nn.Module] = None,
        log_csv: str = "logs/deeplob_ce_training.csv",
        **kwargs,
    ) -> None:
        """
        Trains DeepLOB with the supplied loss function (defaults to
        CrossEntropyLoss).

        Parameters
        ----------
        train_ds, val_ds : LOBDataset
        loss_fn : nn.Module, optional
            Loss callable.  Signature: loss_fn(logits, labels) -> scalar.
            Pass a callable supporting (logits, labels, snapshots) for L_EXEC.
        log_csv : str
            Path to the per-epoch training log CSV.
        """
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

        criterion  = loss_fn or nn.CrossEntropyLoss()
        optimizer  = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)

        train_loader = self._make_loader(train_ds, shuffle=True)
        val_loader   = self._make_loader(val_ds,   shuffle=False)

        best_val_f1   = -1.0
        patience_ctr  = 0
        self._history = []

        csv_fields = ["epoch", "train_loss", "val_f1"]
        with open(log_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()

            epoch_bar = tqdm(
                range(1, self.max_epochs + 1),
                desc=f"[{self.name}] Training",
                unit="ep",
                dynamic_ncols=True,
            )
            for epoch in epoch_bar:
                # ── train ────────────────────────────────────────────────
                self.net.train()
                total_loss = 0.0
                n_batches  = 0
                batch_bar  = tqdm(train_loader, desc="  batches", leave=False,
                                  unit="b", dynamic_ncols=True)
                for xb, yb in batch_bar:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.net(xb)

                    # Support extended loss signature (logits, labels, snapshots)
                    if hasattr(criterion, "_needs_snapshots") and criterion._needs_snapshots:
                        # placeholder snapshot — overridden in Module 5
                        dummy_snap = torch.zeros(xb.size(0), 20, device=self.device)
                        loss = criterion(logits, yb, dummy_snap)
                    else:
                        loss = criterion(logits, yb)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches  += 1
                    batch_bar.set_postfix(loss=f"{loss.item():.4f}")

                avg_train_loss = total_loss / max(n_batches, 1)

                # ── validate ─────────────────────────────────────────────
                val_f1 = self._eval_f1(val_loader)
                scheduler.step(val_f1)

                row = {"epoch": epoch, "train_loss": avg_train_loss, "val_f1": val_f1}
                self._history.append(row)
                writer.writerow(row)

                is_best = val_f1 > best_val_f1
                epoch_bar.set_postfix(
                    loss=f"{avg_train_loss:.4f}",
                    val_F1=f"{val_f1:.4f}",
                    best=f"{best_val_f1:.4f}",
                    patience=f"{patience_ctr}/{self.patience}",
                )

                if is_best:
                    best_val_f1 = val_f1
                    patience_ctr = 0
                    torch.save(self.net.state_dict(), self.checkpoint_path)
                    tqdm.write(f"  Epoch {epoch:3d}/{self.max_epochs} | loss={avg_train_loss:.4f} | val_F1={val_f1:.4f}  ✓ best")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        tqdm.write(f"  Early stopping at epoch {epoch} (patience={self.patience}).")
                        break

        # Restore best weights
        self.net.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        print(f"[DeepLOB] Training complete. Best val macro-F1 = {best_val_f1:.4f}")

    # ------------------------------------------------------------------

    def _eval_f1(self, loader: DataLoader) -> float:
        self.net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self.net(xb)
                preds  = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.numpy().tolist())
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def predict(self, dataset: LOBDataset) -> np.ndarray:
        loader = self._make_loader(dataset, shuffle=False)
        self.net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                preds.append(self.net(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def predict_proba(self, dataset: LOBDataset) -> np.ndarray:
        """Returns softmax probabilities, shape (N, 3)."""
        loader = self._make_loader(dataset, shuffle=False)
        self.net.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                p = torch.softmax(self.net(xb), dim=1)
                probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_model(
    model: BaseModel,
    train_ds: LOBDataset,
    val_ds: LOBDataset,
    n_epochs: int = 50,
    loss_fn: Optional[nn.Module] = None,
    log_csv: str = "logs/training_log.csv",
    curve_png: str = "images/training_curves.png",
) -> None:
    """
    Generic training loop that dispatches to the model's fit() method.

    For deep-learning models the loss function and epoch count are forwarded.
    For classical models fit() is called once (epochs / loss ignored).

    Training loss and validation F1 are logged to *log_csv* after each epoch
    for DeepLOB; a single-row entry is written for classical models.

    A training-curve PNG is saved to *curve_png* if the model has history.

    Parameters
    ----------
    model     : BaseModel subclass instance
    train_ds  : LOBDataset
    val_ds    : LOBDataset
    n_epochs  : int
    loss_fn   : nn.Module, optional
    log_csv   : str
    curve_png : str
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Training: {model.name}")
    print(f"{'='*60}")

    if isinstance(model, DeepLOBModel):
        model.fit(
            train_ds, val_ds,
            loss_fn=loss_fn,
            log_csv=log_csv,
            max_epochs=n_epochs,
        )
        # Plot training curves
        if model._history:
            _plot_training_curves(model._history, curve_png, model.name)
    else:
        # Classical / tree-based models
        model.fit(train_ds, val_ds)
        # Write a single evaluation entry to the log
        val_metrics = model.evaluate(val_ds)
        with open(log_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["epoch", "train_loss", "val_f1"]
            )
            writer.writeheader()
            writer.writerow({
                "epoch": 1,
                "train_loss": float("nan"),
                "val_f1": val_metrics["F1_macro"],
            })

    train_metrics = model.evaluate(train_ds)
    val_metrics   = model.evaluate(val_ds)
    print(f"\n[{model.name}] Train F1={train_metrics['F1_macro']:.4f}  "
          f"Val F1={val_metrics['F1_macro']:.4f}")


def _plot_training_curves(
    history: List[Dict], out_path: str, model_name: str
) -> None:
    """Plots training loss and validation F1 over epochs and saves PNG."""
    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_f1      = [h["val_f1"]     for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "b-o", markersize=3)
    ax1.set_title(f"{model_name}: Training Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, val_f1, "g-o", markersize=3)
    ax2.set_title(f"{model_name}: Validation Macro-F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Module 2] Training curves saved -> {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.module1_data_pipeline import FI2010DataLoader, download_fi2010, DATA_DIR

    paths = download_fi2010(DATA_DIR)
    loader = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
    train_ds, val_ds, _ = loader.load_and_split(paths["Train"], val_fraction=0.2)
    test_ds = loader.load_test([paths["Test1"], paths["Test2"], paths["Test3"]])

    models = [
        MomentumBaseline(),
        OLSImbalanceModel(),
        RandomForestLOB(n_estimators=200),
    ]
    if _XGB_AVAILABLE:
        models.append(XGBoostLOB())

    results = []
    for m in models:
        train_model(m, train_ds, val_ds,
                    log_csv=f"logs/{m.name}_log.csv",
                    curve_png=f"images/{m.name}_curves.png")
        metrics = m.evaluate(test_ds)
        metrics["model"] = m.name
        results.append(metrics)
        print(f"  {m.name}: test F1={metrics['F1_macro']:.4f}")

    # DeepLOB is trained separately (expensive)
    deeplob = DeepLOBModel(seq_len=10, max_epochs=50, patience=5)
    train_model(deeplob, train_ds, val_ds,
                log_csv="logs/deeplob_ce_log.csv",
                curve_png="images/deeplob_ce_curves.png")
    metrics = deeplob.evaluate(test_ds)
    metrics["model"] = deeplob.name
    results.append(metrics)

    df = pd.DataFrame(results).set_index("model")
    print("\n=== Baseline Results ===")
    print(df[["F1_macro", "Precision_macro", "Recall_macro", "Accuracy"]].to_string())
    df.to_csv("logs/baseline_results.csv")


# ---------------------------------------------------------------------------
# Assumptions and edge-case notes
# ---------------------------------------------------------------------------
# ASSUMPTIONS:
#   1. DeepLOB receives the 40-column raw LOB features tensor.  When only a
#      10-feature engineered matrix is available the model zero-pads to reach
#      40 dimensions — this is a degraded input but allows architectural
#      compatibility for ablation purposes.
#   2. Early stopping monitors validation macro-F1, NOT validation loss.
#      Macro-F1 is more meaningful than cross-entropy for imbalanced LOB data.
#   3. RandomForest and XGBoost are given the engineered 10-feature matrix,
#      NOT the raw 40-column LOB.  DeepLOB is given the raw LOB following the
#      original paper.
#
# EDGE CASES FOR MANUAL REVIEW:
#   • STATIONARY dominance: if the model collapses to always predicting
#     STATIONARY (the majority class), macro-F1 will be ~0.33 even with
#     perfect STATIONARY recall.  Monitor per-class recall in training logs.
#   • XGBoost early stopping requires at least one evaluation round to have
#     elapsed; if the validation set is very small (< batch_size) the first
#     round may throw a warning.
#   • DeepLOB's LSTM dropout is disabled during inference (model.eval() is
#     called in predict()) — confirm this is set correctly if you add MC-
#     dropout for uncertainty estimation later.
#
# REAL LOB vs FI-2010 DIFFERENCES:
#   • The inception kernel sizes (1, 2, 4) in Zhang et al. (2019) are
#     designed for a 10-level × 4-column LOB layout.  On deeper LOBs (e.g.
#     20-level) the architectural hyper-params would need revisiting.
#   • In live trading, the model receives a continuous stream of partial
#     LOB updates; FI-2010 provides complete snapshots after each update.
#     The seq_len window assumption implicitly treats each snapshot as equally
#     informative, which may not hold for very bursty order flow.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compatibility factory: DeepLOB(num_classes=3)
# ---------------------------------------------------------------------------

def DeepLOB(
    num_classes: int = 3,
    seq_len: int = 100,
    lob_dim: int = 40,
    lstm_hidden: int = 64,
) -> DeepLOBNet:
    """
    Factory function creating a DeepLOBNet with keyword names that match
    the training-loop spec (num_classes instead of n_classes, seq_len=100
    by default to match the 100-step window from create_dataloaders).
    """
    return DeepLOBNet(
        seq_len=seq_len,
        lob_dim=lob_dim,
        lstm_hidden=lstm_hidden,
        n_classes=num_classes,
    )

