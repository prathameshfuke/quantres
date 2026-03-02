"""
Module 1 â€” Data Pipeline and LOB Feature Engineering
=====================================================
Implements a full, reproducible data pipeline for the FI-2010 benchmark
Limit Order Book dataset (Ntakaris et al., 2018).

FI-2010 Source:
  Aalto University / Finnish Meteorological Institute.
  Public mirror via the BenchmarkDatasets GitHub repo used by DeepLOB
  (Zhang et al., 2019):
  https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

Column schema (per level i = 1..10, interleaved):
  AskPrice_i, AskVol_i, BidPrice_i, BidVol_i
Total first-40 raw LOB columns per snapshot.

Label encoding following Zhang et al. (2019) smooth labeling:
  0 = DOWN, 1 = STATIONARY, 2 = UP

Usage:
  python -m src.module1_data_pipeline
"""

import os
import io
import zipfile
import pathlib
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")          # non-interactive backend safe for servers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FI-2010 files hosted on the DeepLOB benchmark GitHub mirror.
# The repository stores five stocks Ã— two auction-types Ã— normalization.
# We download the "No-Auction" dÃ©cimal-price normalised files,
# which are the standard files used by every paper citing FI-2010.
_FI2010_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
    "master/data/data.zip"
)

# The five canonical FI-2010 training / test file-stems inside data.zip.
_FI2010_FILES = {
    "Train": "Train_Dst_NoAuction_DecPre_CF_7.txt",
    "Test1": "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "Test2": "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "Test3": "Test_Dst_NoAuction_DecPre_CF_9.txt",
}

# Expected number of rows in the FI-2010 transposed format (features axis)
_FI2010_NUM_FEATURES = 144

# Label rows inside the file correspond to these forward-horizon values
_FI2010_K_VALUES = [1, 2, 3, 5, 10]

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def get_fi2010_columns() -> List[str]:
    """
    Returns the 40-column standardised LOB schema for the raw price/volume
    fields (levels 1 â€“ 10, interleaved Ask/Bid).

    Returns
    -------
    List[str]
        Column names in the order they appear in the FI-2010 files.
    """
    cols: List[str] = []
    for i in range(1, 11):
        cols.extend([f"AskPrice_{i}", f"AskVol_{i}", f"BidPrice_{i}", f"BidVol_{i}"])
    return cols


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_fi2010(data_dir: pathlib.Path = DATA_DIR, force: bool = False) -> Dict[str, pathlib.Path]:
    """
    Downloads the FI-2010 NoAuction DecimalPrecision data files from the
    public DeepLOB GitHub mirror.  The repo ships a single ``data.zip`` which
    is downloaded once, extracted, and cached locally.

    Parameters
    ----------
    data_dir : pathlib.Path
        Local directory to place the extracted files.
    force : bool
        Re-download and re-extract even if cached.

    Returns
    -------
    Dict[str, pathlib.Path]
        Mapping of split name â†’ local file path.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if all target files already exist
    local_paths: Dict[str, pathlib.Path] = {}
    all_cached = all((data_dir / fname).exists() for fname in _FI2010_FILES.values())

    if all_cached and not force:
        print("[download] All FI-2010 files already cached.")
        return {split: data_dir / fname for split, fname in _FI2010_FILES.items()}

    # Download zip
    zip_dest = data_dir / "data.zip"
    if not zip_dest.exists() or force:
        print(f"[download] Fetching {_FI2010_ZIP_URL} â€¦")
        try:
            urllib.request.urlretrieve(_FI2010_ZIP_URL, zip_dest)
            print(f"[download] Saved zip â†’ {zip_dest}  ({zip_dest.stat().st_size // 1024} KB)")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to download FI-2010 zip from {_FI2010_ZIP_URL}.\n"
                "Check network connectivity. Error: " + str(exc)
            ) from exc

    # Extract zip
    print("[download] Extracting data.zip ...")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        namelist = zf.namelist()
        print(f"[download] Zip contains {len(namelist)} entries.")
        # Extract everything flat into data_dir, stripping any path prefix
        for member in tqdm(namelist, desc="  Extracting", unit="file", dynamic_ncols=True):
            basename = pathlib.Path(member).name
            if not basename:
                continue
            dest = data_dir / basename
            if dest.exists() and not force:
                continue
            data = zf.read(member)
            dest.write_bytes(data)
            tqdm.write(f"[download] Extracted -> {dest}")

    # Verify target files are now present
    for split_name, fname in _FI2010_FILES.items():
        dest = data_dir / fname
        if not dest.exists():
            raise RuntimeError(
                f"Expected file '{fname}' not found in zip after extraction.\n"
                f"Zip contents: {[pathlib.Path(n).name for n in namelist]}"
            )
        local_paths[split_name] = dest
        print(f"[download] {split_name}: ready at {dest}")

    return local_paths


# ---------------------------------------------------------------------------
# PyTorch Dataset  (new API — window-based, used by create_dataloaders)
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    PyTorch Dataset for DeepLOB-style training.

    Each sample is a (window_size, 40) window of normalised LOB features,
    labelled at the step immediately following the window.

    Parameters
    ----------
    X          : (N, 40) float32 numpy — StandardScaler-normalised features.
    y          : (N,)    int64   numpy — labels {0=DOWN, 1=STAT, 2=UP}.
    window_size: int — number of consecutive snapshots per sample (default 100).
    raw_X      : (N, 40) float32 numpy — un-normalised features used for the
                 execution-loss snap tensor.  If None, snap returns zeros(40).
    """

    def __init__(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        window_size: int = 100,
        raw_X:       Optional[np.ndarray] = None,
    ) -> None:
        N = len(X)
        assert len(y) == N, "X and y must have the same length."
        if raw_X is not None:
            assert len(raw_X) == N, "raw_X must have the same length as X."

        self.window_size = window_size

        # Convert once at construction time — critical for __getitem__ speed
        self.X     = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y     = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.raw_X = (
            torch.from_numpy(np.ascontiguousarray(raw_X)).float()
            if raw_X is not None else None
        )

    def __len__(self) -> int:
        # Valid label positions: window_size … N-1  → total = N - window_size
        return max(0, len(self.X) - self.window_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        sequence : float32 tensor, shape (1, window_size, 40)
            One-channel CNN input: X[idx : idx+window_size] reshaped.
        label    : int64 scalar tensor
            y[idx + window_size]
        snap     : float32 tensor, shape (40,)
            raw_X[idx + window_size] if raw_X is available, else zeros(40).
        """
        end       = idx + self.window_size
        sequence  = self.X[idx:end].reshape(1, self.window_size, 40)   # (1, W, 40)
        label     = self.y[end]                                          # scalar
        snap      = self.raw_X[end] if self.raw_X is not None \
                    else torch.zeros(40, dtype=torch.float32)
        return sequence, label, snap


# ---------------------------------------------------------------------------
# _LOBDatasetLegacy  (used internally by FI2010DataLoader — do not rename)
# ---------------------------------------------------------------------------

class _LOBDatasetLegacy(Dataset):
    """
    Legacy dataset used by FI2010DataLoader's class-based pipeline.
    Accepts pre-computed engineered feature tensors and 20-column snap tensors.
    """

    def __init__(
        self,
        features:  torch.Tensor,
        labels:    torch.Tensor,
        snapshots: torch.Tensor,
        seq_len:   int = 10,
    ) -> None:
        assert features.shape[0] == labels.shape[0] == snapshots.shape[0], (
            "features, labels, snapshots must share the same first dimension."
        )
        self.features  = features
        self.labels    = labels
        self.snapshots = snapshots
        self.seq_len   = seq_len

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        feat_seq   = self.features[idx : idx + self.seq_len]
        target_idx = idx + self.seq_len - 1
        label      = int(self.labels[target_idx].item())
        snapshot   = self.snapshots[target_idx]
        return feat_seq, label, snapshot


# ---------------------------------------------------------------------------
# DataLoader factory  (new API)
# ---------------------------------------------------------------------------

def create_dataloaders(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    raw_X_train: Optional[np.ndarray] = None,
    raw_X_test:  Optional[np.ndarray] = None,
    window_size: int = 100,
    batch_size:  int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, float]:
    """
    Build train and test DataLoaders from normalised numpy arrays produced
    by load_fi2010_dataset(), then runs a mandatory shape assertion block.

    Parameters
    ----------
    X_train, y_train : training split (StandardScaler-normalised).
    X_test,  y_test  : test split.
    raw_X_train      : un-normalised X_train for snap tensors / spread_mean.
                       If None, snap tensors are zeros and spread_mean is
                       computed from normalised prices (still positive in
                       FI-2010 decimal-precision data).
    raw_X_test       : un-normalised X_test for snap tensors.
    window_size      : look-back window per sample (default 100).
    batch_size       : mini-batch size (default 64).
    num_workers      : DataLoader worker processes (default 0 — safe on Windows).

    Returns
    -------
    train_loader : DataLoader  (shuffle=True,  drop_last=True)
    test_loader  : DataLoader  (shuffle=False, drop_last=False)
    spread_mean  : float > 0   (mean bid-ask spread from raw_X_train col 0, 2)
    """
    pin = torch.cuda.is_available()

    train_ds = LOBDataset(X_train, y_train, window_size=window_size, raw_X=raw_X_train)
    test_ds  = LOBDataset(X_test,  y_test,  window_size=window_size, raw_X=raw_X_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # ------------------------------------------------------------------
    # Mandatory shape assertion block
    # ------------------------------------------------------------------
    seq_batch, lbl_batch, snap_batch = next(iter(train_loader))

    assert seq_batch.shape == (batch_size, 1, window_size, 40), \
        f"SEQ SHAPE FAIL: got {seq_batch.shape}"
    assert lbl_batch.shape == (batch_size,), \
        f"LABEL SHAPE FAIL: got {lbl_batch.shape}"
    assert snap_batch.shape == (batch_size, 40), \
        f"SNAP SHAPE FAIL: got {snap_batch.shape}"
    assert lbl_batch.dtype == torch.int64, \
        f"LABEL DTYPE FAIL: got {lbl_batch.dtype}"
    assert not torch.isnan(seq_batch).any(), "NaN in sequences"
    assert not torch.isinf(seq_batch).any(), "Inf in sequences"

    unique_lbls = lbl_batch.unique().tolist()
    assert len(unique_lbls) > 1, \
        f"COLLAPSE: all labels in first batch are class {unique_lbls}"

    counts = torch.bincount(lbl_batch, minlength=3)
    print(f"\n{'='*55}")
    print(f"  DATALOADER VERIFICATION")
    print(f"{'='*55}")
    print(f"  seq_batch  : {tuple(seq_batch.shape)}  dtype={seq_batch.dtype}")
    print(f"  lbl_batch  : {tuple(lbl_batch.shape)}  dtype={lbl_batch.dtype}")
    print(f"  snap_batch : {tuple(snap_batch.shape)} dtype={snap_batch.dtype}")
    print(
        f"  Label counts in first batch — "
        f"DOWN={counts[0].item()}  STAT={counts[1].item()}  UP={counts[2].item()}"
    )
    print(f"  NaN check  : PASS")
    print(f"  Inf check  : PASS")
    print(f"  Train batches total : {len(train_loader)}")
    print(f"  Test  batches total : {len(test_loader)}")
    print(f"{'='*55}")
    print(f"  \u2713 DATASET AND DATALOADER OK")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Compute and cache spread_mean for loss function calibration
    # source: raw_X_train if provided, else normalised X_train
    # ------------------------------------------------------------------
    src = raw_X_train if raw_X_train is not None else X_train
    best_ask = src[:, 0]   # AskPrice_1
    best_bid = src[:, 2]   # BidPrice_1
    spread_mean = float(np.mean(best_ask - best_bid))

    if spread_mean <= 0:
        raise ValueError(
            f"spread_mean={spread_mean:.8f} is not positive. "
            "Pass un-normalised raw_X_train for a valid spread calibration, "
            "or check data integrity."
        )

    spread_path = DATA_DIR / "spread_mean.txt"
    spread_path.write_text(f"{spread_mean}\n", encoding="ascii")
    print(f"  Calibrated spread_mean = {spread_mean:.6f}")
    print(f"  Spread saved -> {spread_path}\n")

    return train_loader, test_loader, spread_mean


# ---------------------------------------------------------------------------
# Main data-loader / preprocessor
# ---------------------------------------------------------------------------

class FI2010DataLoader:
    """
    End-to-end data pipeline for the FI-2010 benchmark dataset.

    Handles:
      * Raw file ingestion (transposed FI-2010 format).
      * Feature engineering (spread, OBI, weighted mid-price, etc.).
      * Smooth label generation (Zhang et al., 2019).
      * Temporal train / val / test splits.
      * Summary statistics and diagnostic plots.

    Parameters
    ----------
    seq_len : int
        Look-back window length fed to sequence models (default 10).
    k : int
        Forward horizon used for smooth label generation (default 10).
    alpha : float
        Relative mid-price change threshold for UP/DOWN classification
        (default 0.002, i.e. 0.2 %).
    """

    def __init__(
        self,
        seq_len: int = 10,
        k: int = 10,
        alpha: float = 0.002,
    ) -> None:
        self.seq_len = seq_len
        self.k = k
        self.alpha = alpha
        self.lob_columns = get_fi2010_columns()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_raw_file(
        self,
        file_path: str | os.PathLike,
        is_transposed: bool = True,
    ) -> pd.DataFrame:
        """
        Reads one FI-2010 data file and returns a DataFrame with the 40
        canonical LOB columns (feature rows 0-39 only).

        The FI-2010 files are stored transposed: rows = features/labels,
        columns = time steps.  We flip the matrix so that rows = time steps.

        Parameters
        ----------
        file_path : path-like
            Path to the raw FI-2010 txt/csv file.
        is_transposed : bool
            Set True (default) for the standard FI-2010 transposed layout.

        Returns
        -------
        pd.DataFrame, shape (N, 40)
            Raw LOB price/volume columns only.
        """
        fpath = pathlib.Path(file_path)
        size_mb = fpath.stat().st_size / 1024**2
        with tqdm(total=None, desc=f"  Loading {fpath.name} ({size_mb:.0f} MB)",
                  bar_format="{desc} ... {elapsed}", dynamic_ncols=True) as pbar:
            raw = np.loadtxt(file_path)
            pbar.set_description(f"  Loaded  {fpath.name} ({size_mb:.0f} MB)")
        # raw shape: (149, N) --- rows=features+labels, cols=time steps
        if is_transposed and raw.shape[0] < raw.shape[1]:
            lob_np = raw[:40, :].T          # (N, 40) — first 40 feature rows
        else:
            lob_np = raw[:, :40]
        lob = pd.DataFrame(lob_np.astype(np.float32), columns=self.lob_columns)
        return lob.reset_index(drop=True)

    def load_raw_file_with_labels(
        self,
        file_path: str | os.PathLike,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads LOB features AND the precomputed direction labels that ship
        inside every FI-2010 file (last 5 rows, for k=1,2,3,5,10).

        The FI-2010 label encoding is {1=DOWN, 2=STATIONARY, 3=UP}.
        We remap to {0=DOWN, 1=STATIONARY, 2=UP} to match the rest of the code.

        Parameters
        ----------
        file_path : path-like

        Returns
        -------
        lob_df    : pd.DataFrame, shape (N, 40)
        labels    : np.ndarray, shape (N,), dtype int64 — labels for self.k
        """
        fpath = pathlib.Path(file_path)
        size_mb = fpath.stat().st_size / 1024**2
        with tqdm(total=None, desc=f"  Loading {fpath.name} ({size_mb:.0f} MB)",
                  bar_format="{desc} ... {elapsed}", dynamic_ncols=True) as pbar:
            raw = np.loadtxt(file_path)   # (149, N)
            pbar.set_description(f"  Loaded  {fpath.name} ({size_mb:.0f} MB)")
        if raw.shape[0] >= raw.shape[1]:
            raw = raw.T               # ensure (149, N) orientation

        lob_np = raw[:40, :].T.astype(np.float32)   # (N, 40)
        lob_df = pd.DataFrame(lob_np, columns=self.lob_columns)

        # Rows 144-148 are labels for k = 1, 2, 3, 5, 10
        k_idx = _FI2010_K_VALUES.index(self.k) if self.k in _FI2010_K_VALUES else 4
        label_row = raw[144 + k_idx, :]                        # shape (N,)
        # FI-2010 encoding: 1=UP, 2=STATIONARY, 3=DOWN
        # Re-encode to:     2=UP, 1=STATIONARY, 0=DOWN  via 3 - label
        labels = (3 - label_row.astype(np.int64))              # {1,2,3} -> {2,1,0}

        return lob_df.reset_index(drop=True), labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all engineered features from the raw 10-level LOB snapshot
        DataFrame.

        Features produced (in column order):
          MidPrice, Spread,
          OBI_1, OBI_3, OBI_5,
          WeightedMidPrice,
          TradeIntensity_10, TradeIntensity_50, TradeIntensity_100,
          QueueDepthRatio

        Parameters
        ----------
        df : pd.DataFrame
            Raw LOB DataFrame with the 40 canonical columns.

        Returns
        -------
        pd.DataFrame, shape (N, 10)
        """
        feats = pd.DataFrame(index=df.index)

        # 1. Mid-price â€” best bid/ask average
        feats["MidPrice"] = (df["AskPrice_1"] + df["BidPrice_1"]) / 2.0

        # 2. Bid-Ask Spread at best level
        feats["Spread"] = df["AskPrice_1"] - df["BidPrice_1"]

        # 3. Order Book Imbalance at levels 1, 3, 5
        #    OBI_n = (Î£ bid_vol_i  âˆ’ Î£ ask_vol_i) / (Î£ bid_vol_i + Î£ ask_vol_i)
        for n in [1, 3, 5]:
            bid_v = sum(df[f"BidVol_{i}"] for i in range(1, n + 1))
            ask_v = sum(df[f"AskVol_{i}"] for i in range(1, n + 1))
            feats[f"OBI_{n}"] = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)

        # 4. Volume-weighted mid-price across all 10 levels
        #    Î£ (B_i * b_i  +  A_i * a_i) / Î£ (b_i + a_i)
        num = sum(
            df[f"BidPrice_{i}"] * df[f"BidVol_{i}"]
            + df[f"AskPrice_{i}"] * df[f"AskVol_{i}"]
            for i in range(1, 11)
        )
        den = sum(df[f"BidVol_{i}"] + df[f"AskVol_{i}"] for i in range(1, 11))
        feats["WeightedMidPrice"] = num / (den + 1e-8)

        # 5. Trade intensity proxy â€” rolling count of mid-price change events
        #    over three window sizes.  Captures order-flow arrival rate.
        changed = (feats["MidPrice"].diff().fillna(0).abs() > 0).astype(np.float32)
        for w in [10, 50, 100]:
            feats[f"TradeIntensity_{w}"] = (
                changed.rolling(window=w, min_periods=1).sum()
            )

        # 6. Queue depth ratio â€” total bid depth vs total ask depth
        total_bid = sum(df[f"BidVol_{i}"] for i in range(1, 11))
        total_ask = sum(df[f"AskVol_{i}"] for i in range(1, 11))
        feats["QueueDepthRatio"] = total_bid / (total_ask + 1e-8)

        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def generate_labels(self, mid_prices: pd.Series) -> np.ndarray:
        """
        Smooth labeling method from Zhang et al. (2019) / DeepLOB.

        For each time step t, compute the mean of the NEXT k mid-prices,
        then classify the relative change w.r.t. the current mid-price:

          l_t = (mean_{t+1..t+k}  âˆ’  mid_t) / mid_t
          label = 2 (UP)         if l_t  >  alpha
                = 0 (DOWN)       if l_t  < -alpha
                = 1 (STATIONARY) otherwise

        The last k labels are set to STATIONARY (invalid look-ahead).

        Parameters
        ----------
        mid_prices : pd.Series, length N

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        n = len(mid_prices)
        mp = mid_prices.values.astype(np.float64)

        # Forward-looking mean using cumulative sums for efficiency
        labels = np.ones(n, dtype=np.int64)  # default STATIONARY
        for t in range(n - self.k):
            future_mean = mp[t + 1 : t + 1 + self.k].mean()
            l_t = (future_mean - mp[t]) / (mp[t] + 1e-10)
            if l_t > self.alpha:
                labels[t] = 2   # UP
            elif l_t < -self.alpha:
                labels[t] = 0   # DOWN
            # else stays 1 (STATIONARY)
        # Last k steps: no valid future window â†’ mark as STATIONARY (1)
        # These will be trimmed before training but we keep shape consistent.
        return labels

    # ------------------------------------------------------------------
    # Snapshot extraction
    # ------------------------------------------------------------------

    def extract_snapshots(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the raw 10-level bid and ask volumes as book-state snapshots.
        Used by the execution simulation harness and L_EXEC loss function.

        Parameters
        ----------
        df : pd.DataFrame
            Raw LOB DataFrame.

        Returns
        -------
        np.ndarray, shape (N, 20)
            Columns: AskVol_1..10 interleaved with BidVol_1..10.
        """
        cols = []
        for i in range(1, 11):
            cols.extend([f"AskVol_{i}", f"BidVol_{i}"])
        return df[cols].values.astype(np.float32)

    # ------------------------------------------------------------------
    # End-to-end processing
    # ------------------------------------------------------------------

    def process_dataframe(
        self,
        df: pd.DataFrame,
        precomputed_labels: Optional[np.ndarray] = None,
    ) -> LOBDataset:
        """
        Runs the full pipeline on a raw LOB DataFrame and returns a
        PyTorch Dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw LOB data with the 40 canonical LOB columns.
        precomputed_labels : np.ndarray or None
            If provided (shape (N,), values in {0,1,2}), these labels are used
            directly instead of the smooth-label formula.  Pass the output of
            load_raw_file_with_labels() to use the FI-2010 official labels.

        Returns
        -------
        LOBDataset
        """
        feats_df  = self.engineer_features(df)
        snapshots = self.extract_snapshots(df)

        if precomputed_labels is not None:
            # Use official FI-2010 labels — no tail trimming needed
            n = min(len(df), len(precomputed_labels))
            labels = precomputed_labels[:n].astype(np.int64)
            feat_t  = torch.tensor(feats_df.iloc[:n].values, dtype=torch.float32)
            label_t = torch.tensor(labels, dtype=torch.long)
            snap_t  = torch.tensor(snapshots[:n], dtype=torch.float32)
        else:
            # Fallback: smooth-label formula (only usable if alpha is tuned)
            labels = self.generate_labels(feats_df["MidPrice"])
            valid   = len(df) - self.k
            feat_t  = torch.tensor(feats_df.iloc[:valid].values, dtype=torch.float32)
            label_t = torch.tensor(labels[:valid], dtype=torch.long)
            snap_t  = torch.tensor(snapshots[:valid], dtype=torch.float32)

        return _LOBDatasetLegacy(feat_t, label_t, snap_t, seq_len=self.seq_len)

    def load_and_split(
        self,
        train_path: str | os.PathLike,
        val_fraction: float = 0.2,
    ) -> Tuple["LOBDataset", "LOBDataset", pd.DataFrame]:
        """
        Loads a single FI-2010 training file and splits it temporally into
        train and validation sets without shuffling.

        Parameters
        ----------
        train_path : path-like
            Path to the FI-2010 training file.
        val_fraction : float
            Fraction of the training data (from the end) used for validation.

        Returns
        -------
        train_ds, val_ds : LOBDataset
        raw_df : pd.DataFrame  (full raw frame, useful for calibration)
        """
        raw_df, all_labels = self.load_raw_file_with_labels(train_path)
        n = len(raw_df)
        split_idx = int(n * (1 - val_fraction))

        train_raw     = raw_df.iloc[:split_idx].reset_index(drop=True)
        val_raw       = raw_df.iloc[split_idx:].reset_index(drop=True)
        train_labels  = all_labels[:split_idx]
        val_labels    = all_labels[split_idx:]

        return (
            self.process_dataframe(train_raw, train_labels),
            self.process_dataframe(val_raw, val_labels),
            raw_df,
        )

    def load_test(
        self,
        test_paths: List[str | os.PathLike],
    ) -> LOBDataset:
        """
        Loads one or more FI-2010 test files and concatenates them temporally.

        Parameters
        ----------
        test_paths : list of path-like
            Paths to FI-2010 test txt files.

        Returns
        -------
        LOBDataset
        """
        lob_frames, label_arrays = [], []
        for p in test_paths:
            lf, la = self.load_raw_file_with_labels(p)
            lob_frames.append(lf)
            label_arrays.append(la)
        combined        = pd.concat(lob_frames, ignore_index=True)
        combined_labels = np.concatenate(label_arrays)
        return self.process_dataframe(combined, combined_labels)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def print_summary_statistics(
        self, dataset: LOBDataset, name: str = "Dataset"
    ) -> None:
        """
        Prints class imbalance, mean spread, and mean OBI for a dataset split.

        Feature column index map (produced by engineer_features):
          0: MidPrice  1: Spread  2: OBI_1  3: OBI_3  4: OBI_5
          5: WeightedMidPrice  6: TI_10  7: TI_50  8: TI_100
          9: QueueDepthRatio

        Parameters
        ----------
        dataset : LOBDataset
        name : str
            Human-readable label for this split.
        """
        feats = dataset.features.numpy()
        labels = dataset.labels.numpy()
        total = len(labels)

        if total == 0:
            print(f"[{name}] Empty dataset â€” nothing to summarise.")
            return

        down = int((labels == 0).sum())
        stat = int((labels == 1).sum())
        up   = int((labels == 2).sum())

        print(f"\n{'='*50}")
        print(f" Summary Statistics: {name}")
        print(f"{'='*50}")
        print(f" Total samples : {total:,}")
        print(f" Class balance :")
        print(f"   DOWN  (0) : {down:7,}  ({100*down/total:5.1f}%)")
        print(f"   STAT  (1) : {stat:7,}  ({100*stat/total:5.1f}%)")
        print(f"   UP    (2) : {up:7,}  ({100*up/total:5.1f}%)")
        print(f" Mean spread       : {feats[:,1].mean():.6f}")
        print(f" Mean OBI_1        : {feats[:,2].mean():.6f}")
        print(f" Mean OBI_3        : {feats[:,3].mean():.6f}")
        print(f" Mean OBI_5        : {feats[:,4].mean():.6f}")
        print(f" Mean QDepthRatio  : {feats[:,9].mean():.4f}")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Standalone functional API  (load_fi2010_file / verify_labels /
#                              load_fi2010_dataset)
# ---------------------------------------------------------------------------

# k_horizon -> raw row index (before transpose) that holds those labels
_K_TO_ROW: Dict[int, int] = {1: 144, 2: 145, 3: 146, 5: 147, 10: 148}


def load_fi2010_file(
    filepath: str | os.PathLike,
    k_horizon: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one FI-2010 .txt file, enforce the (149, N) raw orientation,
    transpose to (N, 149), extract the 40 LOB features and the requested
    label row, then re-encode labels to 0=DOWN, 1=STATIONARY, 2=UP.

    Parameters
    ----------
    filepath  : path-like
    k_horizon : int
        Forward horizon for precomputed labels. Must be one of {1, 2, 3, 5, 10}.

    Returns
    -------
    X : np.ndarray, shape (N, 40), float32
    y : np.ndarray, shape (N,),   int64
    """
    if k_horizon not in _K_TO_ROW:
        raise ValueError(
            f"k_horizon={k_horizon} not recognised. "
            f"Valid values: {sorted(_K_TO_ROW)}"
        )

    # 1.  Raw load (np.loadtxt returns float64 matrix)
    arr = np.loadtxt(filepath)
    print(f"Raw shape: {arr.shape}")

    # 2.  Assert / coerce (149, N) orientation
    if arr.shape[0] == 149:
        pass                        # already correct
    elif arr.shape[1] == 149:
        arr = arr.T                 # got (N, 149) — transpose to (149, N)
        print(f"  Transposed to: {arr.shape}")
    else:
        raise ValueError(
            f"Unexpected raw array shape {arr.shape}: "
            "expected (149, N) or (N, 149)."
        )

    if arr.shape[0] != 149:
        raise ValueError(
            f"After orientation fix, first dimension is {arr.shape[0]}, expected 149."
        )

    # 3.  Transpose to (N, 149)
    data = arr.T                    # (N, 149)
    print(f"Shape after transpose: {data.shape}")

    # 4.  Features: first 40 columns (raw LOB price/volume fields)
    X = data[:, :40].astype(np.float32)

    # 5.  Labels: the raw row index becomes a column index after transpose
    label_col = _K_TO_ROW[k_horizon]            # e.g. 148 for k=10
    raw_labels = data[:, label_col].astype(np.int64)   # values in {1, 2, 3}

    # 6.  Re-encode: 1(UP)->2, 2(STAT)->1, 3(DOWN)->0   via  3 - label
    y = (3 - raw_labels).astype(np.int64)

    return X, y


def verify_labels(y: np.ndarray) -> None:
    """
    Print class distribution for y in {0, 1, 2} and assert sanity bounds.

    Rules
    -----
    * No single class may exceed 60 % of total samples.
    * Minority class must be at least 15 % of total samples.

    Raises
    ------
    RuntimeError  if either rule is violated.
    """
    total = len(y)
    if total == 0:
        raise RuntimeError("verify_labels: received empty label array.")

    class_names = {0: "DOWN", 1: "STATIONARY", 2: "UP"}
    counts = {c: int((y == c).sum()) for c in [0, 1, 2]}
    pcts   = {c: 100.0 * counts[c] / total for c in [0, 1, 2]}

    print("\nLabel distribution:")
    print(f"  {'Class':<14} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*32}")
    for c in [0, 1, 2]:
        print(f"  {class_names[c]:<14} {counts[c]:>8,}  {pcts[c]:>5.1f}%")
    print(f"  {'TOTAL':<14} {total:>8,}  100.0%")

    max_pct   = max(pcts.values())
    min_pct   = min(pcts.values())
    max_class = max(pcts, key=pcts.get)
    min_class = min(pcts, key=pcts.get)

    errors = []
    if max_pct > 60.0:
        errors.append(
            f"Class {class_names[max_class]} ({max_class}) is {max_pct:.1f}% "
            f"— exceeds 60% ceiling."
        )
    if min_pct < 15.0:
        errors.append(
            f"Class {class_names[min_class]} ({min_class}) is only {min_pct:.1f}% "
            f"— below 15% floor."
        )
    if errors:
        raise RuntimeError("verify_labels FAILED:\n" + "\n".join(errors))

    print("\u2713 LABEL DISTRIBUTION OK")


def load_fi2010_dataset(
    data_dir:   str | os.PathLike,
    k_horizon:  int = 10,
    scaler_path: Optional[str | os.PathLike] = None,
    return_raw: bool = False,
) -> Tuple:
    """
    Discover all FI-2010 .txt files in data_dir, split by Train/Test naming
    convention, load and concatenate each split, verify label distributions,
    fit a StandardScaler on train only, transform both splits, and save the
    fitted scaler.

    Parameters
    ----------
    data_dir    : directory containing FI-2010 .txt files.
    k_horizon   : forward horizon for labels (default 10).
    scaler_path : path to save the fitted scaler; defaults to
                  <data_dir>/fi2010_scaler.joblib.
    return_raw  : if True, also return the pre-normalization feature arrays.
                  Returns (X_train, y_train, X_test, y_test, raw_X_train, raw_X_test).
                  Default False (backward-compatible 4-tuple).

    Returns
    -------
    X_train, y_train, X_test, y_test          (return_raw=False)
    X_train, y_train, X_test, y_test,
      raw_X_train, raw_X_test                 (return_raw=True)
    """
    data_dir  = pathlib.Path(data_dir)
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    train_files = [f for f in txt_files if "Train" in f.name]
    test_files  = [f for f in txt_files if "Test"  in f.name]

    if not train_files:
        raise FileNotFoundError(f"No 'Train' .txt files in {data_dir}")
    if not test_files:
        raise FileNotFoundError(f"No 'Test' .txt files in {data_dir}")

    print(f"\nFound {len(train_files)} train file(s): {[f.name for f in train_files]}")
    print(f"Found {len(test_files)}  test  file(s): {[f.name for f in test_files]}")

    # --- Load train -------------------------------------------------------
    X_trains, y_trains = [], []
    for f in train_files:
        print(f"\n[TRAIN] {f.name}")
        Xf, yf = load_fi2010_file(f, k_horizon=k_horizon)
        X_trains.append(Xf)
        y_trains.append(yf)
    X_train     = np.concatenate(X_trains, axis=0).astype(np.float32)
    y_train     = np.concatenate(y_trains, axis=0).astype(np.int64)
    raw_X_train = X_train.copy()   # save before scaler

    # --- Load test --------------------------------------------------------
    X_tests, y_tests = [], []
    for f in test_files:
        print(f"\n[TEST]  {f.name}")
        Xf, yf = load_fi2010_file(f, k_horizon=k_horizon)
        X_tests.append(Xf)
        y_tests.append(yf)
    X_test     = np.concatenate(X_tests, axis=0).astype(np.float32)
    y_test     = np.concatenate(y_tests, axis=0).astype(np.int64)
    raw_X_test = X_test.copy()     # save before scaler

    # --- Verify labels ----------------------------------------------------
    print("\n--- Train label verification ---")
    verify_labels(y_train)
    print("\n--- Test  label verification ---")
    verify_labels(y_test)

    # --- Normalisation: fit on train ONLY ---------------------------------
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    if scaler_path is None:
        scaler_path = data_dir / "fi2010_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"\n[Scaler] Fitted on {X_train.shape[0]:,} train samples.")
    print(f"[Scaler] Saved -> {scaler_path}")

    # --- Final shapes -----------------------------------------------------
    print(f"\nFinal shapes:")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    if return_raw:
        return X_train, y_train, X_test, y_test, raw_X_train, raw_X_test
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_mid_price_labels(
    dataset: LOBDataset,
    num_points: int = 1000,
    out_path: str = "images/fig_module1_diagnostic.png",
    title: str = "FI-2010 LOB: Mid-Price & Smooth Labels",
) -> None:
    """
    Plots the mid-price time series overlaid with direction labels as coloured
    markers and saves to file.

    Colour key: green = UP (2), grey = STATIONARY (1), red = DOWN (0).

    Parameters
    ----------
    dataset : LOBDataset
    num_points : int
        Number of consecutive time steps to plot (for clarity).
    out_path : str
        Output PNG path.
    title : str
        Figure title.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    mid = dataset.features[:num_points, 0].numpy()   # MidPrice column
    lbl = dataset.labels[:num_points].numpy()
    x   = np.arange(len(mid))

    colour_map = {0: ("red", "DOWN"), 1: ("grey", "STATIONARY"), 2: ("green", "UP")}

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, mid, color="black", linewidth=0.7, alpha=0.6, label="Mid-Price", zorder=1)

    for cls, (colour, cname) in colour_map.items():
        mask = lbl == cls
        if mask.any():
            ax.scatter(
                x[mask], mid[mask],
                color=colour, s=12, alpha=0.85,
                label=f"{cname} ({mask.sum()})",
                zorder=2,
            )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("LOB Update Index")
    ax.set_ylabel("Mid-Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Module 1] Diagnostic plot saved â†’ {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)

    print("=" * 60)
    print("Module 1 - FI-2010 Data Pipeline  [Phase 2 Verification]")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load normalised arrays + raw copies in one pass (return_raw=True
    #    preserves pre-scaler values without re-reading files from disk).
    # ----------------------------------------------------------------
    X_tr, y_tr, X_te, y_te, raw_X_tr, raw_X_te = load_fi2010_dataset(
        data_dir, return_raw=True
    )

    # ----------------------------------------------------------------
    # 2. Build dataloaders + run shape assertions
    # ----------------------------------------------------------------
    train_loader, test_loader, spread_mean = create_dataloaders(
        X_tr, y_tr, X_te, y_te,
        raw_X_train=raw_X_tr,
        raw_X_test=raw_X_te,
    )

    print(f"spread_mean cached: {spread_mean:.6f}")
    print("Phase 2 complete. Proceed to Fix Prompt 3.")
# ---------------------------------------------------------------------------
# Assumptions and edge-case notes
# ---------------------------------------------------------------------------
# ASSUMPTIONS:
#   1. The FI-2010 'NoAuction DecimalPrecision' normalisation variant is used.
#      The auction-included variant has different statistical properties and
#      should NOT be mixed with the NoAuction version.
#   2. The first 40 columns of every FI-2010 row carry the raw LOB levels.
#      Columns 41-144 contain 104 hand-crafted features and labels from the
#      original paper; we ignore them and recompute labels from scratch to
#      ensure reproducibility with arbitrary k and alpha settings.
#   3. Temporal order is strictly preserved â€” no shuffling at any stage.
#   4. Spread is always positive in FI-2010 (decimal-normalised prices).
#      In raw tick data from other sources, zero-spread quotes occur and
#      must be guarded against.
#
# EDGE CASES FOR MANUAL REVIEW:
#   â€¢ Class imbalance: STATIONARY typically represents ~80-90 % of labels at
#     alpha=0.002.  Downstream models MUST use class-weighted losses or
#     oversampling to avoid degenerate STATIONARY-only predictions.
#   â€¢ The TradeIntensity proxy uses mid-price changes as a coarse proxy for
#     order flow.  In real Level-3 data, individual order events (add/cancel/
#     trade) are granular; FI-2010 only shows aggregated book state changes.
#   â€¢ WeightedMidPrice can diverge from MidPrice significantly during one-
#     sided order flow events â€” sanity check the feature distributions before
#     training.
#   â€¢ The last-k-samples trimming means train and validation sizes are
#     slightly shorter than the raw frame minus the split point.
#
# REAL LOB vs FI-2010 DIFFERENCES:
#   â€¢ FI-2010 prices are decimal-normalised per day.  Live market prices
#     require per-day renormalisation or raw tick normalization.
#   â€¢ FI-2010 contains no explicit timestamps â€” the event index is the only
#     time axis.  Annualised metrics (Sharpe ratio) thus require calibrating
#     "events per second" from external intraday microstructure data.
#   â€¢ Auction periods are excluded in the NoAuction variant, but real
#     execution strategies must handle open/close auction dynamics.
# ---------------------------------------------------------------------------


