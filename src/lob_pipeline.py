"""
FI-2010 Limit Order Book Research Pipeline.

Provides modular code for standardizing, extracting features,
generating DeepLOB smooth labels, and wrapping into PyTorch datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import Tuple, List

# FI-2010 Schema mapping (10 levels, Ask/Bid Price/Volume)
# The FI-2010 format consists of 144 columns. The first 40 are the 10 levels of LOB:
# AskPrice_1, AskVol_1, BidPrice_1, BidVol_1, AskPrice_2, AskVol_2 ... BidVol_10
def get_fi2010_columns() -> List[str]:
    """Returns the standardized column names for the first 40 LOB features."""
    cols = []
    for i in range(1, 11):
        cols.extend([f"AskPrice_{i}", f"AskVol_{i}", f"BidPrice_{i}", f"BidVol_{i}"])
    return cols

class LOBDataset(Dataset):
    """
    PyTorch Dataset yielding (feature_sequence, label, book_state_snapshot).
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, snapshots: torch.Tensor, seq_len: int = 10):
        """
        Args:
            features (torch.Tensor): Extracted feature matrix shape (N, num_features).
            labels (torch.Tensor): Classification labels shape (N,).
            snapshots (torch.Tensor): Raw 10-level depths shape (N, 20).
            seq_len (int): Length of historical sequence to use for predictions.
        """
        self.features = features
        self.labels = labels
        self.snapshots = snapshots
        self.seq_len = seq_len

    def __len__(self) -> int:
        # We can extract up to len - seq_len sequences
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        # Sequence of features from t=idx to t=idx+seq_len-1
        feat_seq = self.features[idx:idx + self.seq_len]
        
        # Label is aligned with the last timestamp in the sequence
        target_idx = idx + self.seq_len - 1
        label = int(self.labels[target_idx].item())
        
        # Snapshot of the raw 10-level depths at the same timestamp
        snapshot = self.snapshots[target_idx]
        
        return feat_seq, label, snapshot

class FI2010DataLoader:
    """
    Data ingestion pipeline for FI-2010 benchmark dataset.
    """
    def __init__(self, seq_len: int = 10, k: int = 10, alpha: float = 0.002):
        self.seq_len = seq_len
        self.k = k
        self.alpha = alpha
        self.lob_columns = get_fi2010_columns()
        
    def load_data(self, file_path: str, is_transposed: bool = True) -> pd.DataFrame:
        """
        Loads the FI-2010 format file. FI-2010 csvs are typically transposed (144, N).
        If the file has standard shape (N, 144) set is_transposed=False.
        """
        df = pd.read_csv(file_path, header=None)
        if is_transposed and df.shape[0] < df.shape[1]:
            df = df.T
            
        # We only need the first 40 columns for raw LOB data
        lob_data = df.iloc[:, :40].copy()
        lob_data.columns = self.lob_columns
        return lob_data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes engineered features based on raw 10-level bid/ask prices and volumes.
        """
        feats = pd.DataFrame(index=df.index)
        
        # 1. Mid-price
        feats['MidPrice'] = (df['AskPrice_1'] + df['BidPrice_1']) / 2.0
        
        # 2. Bid-Ask Spread
        feats['Spread'] = df['AskPrice_1'] - df['BidPrice_1']
        
        # 3. Order Book Imbalance (OBI) at levels 1, 3, 5
        for levels in [1, 3, 5]:
            bid_vols = sum(df[f'BidVol_{i}'] for i in range(1, levels + 1))
            ask_vols = sum(df[f'AskVol_{i}'] for i in range(1, levels + 1))
            feats[f'OBI_{levels}'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)
            
        # 4. Weighted Mid-Price across all 10 levels
        num = sum(df[f'BidPrice_{i}'] * df[f'BidVol_{i}'] + df[f'AskPrice_{i}'] * df[f'AskVol_{i}'] for i in range(1, 11))
        den = sum(df[f'BidVol_{i}'] + df[f'AskVol_{i}'] for i in range(1, 11))
        feats['WeightedMidPrice'] = num / (den + 1e-8)
        
        # 5. Trade Intensity Proxy: rolling count of mid-price changes
        mid_price_diff = feats['MidPrice'].diff().fillna(0).abs() > 0
        for w in [10, 50, 100]:
            feats[f'TradeIntensity_{w}'] = mid_price_diff.rolling(window=w, min_periods=1).sum()
            
        # 6. Queue Depth Ratio across all 10 levels
        total_bid_depth = sum(df[f'BidVol_{i}'] for i in range(1, 11))
        total_ask_depth = sum(df[f'AskVol_{i}'] for i in range(1, 11))
        feats['QueueDepthRatio'] = total_bid_depth / (total_ask_depth + 1e-8)
        
        return feats

    def generate_labels(self, mid_prices: pd.Series) -> np.ndarray:
        """
        Smooth labeling method (Zhang et al., 2019).
        Predicts upward movement (2), stationary (1), or downward (0).
        """
        # m_t is the mean of the NEXT k mid-prices
        m_t = mid_prices.rolling(window=self.k, min_periods=self.k).mean().shift(-self.k)
        
        l_t = (m_t - mid_prices) / mid_prices
        
        labels = np.ones(len(mid_prices), dtype=int) # Default stationary = 1
        labels[l_t > self.alpha] = 2 # Up
        labels[l_t < -self.alpha] = 0 # Down
        
        return labels

    def extract_snapshots(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts raw 10-level bid and ask volumes as book state snapshots.
        Shape will be (N, 20) -> 10 ask vols, 10 bid vols.
        """
        cols = []
        for i in range(1, 11):
            cols.extend([f"AskVol_{i}", f"BidVol_{i}"])
        return df[cols].values

    def process_file(self, file_path: str, is_transposed: bool = True) -> LOBDataset:
        """
        E2E processing returning a PyTorch dataset.
        """
        df = self.load_data(file_path, is_transposed)
        return self.process_dataframe(df)

    def process_dataframe(self, df: pd.DataFrame) -> LOBDataset:
        """
        Internal function useful for processing data splits directly.
        """
        feats_df = self.engineer_features(df)
        labels = self.generate_labels(feats_df['MidPrice'])
        snapshots = self.extract_snapshots(df)
        
        # Drop the last `k` rows because their labels are invalid (NaN m_t).
        valid_idx = len(df) - self.k
        
        feat_tensor = torch.tensor(feats_df.iloc[:valid_idx].values, dtype=torch.float32)
        label_tensor = torch.tensor(labels[:valid_idx], dtype=torch.long)
        snap_tensor = torch.tensor(snapshots[:valid_idx], dtype=torch.float32)
        
        return LOBDataset(feat_tensor, label_tensor, snap_tensor, seq_len=self.seq_len)

    def print_summary_statistics(self, dataset: LOBDataset, dataset_name: str = "Dataset"):
        """
        Prints class imbalance, mean spread, and mean OBI for the dataset split.
        """
        labels = dataset.labels.numpy()
        feats = dataset.features.numpy()
        
        # Feature column index reference:
        # 0: MidPrice, 1: Spread, 2: OBI_1, 3: OBI_3, 4: OBI_5,
        # 5: WeightedMidPrice, 6: TradeIntensity_10, 7: TradeIntensity_50, 
        # 8: TradeIntensity_100, 9: QueueDepthRatio
        spreads = feats[:, 1]
        obi_1 = feats[:, 2]
        obi_3 = feats[:, 3]
        obi_5 = feats[:, 4]
        
        down_count = np.sum(labels == 0)
        stat_count = np.sum(labels == 1)
        up_count = np.sum(labels == 2)
        total = len(labels)
        
        print(f"--- Summary Statistics: {dataset_name} ---")
        print(f"Total valid samples: {total}")
        print(f"Class Imbalance:")
        if total > 0:
            print(f"  Down (0):       {down_count} ({down_count/total*100:.2f}%)")
            print(f"  Stationary (1): {stat_count} ({stat_count/total*100:.2f}%)")
            print(f"  Up (2):         {up_count} ({up_count/total*100:.2f}%)")
        print(f"Mean Spread: {np.mean(spreads):.6f}")
        print(f"Mean OBI_1:  {np.mean(obi_1):.6f}")
        print(f"Mean OBI_3:  {np.mean(obi_3):.6f}")
        print(f"Mean OBI_5:  {np.mean(obi_5):.6f}")
        print("-" * 40)

def plot_diagnostics(mid_prices: np.ndarray, labels: np.ndarray, num_points: int = 500, out_path: str = "diagnostic_plot.png"):
    """
    Plots the mid-price time series for the first stock/chunk, 
    overlaying the generated labels as colored markers.
    """
    plt.figure(figsize=(14, 6))
    
    mp_slice = mid_prices[:num_points]
    lbl_slice = labels[:num_points]
    
    x = np.arange(len(mp_slice))
    
    # Plot line
    plt.plot(x, mp_slice, color='black', alpha=0.5, label='Mid Price')
    
    # Scatter points for labels
    # 0 = down (red), 1 = stationary (grey), 2 = up (green)
    colors = {0: 'red', 1: 'grey', 2: 'green'}
    
    for lbl, color in colors.items():
        mask = (lbl_slice == lbl)
        if np.any(mask):
            plt.scatter(x[mask], mp_slice[mask], color=color, s=20, 
                        label=f'Label {lbl} ({"Down" if lbl==0 else "Up" if lbl==2 else "Stationary"})')
        
    plt.title('FI-2010 Limit Order Book: Mid-Price and Movement Labels')
    plt.xlabel('Time Step (Updates)')
    plt.ylabel('Mid Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Diagnostic plot saved to {os.path.abspath(out_path)}")
    plt.close()

if __name__ == '__main__':
    # ========================================================
    # Mock Data Execution to Verify Pipeline Workings
    # ========================================================
    print("Initializing FI-2010 DataLoader Pipeline test run...")
    np.random.seed(42)
    # Generate 2000 events mimicking LOB levels
    mock_data = np.random.rand(2000, 40) * 100
    df_mock = pd.DataFrame(mock_data)
    
    # Structuring mock data so Spread is strictly > 0 and fluctuates realistically
    for i in range(1, 11): 
        ask_p_idx = (i-1)*4
        bid_p_idx = (i-1)*4 + 2
        base_price = 1000 + np.sin(np.linspace(0, 10, 2000)) * 50 # Oscillating mid price
        df_mock.iloc[:, bid_p_idx] = base_price - i * 0.5 - np.random.rand(2000) * 0.2
        df_mock.iloc[:, ask_p_idx] = base_price + i * 0.5 + np.random.rand(2000) * 0.2
        
    loader = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
    df_mock.columns = loader.lob_columns
    
    # We split 1500 train, 500 test representing a temporal dataset split without shuffling
    train_df = df_mock.iloc[:1500]
    test_df = df_mock.iloc[1500:]
    
    train_ds = loader.process_dataframe(train_df)
    test_ds = loader.process_dataframe(test_df)
    
    loader.print_summary_statistics(train_ds, "Train Split (First 1500 events)")
    loader.print_summary_statistics(test_ds, "Test Split (Last 500 events)")
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_plot.png")
    plot_diagnostics(train_ds.features[:, 0].numpy(), train_ds.labels.numpy(), num_points=500, out_path=plot_path)

    print("Pipeline code executed successfully. Classes are modular and ready for the FI-2010 framework.")

