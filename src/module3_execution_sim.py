"""
Module 3 — Execution Simulation Harness
=========================================
Paper-trading execution simulator for evaluating LOB prediction models
on real execution quality, not just classification accuracy.

Components
----------
  QueueModel                  Pro-rata queue position estimator.
  ExecutionProbabilityEstimator  Logistic-regression P(exec) model.
  PaperTradingSimulator       Event-driven paper-trading engine.
  compute_execution_metrics() Aggregated execution quality statistics.

All components operate on LOBDataset outputs from Module 1 and class
predictions from Module 2 models.

Usage:
  python -m src.module3_execution_sim
"""

import os
from tqdm import tqdm
import pathlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.module1_data_pipeline import LOBDataset

warnings.filterwarnings("ignore", category=UserWarning)

FIGURES_DIR = pathlib.Path("images")
LOGS_DIR    = pathlib.Path("logs")


# ---------------------------------------------------------------------------
# 1. Queue Position Model
# ---------------------------------------------------------------------------

class QueueModel:
    """
    Pro-rata queue position estimator for best-level limit orders.

    At time t, a new limit order submitted at the best bid (or ask) joins
    the *back* of the queue.  Using the average volume consumed per LOB
    update event (fitted from training data), we estimate the expected
    number of events tau before the order is filled:

      tau = best_level_volume / avg_volume_consumed_per_event

    Parameters
    ----------
    None — parameters are estimated from data via fit().
    """

    def __init__(self) -> None:
        self.avg_vol_per_event: float = 1.0   # default fallback before fit
        self._is_fitted       : bool  = False

    # ------------------------------------------------------------------

    def fit(self, dataset: LOBDataset) -> None:
        """
        Estimates the average volume consumed per LOB update event.

        We approximate consumed volume as the absolute change in best-level
        bid or ask volume between consecutive snapshots.  Snapshots contain
        [AskVol_1, BidVol_1, AskVol_2, BidVol_2, …] at indices 0, 1, 2, 3 …

        Parameters
        ----------
        dataset : LOBDataset
            Training dataset.  Uses the raw volume snapshots (shape N, 20).
        """
        snaps = dataset.snapshots.numpy()     # (N, 20)

        # Best-ask volume = column 0, best-bid volume = column 1
        ask_vol_1 = snaps[:, 0]
        bid_vol_1 = snaps[:, 1]

        # Absolute change between consecutive events
        ask_change = np.abs(np.diff(ask_vol_1))
        bid_change = np.abs(np.diff(bid_vol_1))
        all_changes = np.concatenate([ask_change, bid_change])
        all_changes = all_changes[all_changes > 0]   # ignore no-change events

        self.avg_vol_per_event = float(np.mean(all_changes)) if len(all_changes) > 0 else 1.0
        self._is_fitted = True
        print(f"[QueueModel] avg_vol_per_event = {self.avg_vol_per_event:.4f}")

    # ------------------------------------------------------------------

    def estimate_tau(self, snapshot: np.ndarray, side: str = "ask") -> float:
        """
        Estimates expected queue wait time tau (in LOB update events) for
        a limit order at the best level.

        Parameters
        ----------
        snapshot : np.ndarray, shape (20,)
            [AskVol_1, BidVol_1, AskVol_2, BidVol_2, ...]
        side : str
            'ask' (for buying) or 'bid' (for selling).

        Returns
        -------
        float  — expected events to fill.
        """
        if not self._is_fitted:
            raise RuntimeError("Call QueueModel.fit() before estimate_tau().")
        # Best-ask volume at index 0, best-bid at index 1
        vol = float(snapshot[0] if side == "ask" else snapshot[1])
        return max(vol / (self.avg_vol_per_event + 1e-8), 0.0)


# ---------------------------------------------------------------------------
# 2. Execution Probability Estimator
# ---------------------------------------------------------------------------

class ExecutionProbabilityEstimator:
    """
    Logistic regression estimating P(execution) for a hypothetical market
    order submitted at the current LOB state in the predicted direction.

    Features used: [spread, OBI_1, queue_depth_ratio, predicted_direction_encoded]

    Target (binary):
      1 if mid-price moved in predicted direction within the next k=10
      LOB update events AND the hypothetical order would have been at
      a price accessible in that window.  Approximated as:
        filled_1 = (label == predicted_direction)
      for in-sample calibration.

    Parameters
    ----------
    k : int
        Look-ahead horizon (default 10, aligned with label generation).
    """

    def __init__(self, k: int = 10) -> None:
        self.k       = k
        self.scaler  = StandardScaler()
        self.clf     = LogisticRegression(
            solver="lbfgs", max_iter=500,
            class_weight="balanced", random_state=42
        )
        self._is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: LOBDataset,
        predictions: np.ndarray,
    ) -> None:
        """
        Fits the execution probability model on training data.

        Parameters
        ----------
        dataset : LOBDataset
            Training dataset (features must include Spread at col 1,
            OBI_1 at col 2, QueueDepthRatio at col 9).
        predictions : np.ndarray, shape (N,)
            Model direction predictions aligned with dataset samples.
        """
        feats  = dataset.features.numpy()
        labels = dataset.labels.numpy()

        # Align features/labels to prediction length (windowed models produce
        # fewer predictions than raw feature rows)
        n_pred = len(predictions)
        if n_pred < len(feats):
            feats  = feats[-n_pred:]
            labels = labels[-n_pred:]

        spread            = feats[:, 1]
        obi_1             = feats[:, 2]
        queue_depth_ratio = feats[:, 9]

        # Encode predicted direction as numeric
        pred_dir = predictions.astype(np.float32) - 1.0   # {-1, 0, +1}

        X = np.column_stack([spread, obi_1, queue_depth_ratio, pred_dir])
        # Binary target: 1 if model predicts correctly (simplification)
        y = (predictions == labels).astype(np.int32)

        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.clf.fit(Xs, y)

        self.coef_      = self.clf.coef_.copy()       # (1, 4)
        self.intercept_ = float(self.clf.intercept_[0])
        self._is_fitted = True

        train_acc = float((self.clf.predict(Xs) == y).mean())
        print(
            f"[ExecProbEstimator] Fitted.  Train accuracy on fill proxy = "
            f"{train_acc:.4f}"
        )
        print(
            f"  Coefficients: spread={self.coef_[0,0]:.4f}  "
            f"obi1={self.coef_[0,1]:.4f}  "
            f"qdr={self.coef_[0,2]:.4f}  "
            f"pred_dir={self.coef_[0,3]:.4f}"
        )

    # ------------------------------------------------------------------

    def predict_proba(
        self,
        spread: float,
        obi_1: float,
        queue_depth_ratio: float,
        predicted_direction: int,
    ) -> float:
        """
        Returns P(execution) ∈ [0, 1].

        Parameters
        ----------
        spread, obi_1, queue_depth_ratio : float
            Scalar LOB state features at time t.
        predicted_direction : int (0=DOWN, 1=STAT, 2=UP)

        Returns
        -------
        float : probability of execution.
        """
        if not self._is_fitted:
            raise RuntimeError("Call ExecutionProbabilityEstimator.fit() first.")
        pred_dir = float(predicted_direction) - 1.0
        X = np.array([[spread, obi_1, queue_depth_ratio, pred_dir]])
        Xs = self.scaler.transform(X)
        return float(self.clf.predict_proba(Xs)[0, 1])


# ---------------------------------------------------------------------------
# 3. Paper Trading Simulator
# ---------------------------------------------------------------------------

class PaperTradingSimulator:
    """
    Event-driven paper-trading engine.

    At each LOB update t:
      • UP prediction   -> simulate a market BUY order.
      • DOWN prediction -> simulate a market SELL order.
      • STATIONARY      -> no action.

    Fill logic (both conditions must hold):
      (a) Mid-price moves in predicted direction within next k steps.
      (b) P(exec) > exec_prob_threshold.

    PnL is measured in ticks (integer multiples of minimum price increment).

    Parameters
    ----------
    queue_model   : QueueModel (fitted)
    exec_estimator: ExecutionProbabilityEstimator (fitted)
    k             : int, look-ahead window for fill detection.
    exec_prob_threshold : float, minimum P(exec) to classify as filled.
    tick_size     : float, minimum price increment (default 1e-4 for normalised).
    """

    def __init__(
        self,
        queue_model: QueueModel,
        exec_estimator: ExecutionProbabilityEstimator,
        k: int = 10,
        exec_prob_threshold: float = 0.5,
        tick_size: float = 1e-4,
    ) -> None:
        self.queue_model      = queue_model
        self.exec_estimator   = exec_estimator
        self.k                = k
        self.exec_prob_thresh = exec_prob_threshold
        self.tick_size        = tick_size

        # Run results (populated by run())
        self.pnl_curve        : List[float] = []
        self.trade_log        : List[Dict]  = []
        self._is_run          : bool        = False

    # ------------------------------------------------------------------

    def run(
        self,
        dataset: LOBDataset,
        predictions: np.ndarray,
    ) -> None:
        """
        Simulates paper trading over the entire dataset using supplied predictions.

        Parameters
        ----------
        dataset     : LOBDataset
        predictions : np.ndarray, shape (N,), integer direction predictions.
        """
        feats     = dataset.features.numpy()    # (N, 10)
        mid_prices = feats[:, 0]                # MidPrice
        spread     = feats[:, 1]
        obi_1      = feats[:, 2]
        qdr        = feats[:, 9]
        snaps      = dataset.snapshots.numpy()  # (N, 20)

        n = min(len(predictions), len(feats))
        pnl = 0.0
        self.pnl_curve = []
        self.trade_log = []

        for t in tqdm(range(n), desc="  Simulating", unit="step",
                      leave=False, dynamic_ncols=True):
            pred_dir = int(predictions[t])
            self.pnl_curve.append(pnl)

            if pred_dir == 1:                   # STATIONARY -> no trade
                continue

            # Compute execution probability
            p_exec = self.exec_estimator.predict_proba(
                spread=float(spread[t]),
                obi_1=float(obi_1[t]),
                queue_depth_ratio=float(qdr[t]),
                predicted_direction=pred_dir,
            )

            # Compute queue wait time for latency discount
            side = "ask" if pred_dir == 2 else "bid"
            tau  = self.queue_model.estimate_tau(snaps[t], side=side)

            # Fill check: does mid move in predicted direction within k steps?
            future_end = min(t + self.k + 1, n)
            if future_end > t + 1:
                future_mid = mid_prices[t + 1 : future_end]
                if pred_dir == 2:
                    moved_correctly = bool(future_mid[-1] > mid_prices[t])
                else:  # DOWN
                    moved_correctly = bool(future_mid[-1] < mid_prices[t])
            else:
                moved_correctly = False

            filled = moved_correctly and (p_exec > self.exec_prob_thresh)

            # PnL calculation (in ticks)
            if filled:
                future_mid_mean = float(np.mean(mid_prices[t+1:future_end]))
                price_move = future_mid_mean - mid_prices[t]
                tick_pnl   = (price_move / self.tick_size) * (1 if pred_dir == 2 else -1)
            else:
                tick_pnl = 0.0

            pnl += tick_pnl

            # Adverse fill: filled but mid reversed in next k steps after fill
            if filled and future_end + self.k < n:
                post_mid = mid_prices[future_end : future_end + self.k]
                adverse  = bool(
                    (pred_dir == 2 and post_mid[-1] < future_mid_mean) or
                    (pred_dir == 0 and post_mid[-1] > future_mid_mean)
                )
            else:
                adverse = False

            self.trade_log.append({
                "t"               : t,
                "pred_dir"        : pred_dir,
                "p_exec"          : round(p_exec, 4),
                "tau"             : round(tau, 2),
                "moved_correctly" : moved_correctly,
                "filled"          : filled,
                "tick_pnl"        : round(tick_pnl, 4),
                "adverse_fill"    : adverse,
                "cumulative_pnl"  : round(pnl, 4),
            })

        self.pnl_curve.append(pnl)   # final step
        self._is_run = True

    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the trade log as a DataFrame."""
        if not self._is_run:
            raise RuntimeError("Call run() before to_dataframe().")
        return pd.DataFrame(self.trade_log)


# ---------------------------------------------------------------------------
# 4. Execution quality metrics
# ---------------------------------------------------------------------------

def compute_execution_metrics(
    simulator: PaperTradingSimulator,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Computes execution quality metrics from a completed PaperTradingSimulator run.

    Returns
    -------
    Dict with keys:
      total_pnl_ticks          : Total simulated PnL in ticks.
      annualized_sharpe         : Annualized Sharpe ratio of the daily PnL curve.
      fill_rate                 : filled / total_non_stationary_predictions.
      execution_weighted_accuracy: (correct_and_filled) / total_non_stationary.
      max_drawdown              : Maximum peak-to-trough drawdown in ticks.
      n_trades                  : Total number of non-stationary predictions.
      n_filled                  : Number of filled orders.
      n_adverse_fills           : Number of adverse fills.
    """
    if not simulator._is_run:
        raise RuntimeError("Call PaperTradingSimulator.run() first.")

    log = simulator.to_dataframe()

    if len(log) == 0:
        # No trades — return degenerate metrics
        return {
            "total_pnl_ticks"            : 0.0,
            "annualized_sharpe"           : 0.0,
            "fill_rate"                   : 0.0,
            "execution_weighted_accuracy" : 0.0,
            "max_drawdown"                : 0.0,
            "n_trades"                    : 0,
            "n_filled"                    : 0,
            "n_adverse_fills"             : 0,
        }

    total_pnl     = float(simulator.pnl_curve[-1])
    pnl_arr       = np.array(simulator.pnl_curve)
    n_trades      = len(log)
    n_filled      = int(log["filled"].sum())
    n_adverse     = int(log["adverse_fill"].sum())
    n_correct_filled = int((log["filled"] & log["moved_correctly"]).sum())

    fill_rate    = n_filled  / max(n_trades, 1)
    ewa          = n_correct_filled / max(n_trades, 1)

    # Sharpe: split PnL curve into pseudo-daily buckets
    # (assuming all trading days have equal number of events)
    n_events      = len(pnl_arr)
    events_per_day = max(n_events // trading_days_per_year, 1)
    daily_pnl     = np.array([
        pnl_arr[min((i + 1) * events_per_day, n_events - 1)]
        - pnl_arr[i * events_per_day]
        for i in range(n_events // events_per_day)
    ])
    if daily_pnl.std() > 1e-10:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(trading_days_per_year))
    else:
        sharpe = 0.0

    # Maximum drawdown
    running_max = np.maximum.accumulate(pnl_arr)
    drawdowns   = running_max - pnl_arr
    max_dd      = float(drawdowns.max())

    return {
        "total_pnl_ticks"            : round(total_pnl, 4),
        "annualized_sharpe"           : round(sharpe, 4),
        "fill_rate"                   : round(fill_rate, 4),
        "execution_weighted_accuracy" : round(ewa, 4),
        "max_drawdown"                : round(max_dd, 4),
        "n_trades"                    : n_trades,
        "n_filled"                    : n_filled,
        "n_adverse_fills"             : n_adverse,
    }


# ---------------------------------------------------------------------------
# 5. Baseline evaluation pipeline
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    models: List,      # List[BaseModel]
    test_ds: LOBDataset,
    train_ds: LOBDataset,
    out_csv: str = "logs/execution_results.csv",
    out_fig: str = "images/fig_module3_motivation_gap.png",
) -> pd.DataFrame:
    """
    Runs all baseline models through the PaperTradingSimulator on the test set.

    Saves results to CSV and generates the motivation-gap figure:
      subplot (a): Classification macro-F1 per model.
      subplot (b): Execution-weighted accuracy + annualised Sharpe per model.

    Parameters
    ----------
    models    : list of fitted BaseModel instances
    test_ds   : LOBDataset  (test set)
    train_ds  : LOBDataset  (used to fit QueueModel and ExecProbEstimator)
    out_csv   : output CSV path
    out_fig   : output PNG path

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Fit shared simulation components on train ────────────────────────
    qm = QueueModel()
    qm.fit(train_ds)

    rows = []
    model_bar = tqdm(models, desc="[Module 3] Eval", unit="model", dynamic_ncols=True)
    for model in model_bar:
        model_bar.set_postfix(model=model.name)
        tqdm.write(f"\n[Module 3] Simulating: {model.name}")

        # Train-set predictions for fitting the exec estimator
        train_preds = model.predict(train_ds)
        epe = ExecutionProbabilityEstimator(k=10)
        epe.fit(train_ds, train_preds)

        sim = PaperTradingSimulator(queue_model=qm, exec_estimator=epe, k=10)
        test_preds = model.predict(test_ds)
        sim.run(test_ds, test_preds)

        exec_metrics = compute_execution_metrics(sim)

        from src.module2_baselines import compute_classification_metrics
        _labels = test_ds.labels.numpy()
        if len(test_preds) < len(_labels):
            _labels = _labels[-len(test_preds):]
        cls_metrics  = compute_classification_metrics(
            _labels, test_preds
        )

        row = {"model": model.name}
        row.update(cls_metrics)
        row.update(exec_metrics)
        rows.append(row)

        print(
            f"  F1={cls_metrics['F1_macro']:.4f}  "
            f"EWA={exec_metrics['execution_weighted_accuracy']:.4f}  "
            f"Sharpe={exec_metrics['annualized_sharpe']:.4f}  "
            f"PnL={exec_metrics['total_pnl_ticks']:.1f} ticks"
        )

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(out_csv)
    print(f"\n[Module 3] Results saved -> {out_csv}")

    _plot_motivation_gap(df, out_fig)
    return df


def _plot_motivation_gap(df: pd.DataFrame, out_path: str) -> None:
    """Produces the two-panel 'motivation gap' figure."""
    models = list(df.index)
    x      = np.arange(len(models))
    width  = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── (a) Classification F1 ────────────────────────────────────────────
    f1_vals = df["F1_macro"].fillna(0).values
    bars1   = ax1.bar(x, f1_vals, width, color="steelblue", alpha=0.85, label="Macro-F1")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("(a) Classification Performance")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # ── (b) Execution-weighted accuracy & Sharpe ────────────────────────
    ewa_vals    = df["execution_weighted_accuracy"].fillna(0).values
    sharpe_vals = df["annualized_sharpe"].fillna(0).values

    bars2a = ax2.bar(x - width/2, ewa_vals,    width, color="green",  alpha=0.75, label="Exec-Wtd Accuracy")
    bars2b = ax2.bar(x + width/2, sharpe_vals, width, color="orange", alpha=0.75, label="Ann. Sharpe")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Value")
    ax2.set_title("(b) Execution Quality")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Motivation Gap: Classification Accuracy vs Execution Quality", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Module 3] Motivation gap figure saved -> {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.module1_data_pipeline import FI2010DataLoader, download_fi2010, DATA_DIR
    from src.module2_baselines import (
        MomentumBaseline, OLSImbalanceModel, RandomForestLOB,
        XGBoostLOB, DeepLOBModel, train_model, _XGB_AVAILABLE
    )

    # ── Data ──────────────────────────────────────────────────────────────
    paths    = download_fi2010(DATA_DIR)
    loader   = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
    train_ds, val_ds, _ = loader.load_and_split(paths["Train"], val_fraction=0.2)
    test_ds  = loader.load_test([paths["Test1"], paths["Test2"], paths["Test3"]])

    # ── Train baselines ───────────────────────────────────────────────────
    classical = [MomentumBaseline(), OLSImbalanceModel(), RandomForestLOB()]
    if _XGB_AVAILABLE:
        classical.append(XGBoostLOB())
    for m in classical:
        m.fit(train_ds, val_ds)

    deeplob = DeepLOBModel(seq_len=10, max_epochs=50, patience=5)
    train_model(deeplob, train_ds, val_ds)

    all_models = classical + [deeplob]

    # ── Run simulator ──────────────────────────────────────────────────────
    results_df = run_baseline_evaluation(
        all_models, test_ds, train_ds,
        out_csv="logs/execution_results_module3.csv",
        out_fig="images/fig_module3_motivation_gap.png",
    )

    print("\n=== Module 3 Execution Results ===")
    print(results_df[["F1_macro", "execution_weighted_accuracy",
                        "annualized_sharpe", "total_pnl_ticks"]].to_string())


# ---------------------------------------------------------------------------
# Assumptions and edge-case notes
# ---------------------------------------------------------------------------
# ASSUMPTIONS:
#   1. "Filled" orders: we use a dual condition (mid moves correctly AND
#      P(exec) > 0.5).  This is an approximation — in genuine limit order
#      execution, fill depends on queue position depletion, not just price
#      movement.  The QueueModel provides a rough estimate of tau but does
#      NOT simulate individual queue drain events.
#   2. Tick size: FI-2010 uses decimal-normalised prices.  The default
#      tick_size=1e-4 corresponds approximately to the normalised minimum
#      price increment.  For raw tick data you would set this to the actual
#      tick size (e.g. 0.01 for equities with cent granularity).
#   3. PnL is measured as mid-price change, not bid/ask crossing cost.
#      Half-spread cost is NOT subtracted here; it will be modelled inside
#      L_EXEC via the asymmetric cost matrix.
#   4. The ExecutionProbabilityEstimator is fitted on *training set* model
#      predictions, not the true oracle.  This creates a circular dependency
#      that may slightly overfit the P(exec) calibration.  For production
#      use, fit on a held-out calibration set.
#
# EDGE CASES FOR MANUAL REVIEW:
#   • If a model predicts STATIONARY for more than ~80 % of samples (which
#     often happens with unbalanced training), total n_trades will be very
#     small and all execution metrics will have high variance.  Monitor this
#     in the baseline evaluation logs.
#   • The Sharpe ratio is computed from pseudo-daily buckets assuming uniform
#     event density.  FI-2010 session boundaries should be respected if
#     available (no events arrive overnight).
#   • adverse_fill detection only checks k steps after the fill window.  If
#     the test set ends within k steps of a fill, adverse_fill is set False
#     by default (conservative).
#
# REAL LOB vs FI-2010 DIFFERENCES:
#   • Real market orders interact with the full depth of the book and can
#     sweep multiple price levels.  Our simulator assumes the order is small
#     enough to be filled at the best level without market impact.
#   • The FI-2010 dataset contains no arrival timestamps.  Computing the
#     annualised Sharpe ratio requires an assumption about events per trading
#     day, which is not recoverable from the benchmark alone.
# ---------------------------------------------------------------------------

