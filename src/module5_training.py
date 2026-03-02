"""
Module 5 — Training, Ablation, and Results
===========================================
Re-trains DeepLOB and XGBoost with L_EXEC and runs the full experimental
suite including an ablation study and publication-quality figures.

Workflow
--------
  1. Retrain DeepLOB with L_EXEC (same hyperparams as baseline run).
  2. Lambda grid search over [0.01, 0.05, 0.1, 0.25, 0.5, 1.0].
  3. Full evaluation of all models through PaperTradingSimulator.
  4. Ablation study: four L_EXEC variants × simulator.
  5. Publication-ready figures (300 DPI):
       Figure 1 : PnL curves for all models.
       Figure 2 : F1 vs Execution-Weighted Accuracy bar chart.
       Figure 3 : Ablation heatmap.
       Figure 4 : L_EXEC training loss decomposition curves.

Usage:
  python -m src.module5_training
"""

import os
import copy
import csv
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
import matplotlib.ticker as mticker

from src.module1_data_pipeline import FI2010DataLoader, LOBDataset, download_fi2010, DATA_DIR
from src.module2_baselines import (
    MomentumBaseline, OLSImbalanceModel, RandomForestLOB,
    DeepLOBModel, DeepLOBNet, train_model, compute_classification_metrics,
    N_CLASSES, _XGB_AVAILABLE,
)
from src.module3_execution_sim import (
    QueueModel, ExecutionProbabilityEstimator,
    PaperTradingSimulator, compute_execution_metrics,
)
from src.module4_loss_function import LExecLoss

if _XGB_AVAILABLE:
    from src.module2_baselines import XGBoostLOB

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
FIGURES_DIR     = pathlib.Path("images")
LOGS_DIR        = pathlib.Path("logs")
CHECKPOINTS_DIR = pathlib.Path("checkpoints")

for d in [FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_sim(
    model,
    test_ds: LOBDataset,
    train_ds: LOBDataset,
) -> Tuple[Dict[str, Any], PaperTradingSimulator]:
    """
    Fits a QueueModel + ExecutionProbabilityEstimator on train_ds,
    runs PaperTradingSimulator on test_ds, returns metrics + simulator.
    """
    qm = QueueModel(); qm.fit(train_ds)
    train_preds = model.predict(train_ds)
    epe = ExecutionProbabilityEstimator(k=10)
    epe.fit(train_ds, train_preds)
    sim = PaperTradingSimulator(queue_model=qm, exec_estimator=epe, k=10)
    test_preds = model.predict(test_ds)
    sim.run(test_ds, test_preds)
    metrics = compute_execution_metrics(sim)
    _labels = test_ds.labels.numpy()
    if len(test_preds) < len(_labels):
        _labels = _labels[-len(test_preds):]
    cls     = compute_classification_metrics(_labels, test_preds)
    return {**cls, **metrics}, sim


# ---------------------------------------------------------------------------
# 1. DeepLOB training loop enhanced for L_EXEC
# ---------------------------------------------------------------------------

def train_deeplob_with_lexec(
    train_ds     : LOBDataset,
    val_ds       : LOBDataset,
    loss_fn      : LExecLoss,
    checkpoint   : str  = "checkpoints/deeplob_lexec.pt",
    log_csv      : str  = "logs/deeplob_lexec_training.csv",
    seq_len      : int  = 10,
    max_epochs   : int  = 50,
    patience     : int  = 5,
    lr           : float = 1e-4,
    batch_size   : int  = 256,
    device       : Optional[str] = None,
) -> Tuple[DeepLOBNet, List[Dict]]:
    """
    Trains DeepLOBNet with L_EXEC as the loss function.

    The training loop passes (logits, labels, snapshots) to the loss
    function and logs the three L_EXEC component contributions at each epoch.

    Parameters
    ----------
    train_ds, val_ds : LOBDataset
    loss_fn       : LExecLoss (already initialised with correct mean_spread)
    checkpoint    : path to save best model weights
    log_csv       : path for per-epoch CSV log
    seq_len       : LSTM look-back
    max_epochs    : int
    patience      : early-stopping patience on val macro-F1
    lr            : Adam learning rate
    batch_size    : int
    device        : 'cpu'|'cuda'|None (auto-detect)

    Returns
    -------
    DeepLOBNet   : trained network
    history      : list of per-epoch dicts
    """
    _set_seed(RANDOM_SEED)
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[train_deeplob_with_lexec] Using device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    net = DeepLOBNet(seq_len=seq_len, n_classes=N_CLASSES).to(dev)
    loss_fn = loss_fn.to(dev)

    # Joint optimisation: network params + cost matrix + exec MLP
    optimizer = optim.Adam(
        list(net.parameters()) + list(loss_fn.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)

    # ---------- Build data loaders ----------
    def _build(ds: LOBDataset, shuffle: bool) -> DataLoader:
        n   = len(ds)
        T   = seq_len
        X_l, y_l, s_l = [], [], []
        for i in range(n):
            f, lbl, snap = ds[i]
            F_ = f.shape[1]
            if F_ < 40:
                pad = torch.zeros(T, 40 - F_)
                f   = torch.cat([f, pad], dim=1)
            X_l.append(f); y_l.append(lbl); s_l.append(snap)
        X = torch.stack(X_l)
        y = torch.tensor(y_l, dtype=torch.long)
        s = torch.stack(s_l)
        td = TensorDataset(X, y, s)
        _pin = dev.type == "cuda"
        return DataLoader(td, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=_pin)

    tr_loader = _build(train_ds, shuffle=True)
    va_loader = _build(val_ds,   shuffle=False)

    from sklearn.metrics import f1_score as sk_f1

    best_val_f1 = -1.0
    patience_ctr = 0
    history: List[Dict] = []

    csv_fields = [
        "epoch", "train_loss", "val_f1",
        "base_loss_mean", "cost_weight_mean", "exec_prob_mean", "latency_disc_mean",
    ]
    os.makedirs(os.path.dirname(log_csv) if os.path.dirname(log_csv) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint) if os.path.dirname(checkpoint) else ".", exist_ok=True)

    with open(log_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()

        epoch_bar = tqdm(range(1, max_epochs + 1), desc="  Epochs", unit="ep",
                         dynamic_ncols=True)
        for epoch in epoch_bar:
            net.train(); loss_fn.train()
            total_loss   = 0.0
            n_batches    = 0
            breakdown_acc = {k: 0.0 for k in ["base_loss_mean","cost_weight_mean","exec_prob_mean","latency_disc_mean"]}

            batch_bar = tqdm(tr_loader, desc=f"  Ep {epoch:3d}/{max_epochs}",
                             leave=False, unit="b", dynamic_ncols=True)
            for xb, yb, sb in batch_bar:
                xb, yb, sb = xb.to(dev), yb.to(dev), sb.to(dev)
                optimizer.zero_grad()
                logits = net(xb)
                loss, _diag = loss_fn(logits, yb, sb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
                for k in breakdown_acc:
                    breakdown_acc[k] += _diag.get(k, 0.0)
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss  = total_loss / max(n_batches, 1)
            for k in breakdown_acc:
                breakdown_acc[k] /= max(n_batches, 1)

            # ── Validation F1 ────────────────────────────────────────────
            net.eval(); loss_fn.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb, _ in va_loader:
                    xb = xb.to(dev)
                    p  = net(xb).argmax(dim=1).cpu().numpy()
                    y_pred.extend(p.tolist())
                    y_true.extend(yb.numpy().tolist())
            val_f1 = float(sk_f1(y_true, y_pred, average="macro", zero_division=0))
            scheduler.step(val_f1)

            row = {
                "epoch"      : epoch,
                "train_loss" : round(avg_loss, 6),
                "val_f1"     : round(val_f1, 6),
                **{k: round(v, 6) for k, v in breakdown_acc.items()},
            }
            history.append(row)
            writer.writerow(row)
            tqdm.write(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"L_EXEC={avg_loss:.4f} | val_F1={val_f1:.4f} | "
                f"p_exec={breakdown_acc['exec_prob_mean']:.3f}"
            )
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                val_F1=f"{val_f1:.4f}",
                best=f"{best_val_f1:.4f}",
                pat=f"{patience_ctr}/{patience}",
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_ctr = 0
                torch.save(net.state_dict(), checkpoint)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    tqdm.write(f"  Early stopping at epoch {epoch}.")
                    break

    net.load_state_dict(torch.load(checkpoint, map_location=dev))
    print(f"[Module 5] L_EXEC training complete. Best val F1 = {best_val_f1:.4f}")
    return net, history


# ---------------------------------------------------------------------------
# 2. Lambda grid search
# ---------------------------------------------------------------------------

def lambda_grid_search(
    train_ds    : LOBDataset,
    val_ds      : LOBDataset,
    mean_spread : float,
    queue_model : QueueModel,
    lambdas     : List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    fast_epochs : int = 20,
    out_fig     : str = "images/fig_module5_lambda_sensitivity.png",
    out_csv     : str = "logs/lambda_grid_search.csv",
) -> float:
    """
    Runs a 20-epoch fast-sweep grid search over lambda values.

    For each lambda, trains DeepLOB with L_EXEC and evaluates
    execution-weighted accuracy on the validation set.

    Parameters
    ----------
    lambdas    : list of lambda values to test.
    fast_epochs: number of epochs per sweep (default 20).

    Returns
    -------
    float : best lambda (maximises val execution-weighted accuracy).
    """
    results = []
    lam_bar = tqdm(lambdas, desc="[GridSearch] lambda", unit="lam", dynamic_ncols=True)
    for lam in lam_bar:
        lam_bar.set_postfix(lam=lam)
        tqdm.write(f"\n[GridSearch] lambda = {lam}")
        _set_seed(RANDOM_SEED)

        loss_fn = LExecLoss(
            spread_mean=mean_spread,
            lambda_=lam,
        )
        net, _ = train_deeplob_with_lexec(
            train_ds, val_ds, loss_fn,
            checkpoint  = f"checkpoints/deeplob_lam{lam}.pt",
            log_csv     = f"logs/deeplob_lam{lam}.csv",
            max_epochs  = fast_epochs,
            patience    = fast_epochs,  # disable early stopping for fair comparison
        )

        # Wrap into a DeepLOBModel-like adapter so we can call _run_sim
        class _NetAdapter:
            def __init__(self, net_, ds_):
                self.net_ = net_
                self.ds_  = ds_
                self.name = f"DeepLOB_lam{lam}"
                self._seq_len = 10

            def predict(self, dataset):
                from src.module2_baselines import DeepLOBModel as _DL
                tmp = _DL(seq_len=10)
                tmp.net = self.net_
                return tmp.predict(dataset)

        adapter = _NetAdapter(net, val_ds)
        # Compute EWA on val set
        qm2 = QueueModel(); qm2.fit(train_ds)
        val_preds = adapter.predict(val_ds)
        epe2 = ExecutionProbabilityEstimator(k=10)
        epe2.fit(train_ds, adapter.predict(train_ds))
        sim = PaperTradingSimulator(queue_model=qm2, exec_estimator=epe2, k=10)
        sim.run(val_ds, val_preds)
        m   = compute_execution_metrics(sim)

        results.append({"lambda": lam, "ewa": m["execution_weighted_accuracy"]})
        print(f"  lambda={lam}  val EWA = {m['execution_weighted_accuracy']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    best_row = df.loc[df["ewa"].idxmax()]
    best_lam = float(best_row["lambda"])
    print(f"\n[GridSearch] Best lambda = {best_lam}  (val EWA = {best_row['ewa']:.4f})")

    # plot
    os.makedirs(os.path.dirname(out_fig) if os.path.dirname(out_fig) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["lambda"], df["ewa"], "bo-", markersize=6)
    ax.axvline(best_lam, color="red", linestyle="--", label=f"Best λ={best_lam}")
    ax.set_xscale("log"); ax.set_xlabel("λ (log scale)"); ax.set_ylabel("Val EWA")
    ax.set_title("Lambda Sensitivity: Execution-Weighted Accuracy"); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[Module 5] Lambda sensitivity curve saved -> {out_fig}")
    return best_lam


# ---------------------------------------------------------------------------
# 3. Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    train_ds    : LOBDataset,
    val_ds      : LOBDataset,
    test_ds     : LOBDataset,
    mean_spread : float,
    queue_model : QueueModel,
    best_lambda : float = 0.1,
    fast_epochs : int = 30,
) -> pd.DataFrame:
    """
    Trains four DeepLOB+L_EXEC variants and evaluates each through the
    PaperTradingSimulator:

      a. Full L_EXEC
      b. No asymmetric cost    (C = identity)
      c. No exec probability   (P_exec = 1 always)
      d. No latency penalty    (lambda = 0)

    Returns
    -------
    pd.DataFrame with ablation results.
    """
    variants = {
        "Full_LEXEC"     : {"C_identity": False, "P_exec_1": False, "no_latency": False},
        "No_CostMatrix"  : {"C_identity": True,  "P_exec_1": False, "no_latency": False},
        "No_ExecProb"    : {"C_identity": False, "P_exec_1": True,  "no_latency": False},
        "No_Latency"     : {"C_identity": False, "P_exec_1": False, "no_latency": True },
    }

    rows = []
    variant_bar = tqdm(variants.items(), desc="[Ablation]", unit="variant",
                       total=len(variants), dynamic_ncols=True)
    for variant_name, flags in variant_bar:
        variant_bar.set_postfix(variant=variant_name)
        tqdm.write(f"\n[Ablation] Variant: {variant_name}")
        _set_seed(RANDOM_SEED)

        lam    = 0.0 if flags["no_latency"] else best_lambda
        loss_fn = LExecLoss(spread_mean=mean_spread, lambda_=lam)

        # Override cost matrix to identity if requested
        if flags["C_identity"]:
            with torch.no_grad():
                loss_fn.cost_matrix.data.fill_(0.0)

        net, _ = train_deeplob_with_lexec(
            train_ds, val_ds, loss_fn,
            checkpoint  = f"checkpoints/ablation_{variant_name}.pt",
            log_csv     = f"logs/ablation_{variant_name}.csv",
            max_epochs  = fast_epochs,
            patience    = fast_epochs,
        )

        # Build adapter
        from src.module2_baselines import DeepLOBModel as _DL

        class _AblAdapter:
            def __init__(self_, net_, name_):
                self_.net_ = net_
                self_.name = name_

            def predict(self_, dataset):
                tmp = _DL(seq_len=10)
                tmp.net = self_.net_

                # P_exec = 1 override: we patch the simulator post-init
                return tmp.predict(dataset)

        adapter = _AblAdapter(net, variant_name)

        # For P_exec=1, override exec_estimator to always return 1.0
        qm2 = QueueModel(); qm2.fit(train_ds)
        train_preds = adapter.predict(train_ds)
        epe2 = ExecutionProbabilityEstimator(k=10)
        epe2.fit(train_ds, train_preds)

        if flags["P_exec_1"]:
            # Monkey-patch so P(exec) = 1 always
            epe2.predict_proba = lambda *a, **kw: 1.0

        sim = PaperTradingSimulator(queue_model=qm2, exec_estimator=epe2, k=10)
        test_preds = adapter.predict(test_ds)
        sim.run(test_ds, test_preds)

        exec_m = compute_execution_metrics(sim)
        _lbl = test_ds.labels.numpy()
        if len(test_preds) < len(_lbl):
            _lbl = _lbl[-len(test_preds):]
        cls_m  = compute_classification_metrics(_lbl, test_preds)
        row    = {"variant": variant_name, **cls_m, **exec_m}
        rows.append(row)
        print(
            f"  Sharpe={exec_m['annualized_sharpe']:.4f}  "
            f"EWA={exec_m['execution_weighted_accuracy']:.4f}  "
            f"F1={cls_m['F1_macro']:.4f}"
        )

    return pd.DataFrame(rows).set_index("variant")


# ---------------------------------------------------------------------------
# 4. Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    all_models    : list,
    test_ds       : LOBDataset,
    train_ds      : LOBDataset,
    out_csv       : str = "logs/master_results.csv",
) -> pd.DataFrame:
    """
    Runs every model through the PaperTradingSimulator and collects
    all metrics into a master DataFrame.

    Parameters
    ----------
    all_models : list of (name_str, model_instance) tuples
    test_ds, train_ds : LOBDataset
    out_csv : output path

    Returns
    -------
    pd.DataFrame indexed by model name.
    """
    rows = []
    pnl_dir = pathlib.Path("logs/pnl_curves")
    pnl_dir.mkdir(parents=True, exist_ok=True)
    eval_bar = tqdm(all_models, desc="[Evaluation]", unit="model",
                    total=len(all_models), dynamic_ncols=True)
    for name, model in eval_bar:
        eval_bar.set_postfix(model=name)
        tqdm.write(f"\n[EvalPipeline] {name}")
        metrics, sim = _run_sim(model, test_ds, train_ds)
        rows.append({"model": name, **metrics})
        # Save PnL curve so generate_figures can load it without re-running sims
        safe_name = name.replace("+", "_").replace(" ", "_")
        np.save(str(pnl_dir / f"{safe_name}.npy"), np.array(sim.pnl_curve))

    df = pd.DataFrame(rows).set_index("model")
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv)
    print(f"\n[Module 5] Master results saved -> {out_csv}")
    return df


# ---------------------------------------------------------------------------
# 5. Publication-ready figures
# ---------------------------------------------------------------------------

def generate_figures(
    master_df  : pd.DataFrame,
    ablation_df: pd.DataFrame,
    lexec_history: List[Dict],
    simulators : Dict[str, PaperTradingSimulator],
    out_dir    : str = "figures",
) -> None:
    """
    Generates Figures 1–4 in publication quality (300 DPI, tight layout).

      Figure 1 : PnL curves for all models.
      Figure 2 : F1 vs Execution-Weighted Accuracy bar chart.
      Figure 3 : Ablation heatmap.
      Figure 4 : L_EXEC training loss decomposition.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: PnL curves ──────────────────────────────────────────────
    # Prefer live simulator objects; fall back to saved .npy files on disk.
    pnl_curves: Dict[str, np.ndarray] = {}
    for name, sim in simulators.items():
        pnl_curves[name] = np.array(sim.pnl_curve)
    if not pnl_curves:   # simulators dict was empty — load from saved files
        pnl_dir = pathlib.Path("logs/pnl_curves")
        if pnl_dir.exists():
            for fpath in sorted(pnl_dir.glob("*.npy")):
                model_name = fpath.stem.replace("_", " ").replace(" ", "+", 1)
                # Restore original display name from master_results if available
                raw = fpath.stem   # e.g. 'DeepLOB_CrossEntropy'
                display = raw.replace("DeepLOB_CrossEntropy", "DeepLOB+CrossEntropy") \
                              .replace("DeepLOB_L_EXEC", "DeepLOB+L_EXEC") \
                              .replace("_", " ")
                pnl_curves[display] = np.load(str(fpath))

    fig1, ax = plt.subplots(figsize=(14, 5))
    tab10 = plt.cm.get_cmap("tab10", max(len(pnl_curves), 1))
    for idx, (name, curve) in enumerate(pnl_curves.items()):
        lw = 2.0 if "L_EXEC" in name or "EXEC" in name else 1.2
        ax.plot(curve, label=name, color=tab10(idx), linewidth=lw, alpha=0.9)
    ax.set_title("Figure 1: Simulated PnL Curves — All Models (Test Set)")
    ax.set_xlabel("LOB Update Index"); ax.set_ylabel("Cumulative PnL (ticks)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, linestyle="--", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_pnl_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Module 5] Figure 1 saved -> {out_dir}/fig1_pnl_curves.png")

    # ── Figure 2: Motivation gap ──────────────────────────────────────────
    models  = list(master_df.index)
    x       = np.arange(len(models))
    w       = 0.35
    f1_vals = master_df["F1_macro"].fillna(0).values
    ewa_vals= master_df["execution_weighted_accuracy"].fillna(0).values

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    b1 = ax1.bar(x, f1_vals, w, color="steelblue", alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
    ax1.set_ylim(0, 1); ax1.set_ylabel("Macro-F1"); ax1.set_title("(a) Classification F1")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in b1:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    b2 = ax2.bar(x, ewa_vals, w, color="darkorange", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
    ax2.set_ylim(0, max(ewa_vals.max() * 1.15, 0.05))
    ax2.set_ylabel("Execution-Weighted Accuracy"); ax2.set_title("(b) Execution Quality")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for b in b2:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    fig2.suptitle("Figure 2: Motivation Gap — Classification vs Execution Performance")
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_motivation_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Module 5] Figure 2 saved -> {out_dir}/fig2_motivation_gap.png")

    # ── Figure 3: Ablation heatmap ────────────────────────────────────────
    heat_cols = ["F1_macro", "execution_weighted_accuracy",
                 "annualized_sharpe", "max_drawdown"]
    heat_cols = [c for c in heat_cols if c in ablation_df.columns]
    col_headers = {
        "F1_macro":                    "F1 (macro)",
        "execution_weighted_accuracy": "EWA",
        "annualized_sharpe":           "Sharpe",
        "max_drawdown":                "Max Drawdown",
    }
    # Columns where HIGH value is BAD → reversed colormap
    inverted_cols = {"max_drawdown"}

    variants   = list(ablation_df.index)
    n_rows     = len(variants)
    n_cols_h   = len(heat_cols)

    fig3, axes = plt.subplots(
        1, n_cols_h,
        figsize=(10, 3.2),
        gridspec_kw={"wspace": 0.03},
    )

    cmap_good = plt.cm.RdYlGn
    cmap_bad  = plt.cm.RdYlGn_r

    for j, col in enumerate(heat_cols):
        ax_j = axes[j]
        vals     = ablation_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        span     = vmax - vmin if vmax != vmin else 1.0
        normed   = (vals - vmin) / span
        cmap_j   = cmap_bad if col in inverted_cols else cmap_good

        for i in range(n_rows):
            rgba = cmap_j(normed[i])
            ax_j.add_patch(plt.Rectangle([0, i], 1, 1, color=rgba, lw=0))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax_j.text(
                0.5, i + 0.5, f"{vals[i]:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="black" if lum > 0.50 else "white",
            )

        ax_j.set_xlim(0, 1)
        ax_j.set_ylim(0, n_rows)
        ax_j.invert_yaxis()

        # Column header above each panel
        header = col_headers.get(col, col)
        if col in inverted_cols:
            header += "\n(↓ lower = better)"
        ax_j.set_title(header, fontsize=9, pad=6,
                       color="crimson" if col in inverted_cols else "black",
                       fontweight="bold")

        ax_j.set_xticks([])
        ax_j.tick_params(axis="y", length=0)

        if j == 0:
            ax_j.set_yticks([i + 0.5 for i in range(n_rows)])
            ax_j.set_yticklabels(variants, fontsize=10)
        else:
            ax_j.set_yticks([])

        # Crisp border; crimson for the inverted column
        for spine in ax_j.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0 if col in inverted_cols else 0.8)
            spine.set_edgecolor("crimson" if col in inverted_cols else "#aaaaaa")

    fig3.suptitle("Figure 3 — Ablation Study: Metric Heatmap", fontsize=11, y=1.01)
    fig3.savefig(f"{out_dir}/fig3_ablation_heatmap.png",
                 dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"[Module 5] Figure 3 saved -> {out_dir}/fig3_ablation_heatmap.png")

    # ── Figure 4: L_EXEC loss decomposition ──────────────────────────────
    if lexec_history:
        epochs   = [h["epoch"]           for h in lexec_history]
        ce_comp  = [h.get("ce_mean", h.get("base_loss_mean", 0))  for h in lexec_history]
        cw_comp  = [h.get("cost_weight_mean", 0)  for h in lexec_history]
        ep_comp  = [h.get("exec_prob_mean", 0)    for h in lexec_history]
        ld_comp  = [h.get("latency_disc_mean", 0) for h in lexec_history]

        fig4, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, ce_comp, "b-o",  markersize=3, label="CE base")
        ax.plot(epochs, cw_comp, "r-s",  markersize=3, label="Cost-weight")
        ax.plot(epochs, ep_comp, "g-^",  markersize=3, label="P(exec)")
        ax.plot(epochs, ld_comp, "m-v",  markersize=3, label="Latency discount")
        ax.set_title("Figure 4: L_EXEC Loss Component Decomposition Over Training")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean component value")
        ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)
        fig4.tight_layout()
        fig4.savefig(f"{out_dir}/fig4_loss_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig4)
        print(f"[Module 5] Figure 4 saved -> {out_dir}/fig4_loss_decomposition.png")


# ---------------------------------------------------------------------------
# LExecAdapter — wraps DeepLOBNet so it can be used in _run_sim / evaluate
# ---------------------------------------------------------------------------

class LExecAdapter:
    """Adapts a raw DeepLOBNet to the BaseModel interface for evaluation."""
    def __init__(self, net, name: str = "DeepLOB+L_EXEC") -> None:
        self.net  = net
        self.name = name

    def predict(self, dataset: "LOBDataset") -> np.ndarray:
        from src.module2_baselines import DeepLOBModel as _DL
        tmp = _DL(seq_len=10)
        tmp.net    = self.net
        tmp.device = next(self.net.parameters()).device
        return tmp.predict(dataset)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _set_seed(RANDOM_SEED)

    # ── Data ──────────────────────────────────────────────────────────────
    print("[Module 5] Loading FI-2010 data...")
    paths    = download_fi2010(DATA_DIR)
    loader   = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
    train_ds, val_ds, raw_df = loader.load_and_split(paths["Train"], val_fraction=0.2)
    test_ds  = loader.load_test([paths["Test1"], paths["Test2"], paths["Test3"]])

    mean_spread = float(train_ds.features[:, 1].mean())
    print(f"  Calibrated mean_spread = {mean_spread:.6f}")

    # ── Fit shared simulation components ──────────────────────────────────
    shared_qm = QueueModel(); shared_qm.fit(train_ds)

    # ── 1. Train baseline suite ───────────────────────────────────────────
    print("\n[Module 5] Training baseline models...")
    classical = [MomentumBaseline(), OLSImbalanceModel(), RandomForestLOB()]
    if _XGB_AVAILABLE:
        classical.append(XGBoostLOB())

    for m in classical:
        m.fit(train_ds, val_ds)

    deeplob_ce = DeepLOBModel(seq_len=10, max_epochs=50, patience=5,
                               checkpoint_path="checkpoints/deeplob_ce.pt")
    train_model(deeplob_ce, train_ds, val_ds,
                log_csv="logs/deeplob_ce_log.csv",
                curve_png="images/deeplob_ce_curves.png")

    # ── 2. Lambda grid search ─────────────────────────────────────────────
    print("\n[Module 5] Running lambda grid search...")
    best_lambda = lambda_grid_search(
        train_ds, val_ds, mean_spread, shared_qm,
        lambdas     = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        fast_epochs = 20,
    )

    # ── 3. Train DeepLOB + L_EXEC with best lambda ────────────────────────
    print("\n[Module 5] Training DeepLOB + L_EXEC (full run)...")
    loss_fn_best = LExecLoss(
        spread_mean=mean_spread,
        lambda_=best_lambda,
    )

    lexec_net, lexec_history = train_deeplob_with_lexec(
        train_ds, val_ds, loss_fn_best,
        checkpoint  = "checkpoints/deeplob_lexec_best.pt",
        log_csv     = "logs/deeplob_lexec_training.csv",
        max_epochs  = 50, patience = 5,
    )

    # Adapter for DeepLOB+L_EXEC to work with _run_sim
    class LExecAdapter:
        def __init__(self, net_, name_):
            self.net_  = net_
            self.name  = name_
        def predict(self, dataset):
            from src.module2_baselines import DeepLOBModel as _DL
            tmp = _DL(seq_len=10); tmp.net = self.net_
            return tmp.predict(dataset)

    deeplob_lexec = LExecAdapter(lexec_net, "DeepLOB+L_EXEC")

    # ── 4. Full evaluation pipeline ───────────────────────────────────────
    print("\n[Module 5] Running full evaluation pipeline...")
    all_models_named = (
        [(m.name, m) for m in classical]
        + [("DeepLOB+CrossEntropy", deeplob_ce)]
        + [("DeepLOB+L_EXEC",       deeplob_lexec)]
    )

    master_df = run_full_evaluation(all_models_named, test_ds, train_ds,
                                     out_csv="logs/master_results.csv")

    # ── 5. Ablation study ─────────────────────────────────────────────────
    print("\n[Module 5] Running ablation study...")
    ablation_df = run_ablation(
        train_ds, val_ds, test_ds,
        mean_spread=mean_spread,
        queue_model=shared_qm,
        best_lambda=best_lambda,
        fast_epochs=30,
    )
    ablation_df.to_csv("logs/ablation_results.csv")
    print(ablation_df.to_string())

    # ── 6. Collect simulators for PnL curves ─────────────────────────────
    sim_dict: Dict[str, PaperTradingSimulator] = {}
    qm_fig = QueueModel(); qm_fig.fit(train_ds)
    for name, model in all_models_named:
        preds = model.predict(test_ds)
        tp    = model.predict(train_ds)
        epe_  = ExecutionProbabilityEstimator(k=10); epe_.fit(train_ds, tp)
        sim_  = PaperTradingSimulator(queue_model=qm_fig, exec_estimator=epe_, k=10)
        sim_.run(test_ds, preds)
        sim_dict[name] = sim_

    # ── 7. Publication figures ────────────────────────────────────────────
    print("\n[Module 5] Generating publication figures...")
    generate_figures(
        master_df    = master_df,
        ablation_df  = ablation_df,
        lexec_history= lexec_history,
        simulators   = sim_dict,
        out_dir      = "images",
    )

    print("\n=== Module 5 Complete ===")
    print(master_df[["F1_macro","execution_weighted_accuracy",
                       "annualized_sharpe","total_pnl_ticks"]].to_string())


# ---------------------------------------------------------------------------
# Assumptions and edge-case notes
# ---------------------------------------------------------------------------
# ASSUMPTIONS:
#   1. All random seeds are set once at the top of each training call via
#      _set_seed().  CUDA non-determinism (e.g. cuDNN convolution algorithms)
#      may still introduce variation — set torch.backends.cudnn.deterministic=True
#      and benchmark=False in reproduce_all.py for full reproducibility.
#   2. The LExecAdapter wraps the raw DeepLOBNet in a temporary DeepLOBModel
#      shell for prediction.  This is safe because predict() only uses the
#      net weights and does not read any other state.
#   3. XGBoost supports a surrogate-based custom objective but we use the
#      standard cross-entropy objective for XGBoost in this module.
#      "XGBoost+L_EXEC" in the paper tables therefore means XGBoost trained
#      with standard loss but evaluated through the L_EXEC-informed simulator.
#
# EDGE CASES FOR MANUAL REVIEW:
#   • The lambda grid search uses fast_epochs=20 sweeps.  If the best lambda
#     is at either boundary of the search grid, extend the grid.
#   • The ablation P_exec=1 override monkey-patches the EstimatorEstimator.
#     double-check that the patch is applied before sim.run() (it is, per the
#     current code ordering).
#   • generate_figures() silently tolerates missing columns in master_df/
#     ablation_df by filling with NaN.  Verify all columns are present before
#     finalising paper tables.
#
# REAL LOB vs FI-2010 DIFFERENCES:
#   • In live trading, the optimal lambda would be re-calibrated daily as
#     market conditions (volatility regime, session time) change.  The static
#     lambda selected here is optimal for the FI-2010 test period only.
#   • Training DeepLOB on FI-2010's ~250k events is much smaller than typical
#     production LOB archives (billions of events per stock per year).
#     The model may underfit on live data and require larger architectures.
# ---------------------------------------------------------------------------

