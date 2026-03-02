"""
module_training_loop.py
=======================
Production-grade training loop for DeepLOB that works with both
CrossEntropyLoss and LExecLoss.

Public API
----------
  check_collapse(all_preds, epoch, threshold)  -> bool
  train_one_epoch(model, loader, optimizer, loss_fn, device, is_lexec) -> dict
  validate_one_epoch(model, loader, loss_fn, device, is_lexec)          -> dict
  train_model(model, train_loader, val_loader, loss_fn, optimizer, ...)  -> dict
  print_training_report(history)
  run_experiment(data_dir, ...)                                           -> dict

Input contracts (from create_dataloaders)
------------------------------------------
  Loaders yield  (seq, label, snap)  tuples where:
    seq   : (B, 1, 100, 40)  float32
    label : (B,)             int64,  values in {0, 1, 2}
    snap  : (B, 40)          float32

  NOTE: DeepLOBNet.forward() adds the channel dim itself via x.unsqueeze(1)
  and therefore expects (B, T, F) input.  seq is squeezed to (B, 100, 40)
  before every model call.

Loss function contracts
-----------------------
  LExecLoss.forward(logits, targets, snap)  -> (loss_scalar, diagnostics_dict)
  CrossEntropyLoss.forward(logits, targets) -> loss_scalar
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.module4_loss_function import LExecLoss


# ---------------------------------------------------------------------------
# 1. Collapse detector
# ---------------------------------------------------------------------------

def check_collapse(
    all_preds: np.ndarray,
    epoch: int,
    threshold: float = 0.85,
) -> bool:
    """
    Returns True and prints an ANSI-red warning when one class accounts
    for more than *threshold* of all predictions.

    Parameters
    ----------
    all_preds : np.ndarray  shape (N,)  integer predictions in {0, 1, 2}
    epoch     : int         current epoch number (for the warning message)
    threshold : float       dominant-class fraction above which collapse is
                            declared (default 0.85)
    """
    counts    = np.bincount(all_preds, minlength=3)
    fracs     = counts / counts.sum()
    dominant  = int(fracs.argmax())
    dominant_f = float(fracs[dominant])
    class_name = {0: "DOWN", 1: "STAT", 2: "UP"}[dominant]

    if dominant_f > threshold:
        print(
            f"\033[91m"
            f"  ⚠ COLLAPSE epoch {epoch:03d}: "
            f"{dominant_f*100:.1f}% of predictions = {class_name}"
            f"\033[0m"
        )
        return True
    return False


# ---------------------------------------------------------------------------
# 2. Single-epoch train
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: str,
    is_lexec: bool,
) -> dict:
    """
    Runs one full pass over *loader* in training mode.

    Returns
    -------
    dict with at minimum:
      "train_loss"  : float  — average loss per sample
    Plus averaged LExecLoss diagnostics when is_lexec=True.
    """
    model.train()
    total_loss    = 0.0
    total_samples = 0
    diag_accum    = defaultdict(float)
    n_batches     = 0

    for seq, label, snap in loader:
        # seq shape from loader: (B, 1, T, F)
        # DeepLOBNet.forward expects (B, T, F) — squeeze channel dim
        seq   = seq.squeeze(1).to(device)     # (B, 100, 40)
        label = label.to(device)
        snap  = snap.to(device)

        optimizer.zero_grad()
        logits = model(seq)                   # (B, 3)

        if is_lexec:
            loss, diag = loss_fn(logits, label, snap)
            for k, v in diag.items():
                diag_accum[k] += v
        else:
            loss = loss_fn(logits, label)

        loss.backward()
        # Gradient clipping — critical for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if is_lexec:
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * seq.size(0)
        total_samples += seq.size(0)
        n_batches     += 1

    avg_loss = total_loss / max(total_samples, 1)
    avg_diag = {k: v / n_batches for k, v in diag_accum.items()}
    return {"train_loss": avg_loss, **avg_diag}


# ---------------------------------------------------------------------------
# 3. Single-epoch validation
# ---------------------------------------------------------------------------

def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device: str,
    is_lexec: bool,
) -> dict:
    """
    Runs one full pass over *loader* in eval mode (no gradients).

    Returns
    -------
    dict with keys:
      val_loss, val_f1, val_acc,
      pct_down, pct_stat, pct_up,
      all_preds, all_labels
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    n_samples  = 0

    with torch.no_grad():
        for seq, label, snap in loader:
            seq   = seq.squeeze(1).to(device)   # (B, 100, 40)
            label = label.to(device)
            snap  = snap.to(device)
            logits = model(seq)

            if is_lexec:
                loss, _ = loss_fn(logits, label, snap)
            else:
                loss = loss_fn(logits, label)

            preds = logits.argmax(dim=1)
            all_preds .extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            total_loss += loss.item() * seq.size(0)
            n_samples  += seq.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    counts = np.bincount(all_preds, minlength=3)
    fracs  = counts / counts.sum()

    return {
        "val_loss"   : total_loss / max(n_samples, 1),
        "val_f1"     : float(f1),
        "val_acc"    : float(acc),
        "pct_down"   : float(fracs[0] * 100),
        "pct_stat"   : float(fracs[1] * 100),
        "pct_up"     : float(fracs[2] * 100),
        "all_preds"  : all_preds,
        "all_labels" : all_labels,
    }


# ---------------------------------------------------------------------------
# 4. Main training function
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    n_epochs:        int   = 50,
    patience:        int   = 7,
    device:          str   = "cuda",
    model_save_path: str   = "best_model.pt",
    experiment_name: str   = "experiment",
) -> dict:
    """
    Full training loop with early stopping, checkpointing, and live
    per-epoch diagnostics.

    Returns
    -------
    dict — training history with keys:
      train_loss, val_loss, val_f1, val_acc,
      pct_down, pct_stat, pct_up, collapsed,
      (LExecLoss keys when applicable),
      best_val_f1, collapse_epochs, experiment_name
    """
    is_lexec = isinstance(loss_fn, LExecLoss)
    model    = model.to(device)
    if is_lexec:
        loss_fn = loss_fn.to(device)

    history         = defaultdict(list)
    best_val_f1     = -1.0
    patience_count  = 0
    collapse_epochs: List[int] = []

    # ── Header ───────────────────────────────────────────────────────────
    header = (
        f"{'Ep':>4} | {'TrLoss':>7} | {'VF1':>6} | {'VAcc':>6} | "
        f"{'CostW':>6} | {'ExecP':>6} | {'LatD':>6} | "
        f"{'DOWN%':>6} {'UP%':>5} {'STAT%':>6}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{sep}")
    print(header)
    print(f"{'-' * len(header)}")

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, is_lexec)

        # ── Validate ──────────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model, val_loader, loss_fn, device, is_lexec)

        # ── Collapse detection ────────────────────────────────────────
        collapsed = check_collapse(val_metrics["all_preds"], epoch)
        if collapsed:
            collapse_epochs.append(epoch)

        # ── Log history ───────────────────────────────────────────────
        for k, v in {**train_metrics, **val_metrics}.items():
            if k not in ("all_preds", "all_labels"):
                history[k].append(v)
        history["collapsed"].append(int(collapsed))

        # ── Console row ───────────────────────────────────────────────
        warn_flag = " ⚠" if (
            val_metrics["pct_down"] < 5 or
            val_metrics["pct_up"]   < 5 or
            val_metrics["pct_stat"] < 2
        ) else "  "

        cost_w = train_metrics.get("cost_weight_mean", float("nan"))
        exec_p = train_metrics.get("exec_prob_mean",   float("nan"))
        # LExecLoss emits "latency_disc_mean"; fall back to "latency_discount_mean"
        lat_d  = train_metrics.get(
            "latency_disc_mean",
            train_metrics.get("latency_discount_mean", float("nan")),
        )

        print(
            f"{epoch:>4} | "
            f"{train_metrics['train_loss']:>7.4f} | "
            f"{val_metrics['val_f1']:>6.4f} | "
            f"{val_metrics['val_acc']:>6.4f} | "
            f"{cost_w:>6.3f} | "
            f"{exec_p:>6.3f} | "
            f"{lat_d:>6.3f} | "
            f"{val_metrics['pct_down']:>5.1f}% "
            f"{val_metrics['pct_up']:>5.1f}% "
            f"{val_metrics['pct_stat']:>5.1f}%"
            f"{warn_flag}"
        )

        # ── Checkpoint ────────────────────────────────────────────────
        if val_metrics["val_f1"] > best_val_f1:
            best_val_f1    = val_metrics["val_f1"]
            patience_count = 0
            torch.save(
                {
                    "epoch"          : epoch,
                    "model_state"    : model.state_dict(),
                    "loss_fn_state"  : loss_fn.state_dict() if is_lexec else None,
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1"         : best_val_f1,
                },
                model_save_path,
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                print(
                    f"\n  Early stopping at epoch {epoch}. "
                    f"Best val_F1={best_val_f1:.4f}"
                )
                break

    # ── Reload best checkpoint ────────────────────────────────────────
    ckpt = torch.load(model_save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if is_lexec and ckpt["loss_fn_state"] is not None:
        loss_fn.load_state_dict(ckpt["loss_fn_state"])

    history["best_val_f1"]     = best_val_f1
    history["collapse_epochs"] = collapse_epochs
    history["experiment_name"] = experiment_name
    return dict(history)


# ---------------------------------------------------------------------------
# 5. Post-training report
# ---------------------------------------------------------------------------

def print_training_report(history: dict) -> None:
    """
    Prints a compact summary of a completed training run, including:
      • best val_F1 and the epoch it occurred
      • total epochs run
      • number of collapse events
      • cost_weight_mean health check   (LExecLoss only)
      • ASCII val_F1 sparkline
    """
    name      = history.get("experiment_name", "unknown")
    best_f1   = history.get("best_val_f1", 0.0)
    collapses = history.get("collapse_epochs", [])
    val_f1s   = history.get("val_f1", [])
    best_ep   = int(np.argmax(val_f1s)) + 1 if val_f1s else -1

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  TRAINING REPORT — {name}")
    print(f"{sep}")
    print(f"  Best val_F1       : {best_f1:.4f}  (epoch {best_ep})")
    print(f"  Total epochs run  : {len(val_f1s)}")
    print(f"  Collapse detected : {len(collapses)} epoch(s) → {collapses}")

    cost_w_vals = history.get("cost_weight_mean", [])
    if cost_w_vals:
        mean_cw = float(np.mean(cost_w_vals))
        health  = "OK ✓" if 0.5 < mean_cw < 2.0 else "CHECK ⚠"
        print(f"  cost_weight_mean  : avg={mean_cw:.4f} ({health})")

    if val_f1s:
        lo, hi    = min(val_f1s), max(val_f1s)
        rng       = hi - lo + 1e-9
        bars      = "▁▂▃▄▅▆▇█"
        sparkline = "".join(
            bars[min(7, int((v - lo) / rng * 8))] for v in val_f1s
        )
        print(f"  val_F1 trend      : {sparkline}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 6. Full pipeline runner
# ---------------------------------------------------------------------------

def run_experiment(
    data_dir: str,
    k_horizon: int  = 10,
    device:    str  = "cuda" if torch.cuda.is_available() else "cpu",
    n_epochs:  int  = 50,
    patience:  int  = 7,
    batch_size: int = 64,
    lr:        float = 1e-4,
) -> dict:
    """
    End-to-end pipeline: load data → build loaders → pre-training checks →
    train CrossEntropy model → train LExecLoss model → print comparison.

    Parameters
    ----------
    data_dir  : path to folder containing FI-2010 Train_* / Test_* files
    k_horizon : prediction horizon in LOB events (default 10)
    device    : "cuda" or "cpu"
    n_epochs  : maximum training epochs per experiment
    patience  : early-stopping patience
    batch_size: DataLoader batch size
    lr        : Adam learning rate

    Returns
    -------
    dict  {"CrossEntropy": history_ce, "LExec": history_lex}
    """
    from src.module1_data_pipeline import load_fi2010_dataset, create_dataloaders
    from src.module2_baselines import DeepLOB

    print(f"\nDevice: {device}")

    # ── Step 1: Load data ─────────────────────────────────────────────
    X_tr, y_tr, X_te, y_te = load_fi2010_dataset(data_dir, k_horizon)
    raw_X_tr, raw_X_te = X_tr.copy(), X_te.copy()

    # ── Step 2: DataLoaders ───────────────────────────────────────────
    train_loader, test_loader, spread_mean = create_dataloaders(
        X_tr, y_tr, X_te, y_te,
        raw_X_train=raw_X_tr, raw_X_test=raw_X_te,
        batch_size=batch_size,
    )

    # ── Step 3: Pre-training sanity checks ────────────────────────────
    print("\nRunning pre-training checks...")
    seq_b, lbl_b, snap_b = next(iter(train_loader))

    assert set(lbl_b.unique().tolist()).issubset({0, 1, 2}), \
        "Label values outside {0,1,2}"
    assert len(lbl_b.unique()) > 1, \
        "All labels identical in first batch — data pipeline broken"
    assert not torch.isnan(seq_b).any(), "NaN in sequences"
    assert not torch.isinf(seq_b).any(), "Inf in sequences"

    _probe = DeepLOB(num_classes=3).to(device)
    with torch.no_grad():
        _out = _probe(seq_b.squeeze(1).to(device))
    assert _out.shape == (seq_b.size(0), 3), \
        f"Model output shape wrong: {_out.shape}"
    del _probe
    print("  ✓ PRE-TRAINING CHECKS PASSED\n")

    results: dict = {}

    # ── Experiment A: CrossEntropy ────────────────────────────────────
    torch.manual_seed(42)
    model_ce = DeepLOB(num_classes=3)
    opt_ce   = torch.optim.Adam(model_ce.parameters(), lr=lr)
    loss_ce  = nn.CrossEntropyLoss()

    history_ce = train_model(
        model_ce, train_loader, test_loader,
        loss_ce, opt_ce,
        n_epochs=n_epochs, patience=patience,
        device=device,
        model_save_path="best_deeplob_ce.pt",
        experiment_name="DeepLOB + CrossEntropy",
    )
    print_training_report(history_ce)
    results["CrossEntropy"] = history_ce

    # ── Experiment B: LExecLoss ───────────────────────────────────────
    torch.manual_seed(42)
    model_lex = DeepLOB(num_classes=3)
    loss_lex  = LExecLoss(spread_mean=spread_mean, lambda_=0.1)
    opt_lex   = torch.optim.Adam(
        list(model_lex.parameters()) + list(loss_lex.parameters()),
        lr=lr,
    )

    history_lex = train_model(
        model_lex, train_loader, test_loader,
        loss_lex, opt_lex,
        n_epochs=n_epochs, patience=patience,
        device=device,
        model_save_path="best_deeplob_lexec.pt",
        experiment_name="DeepLOB + L_EXEC",
    )
    print_training_report(history_lex)
    results["LExec"] = history_lex

    # ── Side-by-side comparison ───────────────────────────────────────
    ce_f1  = results["CrossEntropy"]["best_val_f1"]
    lex_f1 = results["LExec"]["best_val_f1"]
    ce_ep  = int(np.argmax(results["CrossEntropy"]["val_f1"])) + 1
    lex_ep = int(np.argmax(results["LExec"]["val_f1"])) + 1
    ce_col = len(results["CrossEntropy"]["collapse_epochs"])
    lex_col = len(results["LExec"]["collapse_epochs"])
    lex_cw  = float(np.mean(results["LExec"].get(
        "cost_weight_mean", [float("nan")])))

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{sep}")
    print(f"  {'Metric':<22} | {'CrossEntropy':>12} | {'L_EXEC':>8}")
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*8}")
    print(f"  {'val_F1 (best)':<22} | {ce_f1:>12.4f} | {lex_f1:>8.4f}")
    print(f"  {'Epoch of best F1':<22} | {ce_ep:>12d} | {lex_ep:>8d}")
    print(f"  {'Collapse events':<22} | {ce_col:>12d} | {lex_col:>8d}")
    print(f"  {'cost_weight_mean':<22} | {'N/A':>12} | {lex_cw:>8.3f}")
    print(f"{sep}\n")

    # ── Save histories as JSON ────────────────────────────────────────
    for name, hist in results.items():
        safe = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in hist.items()
            if k not in ("all_preds", "all_labels")
        }
        fname = f"history_{name.lower().replace(' ', '_')}.json"
        with open(fname, "w") as f:
            json.dump(safe, f, indent=2)
    print("  Histories saved to history_crossentropy.json and "
          "history_lexec.json")

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_experiment(data_dir)
