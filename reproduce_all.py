"""
reproduce_all.py
================
Single-command end-to-end reproduction script for the LOB execution-aware
loss function research pipeline.

Run:
    python reproduce_all.py

All random seeds (Python, NumPy, PyTorch, CUDA) are fixed to 42.
All artefacts (data, checkpoints, logs, images, tables) are written to
sub-directories of the workspace.

Exit codes
----------
  0  — success
  1  — download failure
  2  — unexpected exception

Reference timeline (on a machine with a GPU):
  Module 1: ~1 min   (download + feature engineering)
  Module 2: ~20 min  (DeepLOB baseline training)
  Module 3: ~5 min   (simulation)
  Module 4: ~1 min   (unit tests)
  Module 5: ~60 min  (grid search + ablation + re-training)
  Module 6: ~5 min   (statistical tests + plots)
  Total    : ~90 min
"""

# ---------------------------------------------------------------------------
# Seed everything FIRST, before any other import or computation
# ---------------------------------------------------------------------------
import os
import random
import sys

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ---------------------------------------------------------------------------
import pathlib
import traceback
import unittest
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    border = "=" * 64
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def _check_imports() -> None:
    """Fails fast if required third-party packages are missing."""
    required = [
        ("torch",      "torch"),
        ("numpy",      "numpy"),
        ("pandas",     "pandas"),
        ("sklearn",    "scikit-learn"),
        ("scipy",      "scipy"),
        ("matplotlib", "matplotlib"),
        ("xgboost",    "xgboost"),
    ]
    missing = []
    for mod, pkg in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"[reproduce_all] Missing packages: {', '.join(missing)}\n"
            "Install with:  pip install -r requirements.txt"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    _check_imports()
    _banner("LOB Execution-Aware Loss Research — Full Reproduction Run")

    # ── Ensure workspace directories exist ────────────────────────────────
    for d in ["data", "images", "checkpoints", "logs", "tables"]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # MODULE 1 — Data Pipeline
    # ==================================================================
    _banner("Module 1 — Data Pipeline and LOB Feature Engineering")

    from src.module1_data_pipeline import (
        FI2010DataLoader, download_fi2010, DATA_DIR, plot_mid_price_labels
    )

    try:
        paths = download_fi2010(DATA_DIR)
    except RuntimeError as exc:
        print(f"[ERROR] Data download failed: {exc}")
        return 1

    loader   = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
    train_ds, val_ds, raw_df = loader.load_and_split(paths["Train"], val_fraction=0.2)
    test_ds  = loader.load_test([paths["Test1"], paths["Test2"], paths["Test3"]])

    loader.print_summary_statistics(train_ds, "Train")
    loader.print_summary_statistics(val_ds,   "Validation")
    loader.print_summary_statistics(test_ds,  "Test")

    plot_mid_price_labels(
        train_ds,
        num_points=2000,
        out_path="images/fig_module1_diagnostic.png",
    )

    mean_spread = float(train_ds.features[:, 1].mean())
    print(f"\n  mean_spread (calibrated) = {mean_spread:.6f}")

    # ==================================================================
    # MODULE 2 — Baseline Model Suite
    # ==================================================================
    _banner("Module 2 — Baseline Model Suite")

    from src.module2_baselines import (
        MomentumBaseline, OLSImbalanceModel, RandomForestLOB,
        DeepLOBModel, train_model, _XGB_AVAILABLE
    )
    if _XGB_AVAILABLE:
        from src.module2_baselines import XGBoostLOB

    classical_models = [
        MomentumBaseline(),
        OLSImbalanceModel(),
        RandomForestLOB(n_estimators=200),
    ]
    if _XGB_AVAILABLE:
        classical_models.append(XGBoostLOB())

    for m in classical_models:
        train_model(
            m, train_ds, val_ds,
            log_csv    = f"logs/{m.name}_log.csv",
            curve_png  = f"images/{m.name}_curves.png",
        )

    deeplob_ce = DeepLOBModel(
        seq_len=10, max_epochs=50, patience=5,
        checkpoint_path="checkpoints/deeplob_ce.pt"
    )
    train_model(
        deeplob_ce, train_ds, val_ds,
        log_csv   = "logs/deeplob_ce_log.csv",
        curve_png = "images/deeplob_ce_curves.png",
    )

    # ==================================================================
    # MODULE 3 — Execution Simulation (baseline pass)
    # ==================================================================
    _banner("Module 3 — Execution Simulation Harness (Baseline Run)")

    from src.module3_execution_sim import (
        QueueModel, ExecutionProbabilityEstimator,
        PaperTradingSimulator, compute_execution_metrics,
        run_baseline_evaluation,
    )

    baseline_results = run_baseline_evaluation(
        models    = classical_models + [deeplob_ce],
        test_ds   = test_ds,
        train_ds  = train_ds,
        out_csv   = "logs/execution_results_module3.csv",
        out_fig   = "images/fig_module3_motivation_gap.png",
    )
    print("\n" + baseline_results.to_string())

    # ==================================================================
    # MODULE 4 — L_EXEC Unit Tests
    # ==================================================================
    _banner("Module 4 — L_EXEC Loss Function (Unit Tests)")

    from src.module4_loss_function import TestLExecLoss, LExecLoss

    suite  = unittest.TestLoader().loadTestsFromTestCase(TestLExecLoss)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(f"\n[WARNING] {len(result.failures)} unit test(s) FAILED.  "
              "Review src/module4_loss_function.py before submitting.")

    # ==================================================================
    # MODULE 5 — Training with L_EXEC, Ablation, and Results
    # ==================================================================
    _banner("Module 5 — Training, Ablation, and Results")

    from src.module5_training import (
        lambda_grid_search, train_deeplob_with_lexec,
        run_ablation, run_full_evaluation, generate_figures,
        LExecAdapter,
    )
    from src.module3_execution_sim import QueueModel
    from src.module4_loss_function import LExecLoss
    from src.module3_execution_sim import ExecutionProbabilityEstimator

    shared_qm = QueueModel(); shared_qm.fit(train_ds)

    # ── Lambda grid search ────────────────────────────────────────────
    best_lambda = lambda_grid_search(
        train_ds    = train_ds,
        val_ds      = val_ds,
        mean_spread = mean_spread,
        queue_model = shared_qm,
        lambdas     = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        fast_epochs = 20,
    )

    # ── Full DeepLOB + L_EXEC training ────────────────────────────────
    loss_fn_best = LExecLoss(
        spread_mean=mean_spread,
        lambda_=best_lambda,
    )

    lexec_net, lexec_history = train_deeplob_with_lexec(
        train_ds    = train_ds,
        val_ds      = val_ds,
        loss_fn     = loss_fn_best,
        checkpoint  = "checkpoints/deeplob_lexec_best.pt",
        log_csv     = "logs/deeplob_lexec_training.csv",
        max_epochs  = 50,
        patience    = 5,
    )
    deeplob_lexec = LExecAdapter(lexec_net, "DeepLOB+L_EXEC")

    # ── Full evaluation pipeline ──────────────────────────────────────
    all_models_named = (
        [(m.name, m) for m in classical_models]
        + [("DeepLOB+CrossEntropy", deeplob_ce)]
        + [("DeepLOB+L_EXEC",       deeplob_lexec)]
    )

    master_df = run_full_evaluation(
        all_models_named,
        test_ds  = test_ds,
        train_ds = train_ds,
        out_csv  = "logs/master_results.csv",
    )

    # ── Ablation study ────────────────────────────────────────────────
    ablation_df = run_ablation(
        train_ds    = train_ds,
        val_ds      = val_ds,
        test_ds     = test_ds,
        mean_spread = mean_spread,
        queue_model = shared_qm,
        best_lambda = best_lambda,
        fast_epochs = 30,
    )
    ablation_df.to_csv("logs/ablation_results.csv")
    print(ablation_df.to_string())

    # ── PnL curves for figures ────────────────────────────────────────
    simulators_for_fig: dict = {}
    for name, model in all_models_named:
        preds_ = model.predict(test_ds)
        tp_    = model.predict(train_ds)
        epe_   = ExecutionProbabilityEstimator(k=10); epe_.fit(train_ds, tp_)
        sim_   = PaperTradingSimulator(
            queue_model=shared_qm, exec_estimator=epe_, k=10
        )
        sim_.run(test_ds, preds_)
        simulators_for_fig[name] = sim_

    generate_figures(
        master_df     = master_df,
        ablation_df   = ablation_df,
        lexec_history = lexec_history,
        simulators    = simulators_for_fig,
        out_dir      = "images",
    )

    # ==================================================================
    # MODULE 6 — Statistical Validation and Paper Export
    # ==================================================================
    _banner("Module 6 — Statistical Validation and Paper Export")

    from src.module6_validation import run_module6

    predictions = {
        name: model.predict(test_ds)
        for name, model in all_models_named
    }

    run_module6(
        master_df   = master_df,
        test_ds     = test_ds,
        train_ds    = train_ds,
        predictions = predictions,
        mean_spread = mean_spread,
        reference   = "DeepLOB+L_EXEC",
    )

    # ==================================================================
    # Final summary
    # ==================================================================
    _banner("Reproduction Complete — Output Summary")

    print("Images:")
    for f in sorted(pathlib.Path("images").glob("*.png")):
        size = f.stat().st_size / 1024
        print(f"  {f}  ({size:.0f} KB)")

    print("\nLogs:")
    for f in sorted(pathlib.Path("logs").glob("*.csv")):
        size = f.stat().st_size / 1024
        print(f"  {f}  ({size:.0f} KB)")

    print("\nCheckpoints:")
    for f in sorted(pathlib.Path("checkpoints").glob("*.pt")):
        size = f.stat().st_size / 1024
        print(f"  {f}  ({size:.0f} KB)")

    print("\nTables:")
    for f in sorted(pathlib.Path("tables").glob("*.tex")):
        print(f"  {f}")

    print("\n✓ End-to-end reproduction finished successfully.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except KeyboardInterrupt:
        print("\n[reproduce_all] Interrupted by user.")
        sys.exit(1)
    except Exception:
        print("\n[reproduce_all] Unexpected error:")
        traceback.print_exc()
        sys.exit(2)

