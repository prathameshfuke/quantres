"""
regen_figures.py
────────────────
Regenerate all publication figures WITHOUT re-training any model.

Steps:
  1. Load master_df / ablation_df / lexec_history from saved CSVs.
  2. Load FI-2010 data (already on disk).
  3. Fit lightweight classical models (fast – no GPU).
  4. Load DeepLOB CE and L_EXEC from their .pt checkpoints.
  5. Run PaperTradingSimulator for every model.
  6. Save pnl_curves to logs/pnl_curves/*.npy.
  7. Call generate_figures() → images/fig1..fig4.
"""

import pathlib, sys, os
os.chdir(pathlib.Path(__file__).parent)      # repo root as working directory
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch

# ── 1. Load existing CSVs ──────────────────────────────────────────────────
master_df   = pd.read_csv("logs/master_results.csv",   index_col="model")
ablation_df = pd.read_csv("logs/ablation_results.csv", index_col="variant")

lex_log     = pd.read_csv("logs/deeplob_lexec_training.csv")
# Normalise column name so generate_figures finds it under either key
if "base_loss_mean" in lex_log.columns and "ce_mean" not in lex_log.columns:
    lex_log = lex_log.rename(columns={"base_loss_mean": "ce_mean"})
lexec_history = lex_log.to_dict("records")

print(f"[regen] master_df     : {master_df.shape}")
print(f"[regen] ablation_df   : {ablation_df.shape}")
print(f"[regen] lexec epochs  : {len(lexec_history)}")

# ── 2. Load FI-2010 data ───────────────────────────────────────────────────
from src.module1_data_pipeline import FI2010DataLoader, download_fi2010, DATA_DIR

print("\n[regen] Loading FI-2010 data …")
paths    = download_fi2010(DATA_DIR)
loader   = FI2010DataLoader(seq_len=10, k=10, alpha=0.002)
train_ds, val_ds, _ = loader.load_and_split(paths["Train"], val_fraction=0.2)
test_ds             = loader.load_test(
    [paths["Test1"], paths["Test2"], paths["Test3"]]
)
mean_spread = float(train_ds.features[:, 1].mean())
print(f"[regen] mean_spread = {mean_spread:.6f}")

# ── 3. Fit classical models ────────────────────────────────────────────────
from src.module2_baselines import (
    MomentumBaseline, OLSImbalanceModel, RandomForestLOB,
    DeepLOBModel, DeepLOBNet,
)
try:
    from src.module2_baselines import XGBoostLOB
    _XGB = True
except ImportError:
    _XGB = False

print("\n[regen] Fitting classical models …")
classical = [MomentumBaseline(), OLSImbalanceModel(), RandomForestLOB()]
if _XGB:
    classical.append(XGBoostLOB())

for m in classical:
    m.fit(train_ds, val_ds)
    print(f"  ✓ {m.name}")

# ── 4. Load DeepLOB checkpoints (no training) ─────────────────────────────
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[regen] Loading checkpoints on {dev} …")

# CE model
deeplob_ce = DeepLOBModel(seq_len=10, checkpoint_path="checkpoints/deeplob_ce.pt")
deeplob_ce.net.load_state_dict(
    torch.load("checkpoints/deeplob_ce.pt", map_location=deeplob_ce.device)
)
deeplob_ce.net.eval()
print("  ✓ DeepLOB+CrossEntropy")

# L_EXEC model
from src.module5_training import LExecAdapter
lexec_net = DeepLOBNet(seq_len=10).to(dev)
lexec_net.load_state_dict(
    torch.load("checkpoints/deeplob_lexec_best.pt", map_location=dev)
)
lexec_net.eval()
deeplob_lexec = LExecAdapter(lexec_net, "DeepLOB+L_EXEC")
print("  ✓ DeepLOB+L_EXEC")

# ── 5. Build named model list (same order / names as master_results.csv) ──
all_models_named = (
    [(m.name, m) for m in classical]
    + [("DeepLOB+CrossEntropy", deeplob_ce)]
    + [("DeepLOB+L_EXEC",       deeplob_lexec)]
)

# ── 6. Run PaperTradingSimulator for each model ────────────────────────────
from src.module3_execution_sim import (
    QueueModel, ExecutionProbabilityEstimator, PaperTradingSimulator,
)

print("\n[regen] Fitting QueueModel …")
shared_qm = QueueModel()
shared_qm.fit(train_ds)

pnl_dir = pathlib.Path("logs/pnl_curves")
pnl_dir.mkdir(parents=True, exist_ok=True)

simulators_for_fig: dict = {}
print("\n[regen] Running simulations …")
for name, model in all_models_named:
    print(f"  → {name} …", end=" ", flush=True)
    preds = model.predict(test_ds)
    tp    = model.predict(train_ds)

    epe   = ExecutionProbabilityEstimator(k=10)
    epe.fit(train_ds, tp)

    sim   = PaperTradingSimulator(
        queue_model=shared_qm, exec_estimator=epe, k=10
    )
    sim.run(test_ds, preds)
    simulators_for_fig[name] = sim

    safe  = name.replace("+", "_").replace(" ", "_")
    np.save(str(pnl_dir / f"{safe}.npy"), np.array(sim.pnl_curve))
    print(f"PnL={sim.pnl_curve[-1]:.1f}  (saved)")

print(f"\n[regen] PnL curves saved to {pnl_dir}/")

# ── 7. Generate all figures ────────────────────────────────────────────────
from src.module5_training import generate_figures

print("\n[regen] Generating figures …")
generate_figures(
    master_df     = master_df,
    ablation_df   = ablation_df,
    lexec_history = lexec_history,
    simulators    = simulators_for_fig,
    out_dir       = "images",
)

print("\n[regen] Done — check images/fig1..fig4")
