# LOB Execution-Aware Loss Function Research Pipeline

A modular, reproducible research pipeline for designing and validating
**L\_EXEC** — a custom execution-aware loss function for Limit Order Book
(LOB) mid-price direction prediction using the FI-2010 benchmark dataset.

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)

```bash
pip install -r requirements.txt
```

### GPU (optional but recommended for DeepLOB training)

PyTorch will automatically detect and use CUDA if available.  
Install the CUDA-enabled wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Data

The pipeline downloads the **FI-2010 NoAuction DecimalPrecision** benchmark
automatically on first run from the public DeepLOB GitHub mirror:

```
https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
```

Files are cached in `data/` and reused on subsequent runs.  
**No manual download is required.**  No paid API is used.

### FI-2010 Dataset Details

| Property | Value |
|---|---|
| Stocks | 5 Finnish equities (Nokia, WRT, Kesko, Sampo, Outokumpu) |
| Time horizon | 10 trading days |
| Normalisation | Decimal-precision, No-Auction |
| LOB levels | 10 bid + 10 ask |
| Raw feature columns | 144 per snapshot |
| Snapshot cadence | Each LOB update event |

Reference: Ntakaris et al., "Benchmark Dataset for Mid-Price Prediction of
Limit Order Book Data", *Journal of Forecasting*, 2018.

---

## How to Run

### Full end-to-end reproduction (all 6 modules)

```bash
python reproduce_all.py
```

Estimated wall-clock time: ~90 minutes with a GPU, ~4 hours on CPU.

### Run individual modules

```bash
python -m src.module1_data_pipeline   # data download + feature engineering
python -m src.module2_baselines       # train all baseline models
python -m src.module3_execution_sim   # baseline execution simulation
python -m src.module4_loss_function   # L_EXEC unit tests
python -m src.module5_training        # retrain with L_EXEC + ablation
python -m src.module6_validation      # statistical tests + paper export
```

---

## Project Structure

```
quantres/
├── reproduce_all.py             # End-to-end orchestration script
├── requirements.txt
├── README.md
├── lob_pipeline.py              # Original Module 1 stub (superseded)
├── data/                        # Auto-downloaded FI-2010 files
├── figures/                     # All output PNGs (300 DPI)
│   ├── fig_module1_diagnostic.png
│   ├── fig_module3_motivation_gap.png
│   ├── fig1_pnl_curves.png
│   ├── fig2_motivation_gap.png
│   ├── fig3_ablation_heatmap.png
│   ├── fig4_loss_decomposition.png
│   ├── fig_module5_lambda_sensitivity.png
│   ├── fig_module6_dm_pvalues.png
│   ├── fig_module6_regime_robustness.png
│   └── fig_module6_properness.png
├── logs/                        # CSVs: training logs, metrics, test results
├── checkpoints/                 # Saved PyTorch model weights (.pt)
├── tables/                      # LaTeX table output (.tex)
└── src/
    ├── __init__.py
    ├── module1_data_pipeline.py  # FI-2010 ingestion, feature engineering, labels
    ├── module2_baselines.py      # Momentum, LogReg, RF, XGB, DeepLOB
    ├── module3_execution_sim.py  # QueueModel, PaperTradingSimulator, EWA
    ├── module4_loss_function.py  # L_EXEC (nn.Module) + unit tests
    ├── module5_training.py       # L_EXEC retraining, lambda grid, ablation
    └── module6_validation.py     # DM test, regime analysis, LaTeX export
```

---

## Expected Outputs

After a successful `python reproduce_all.py` run you should see:

| Output | Location |
|---|---|
| Diagnostic mid-price plot | `figures/fig_module1_diagnostic.png` |
| Baseline training curves | `figures/<model>_curves.png` |
| Motivation gap figure | `figures/fig_module3_motivation_gap.png` |
| PnL curves (all models) | `figures/fig1_pnl_curves.png` |
| F1 vs EWA comparison | `figures/fig2_motivation_gap.png` |
| Ablation heatmap | `figures/fig3_ablation_heatmap.png` |
| Loss decomposition curves | `figures/fig4_loss_decomposition.png` |
| Lambda sensitivity | `figures/fig_module5_lambda_sensitivity.png` |
| DM test p-values | `figures/fig_module6_dm_pvalues.png` |
| Regime robustness | `figures/fig_module6_regime_robustness.png` |
| Properness simulation | `figures/fig_module6_properness.png` |
| Master results CSV | `logs/master_results.csv` |
| Ablation results CSV | `logs/ablation_results.csv` |
| LaTeX results table | `tables/results_table.tex` |

---

## Module Summary

| Module | File | Weeks | Milestone |
|---|---|---|---|
| 1 | `module1_data_pipeline.py` | 1–2 | Data pipeline + labels |
| 2 | `module2_baselines.py` | 2–3 | All baselines trained |
| 3 | `module3_execution_sim.py` | 3–4 | Motivation gap confirmed |
| 4 | `module4_loss_function.py` | 4–5 | L_EXEC unit tests passing |
| 5 | `module5_training.py` | 5–6 | Full ablation + figures |
| 6 | `module6_validation.py` | 7–8 | Stats validated + LaTeX |

---

## Key Design Choices

### L_EXEC Loss Function

```
L_EXEC = (1/B) Σ_t  CE(logits_t, target_t)  ×  w_t

w_t = (1 + softmax_cost_t)  ×  P(exec)_t  ×  latency_discount_t

softmax_cost_t  = Σ_i  softmax(logits_t)[i] × C[i, target_t]
P(exec)_t       = ExecutionProbabilityMLP(spread, OBI_1, QDR)
latency_disc_t  = 1 / (1 + λ × τ_t)
```

- **C** (3×3 cost matrix) is a learnable `nn.Parameter` initialised from training-set spread statistics.
- **ExecutionProbabilityMLP** is jointly trained; initialised from logistic regression coefficients.
- **λ** is a fixed hyperparameter tuned by grid search (not gradient descent).

### Label Generation

Smooth labels following Zhang et al. (2019):
```
label = UP   if (mean_{t+1..t+k} - mid_t) / mid_t >  α
      = DOWN if (mean_{t+1..t+k} - mid_t) / mid_t < -α
      = STAT otherwise
```
Default: `k=10`, `α=0.002`.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{yoursurname2026lexec,
  title   = {L\_EXEC: An Execution-Aware Loss Function for Limit Order Book Price Prediction},
  author  = {Your Name},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Code: https://github.com/your-repo}
}
```

---

## References

1. Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. IEEE Transactions on Signal Processing.
2. Ntakaris, A., et al. (2018). *Benchmark Dataset for Mid-Price Prediction of Limit Order Book Data*. Journal of Forecasting.
3. Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics.
4. Newey, W. K., & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. Econometrica.
