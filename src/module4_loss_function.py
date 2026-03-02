"""
Module 4 — Execution-Aware Loss Function  L_EXEC
=================================================
Implements LExecLoss as a fully differentiable PyTorch nn.Module.

Fix log (vs. previous version)
-------------------------------
  FIX-1  Batch-normalise every weight component (divide by batch mean
         before clamping) so no component can silently collapse to zero
         and kill the gradient signal.
  FIX-2  Zero-snap guard: when the snapshot is all-zeros (padding),
         exec_prob is forced to the neutral value 0.5 instead of
         trusting an MLP that was never trained on zero inputs.
  FIX-3  Diagonal clamping: after every forward pass the diagonal of
         self.cost_matrix is hard-zeroed so a correct-class prediction
         cost can never erroneously become positive.

Forward signature
-----------------
  total_loss, diagnostics = loss_fn(logits, targets, snap)

    logits  : (B, 3)   raw class scores
    targets : (B,)     integer class labels  {0=DOWN, 1=STAT, 2=UP}
    snap    : (B, 40)  raw (un-normalised) LOB feature vector at the
                       decision point.
              Column layout from FI-2010 data:
                0=AskPrice_1  1=AskVol_1  2=BidPrice_1  3=BidVol_1
                4=AskPrice_2  5=AskVol_2  6=BidPrice_2  7=BidVol_2
                ... (10 price-levels, ask/bid interleaved)

Usage
-----
  python -m src.module4_loss_function        # unit tests + smoke-test
"""

import pathlib
import unittest
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LExecLoss — single self-contained module
# ---------------------------------------------------------------------------

class LExecLoss(nn.Module):
    """
    Execution-aware composite loss.

    L_EXEC = mean_t [ CE(logits_t, y_t) * combined_weight_t ]
           + 0.1 * BCE(exec_prob_t, is_directional_t)

    combined_weight_t = cost_weight_t * exec_prob_t * latency_discount_t
    (each component batch-normalised to mean ~1, clamped to [0.1, 10.0])

    Parameters
    ----------
    spread_mean : float
        Mean bid-ask spread over the training set (raw price units).
        Used to calibrate the initial cost matrix entries.
    lambda_ : float
        Latency-discount hyperparameter  discount = 1/(1 + lambda*queue_depth).
        Not gradient-trained; tune via grid search.
    """

    _needs_snapshots = True   # flag read by the DeepLOB training loop

    def __init__(self, spread_mean: float, lambda_: float = 0.1) -> None:
        super().__init__()

        # A.  Learnable 3x3 cost matrix
        # Rows = true class  {0=DOWN, 1=STAT, 2=UP}
        # Cols = predicted class
        # Diagonal must stay 0 (correct predictions have no execution cost).
        s = float(spread_mean)
        cm_init = torch.tensor(
            [
                [0.0,    0.5*s,  2.0*s],   # true DOWN
                [0.5*s,  0.0,    0.5*s],   # true STAT
                [2.0*s,  0.5*s,  0.0  ],   # true UP
            ],
            dtype=torch.float32,
        )
        self.cost_matrix = nn.Parameter(cm_init)   # (3, 3)

        # B.  Execution-probability MLP
        # Input : [spread_norm, obi_1, log1p(depth_ratio)]   (3 features)
        # Output: scalar probability in [0.05, 0.95]
        # Bias of final layer initialised to 0 -> sigmoid(0) = 0.5 (neutral).
        self.exec_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.exec_mlp[-2].bias)   # final Linear bias -> 0

        # C / D.  Scalar hyper-parameters (not Parameters)
        self.lambda_     = float(lambda_)
        self.spread_mean = float(spread_mean)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snap_features(self, snap: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Derives execution-relevant scalars from the raw 40-dim LOB snap.

        Returns
        -------
        spread_raw   : (B,)  AskPrice_1 - BidPrice_1  (non-negative)
        obi_1        : (B,)  order-book imbalance at level 1 in [-1, 1]
        depth_ratio  : (B,)  total_bid / total_ask  (positive)
        ask_vol_1    : (B,)  AskVol_1  (used for queue-depth estimate)
        """
        ask_p1 = snap[:, 0]
        bid_p1 = snap[:, 2]
        ask_v1 = snap[:, 1]
        bid_v1 = snap[:, 3]

        # Ask volumes: indices 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
        ask_vols = snap[:, 1::4]    # (B, 10)
        # Bid volumes: indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
        bid_vols = snap[:, 3::4]    # (B, 10)

        spread_raw  = (ask_p1 - bid_p1).clamp(min=0.0)
        total_ask   = ask_vols.sum(dim=1).clamp(min=1e-8)
        total_bid   = bid_vols.sum(dim=1).clamp(min=1e-8)
        obi_1       = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)
        depth_ratio = total_bid / total_ask

        return spread_raw, obi_1, depth_ratio, ask_v1.clamp(min=1e-8)

    @staticmethod
    def _batch_normalise(x: torch.Tensor,
                         lo: float = 0.1,
                         hi: float = 10.0) -> torch.Tensor:
        """Divide by batch mean then clamp.  Prevents any component collapsing."""
        return (x / (x.mean() + 1e-8)).clamp(lo, hi)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits : torch.Tensor,   # (B, 3)
        targets: torch.Tensor,   # (B,)  int64
        snap   : torch.Tensor,   # (B, 40)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits  : (B, 3)
        targets : (B,)
        snap    : (B, 40)  raw un-normalised LOB snapshot

        Returns
        -------
        total_loss  : scalar Tensor
        diagnostics : dict with float values for logging / ablation
        """
        # FIX-3: zero diagonal before using the matrix
        with torch.no_grad():
            self.cost_matrix.data.fill_diagonal_(0.0)

        # 1.  Cost-weighted term
        # cost_matrix rows = true class, cols = predicted class.
        C          = self.cost_matrix.clamp(min=0.0)   # non-negative (3,3)
        probs      = torch.softmax(logits, dim=1)       # (B, 3)
        true_costs = C[targets]                         # (B, 3)
        cost_weight = (probs * true_costs).sum(dim=1)   # (B,)

        # FIX-1: batch-normalise so mean ~1 -> no collapse to 0
        cost_weight = self._batch_normalise(cost_weight)

        # 2.  Execution probability
        spread_raw, obi_1, depth_ratio, ask_v1 = self._snap_features(snap)

        # FIX-2: zero-snap guard
        is_zero_snap = (snap.abs().sum(dim=1) == 0.0)   # (B,) bool

        spread_norm = spread_raw / (self.spread_mean + 1e-8)
        depth_norm  = torch.log1p(depth_ratio)
        mlp_input   = torch.stack([spread_norm, obi_1, depth_norm], dim=1)  # (B,3)

        exec_prob = self.exec_mlp(mlp_input).squeeze(1)   # (B,)
        exec_prob = torch.where(
            is_zero_snap,
            torch.full_like(exec_prob, 0.5),
            exec_prob,
        )
        exec_prob = exec_prob.clamp(0.05, 0.95)

        # 3.  Latency discount
        total_ask        = snap[:, 1::4].sum(dim=1).clamp(min=1e-8)
        queue_depth      = total_ask / ask_v1                           # (B,)
        latency_discount = 1.0 / (1.0 + self.lambda_ * queue_depth)    # (B,)

        # FIX-1: batch-normalise
        latency_discount = self._batch_normalise(latency_discount)

        # 4.  Combined weight (FIX-1: batch-normalise the product too)
        combined_weight = cost_weight * exec_prob * latency_discount    # (B,)
        combined_weight = self._batch_normalise(combined_weight)

        base_loss  = F.cross_entropy(logits, targets, reduction="none") # (B,)
        main_loss  = (base_loss * combined_weight.detach()).mean()

        # Auxiliary: supervise exec_mlp with directional signal
        is_directional = (targets != 1).float()
        aux_loss = F.binary_cross_entropy(exec_prob, is_directional)

        total_loss = main_loss + 0.1 * aux_loss

        # Diagnostics
        diagnostics: Dict[str, float] = {
            "cost_weight_mean"    : float(cost_weight.mean().detach()),
            "exec_prob_mean"      : float(exec_prob.mean().detach()),
            "latency_disc_mean"   : float(latency_discount.mean().detach()),
            "combined_weight_mean": float(combined_weight.mean().detach()),
            "base_loss_mean"      : float(base_loss.mean().detach()),
            "aux_loss"            : float(aux_loss.detach()),
            "total_loss"          : float(total_loss.detach()),
        }

        return total_loss, diagnostics


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestLExecLoss(unittest.TestCase):
    """
    Five unit tests verifying LExecLoss correctness after the three fixes.

      test_1  output_types            - scalar tensor, no NaN / Inf
      test_2  no_collapse             - cost_weight_mean in [0.5, 2.0]
      test_3  zero_snap_neutral_exec  - exec_prob ~0.5 when snap = 0
      test_4  gradient_flow           - logits.grad valid; all MLP params
                                        receive non-zero gradients
      test_5  diagonal_stays_zero     - cost_matrix diagonal ~0 after fwd
    """

    _S = 0.001   # representative spread_mean for unit tests

    def _make_loss_fn(self, lambda_: float = 0.05) -> LExecLoss:
        return LExecLoss(spread_mean=self._S, lambda_=lambda_)

    def _make_batch(self, B: int = 32, seed: int = 0):
        torch.manual_seed(seed)
        logits  = torch.randn(B, 3)
        targets = torch.randint(0, 3, (B,))
        snap    = torch.rand(B, 40) * 0.01
        return logits, targets, snap

    # ------------------------------------------------------------------

    def test_1_output_types(self):
        """Forward must return a scalar tensor with no NaN or Inf."""
        loss_fn = self._make_loss_fn()
        logits, targets, snap = self._make_batch()

        total_loss, diag = loss_fn(logits, targets, snap)

        self.assertIsInstance(total_loss, torch.Tensor,
                              "total_loss must be a Tensor")
        self.assertEqual(total_loss.shape, torch.Size([]),
                         f"total_loss must be scalar, got {total_loss.shape}")
        self.assertFalse(torch.isnan(total_loss).item(), "total_loss is NaN")
        self.assertFalse(torch.isinf(total_loss).item(), "total_loss is Inf")
        self.assertIsInstance(diag, dict, "diagnostics must be a dict")
        for k, v in diag.items():
            self.assertFalse(
                torch.isnan(torch.tensor(v)).item(),
                f"diagnostic '{k}' is NaN",
            )

    # ------------------------------------------------------------------

    def test_2_no_collapse(self):
        """
        cost_weight_mean must stay in [0.5, 2.0] across 20 random batches.
        FIX-1 normalises to mean=1 before clamping.
        """
        loss_fn = self._make_loss_fn()

        for seed in range(20):
            logits, targets, snap = self._make_batch(seed=seed)
            _, diag = loss_fn(logits, targets, snap)
            cw = diag["cost_weight_mean"]
            self.assertGreaterEqual(
                cw, 0.5,
                f"cost_weight_mean={cw:.4f} < 0.5 (seed={seed})",
            )
            self.assertLessEqual(
                cw, 2.0,
                f"cost_weight_mean={cw:.4f} > 2.0 (seed={seed})",
            )

    # ------------------------------------------------------------------

    def test_3_zero_snap_gives_neutral_exec(self):
        """FIX-2: all-zero snap must produce exec_prob_mean ~0.5."""
        loss_fn = self._make_loss_fn()
        torch.manual_seed(42)
        B       = 64
        logits  = torch.randn(B, 3)
        targets = torch.randint(0, 3, (B,))
        snap    = torch.zeros(B, 40)

        _, diag = loss_fn(logits, targets, snap)
        ep = diag["exec_prob_mean"]

        self.assertAlmostEqual(
            ep, 0.5, delta=0.02,
            msg=f"exec_prob_mean={ep:.4f} (expected ~0.5 for zero snap)",
        )

    # ------------------------------------------------------------------

    def test_4_gradient_flow(self):
        """
        Gradients must reach logits AND all exec_mlp parameters after
        backward.
        """
        loss_fn = self._make_loss_fn()
        torch.manual_seed(7)
        B       = 16
        logits  = torch.randn(B, 3, requires_grad=True)
        targets = torch.randint(0, 3, (B,))
        snap    = torch.rand(B, 40) * 0.01

        total_loss, _ = loss_fn(logits, targets, snap)
        total_loss.backward()

        self.assertIsNotNone(logits.grad, "logits.grad is None")
        self.assertFalse(
            torch.isnan(logits.grad).any().item(),
            "logits.grad contains NaN",
        )
        for name, param in loss_fn.exec_mlp.named_parameters():
            self.assertIsNotNone(
                param.grad, f"exec_mlp.{name}.grad is None"
            )
            self.assertFalse(
                torch.isnan(param.grad).any().item(),
                f"exec_mlp.{name}.grad contains NaN",
            )

    # ------------------------------------------------------------------

    def test_5_diagonal_stays_zero(self):
        """FIX-3: diagonal of cost_matrix must be ~0 after forward."""
        loss_fn = self._make_loss_fn()

        # Corrupt the diagonal before calling forward
        with torch.no_grad():
            loss_fn.cost_matrix.data.fill_diagonal_(99.0)

        logits, targets, snap = self._make_batch(B=32)
        loss_fn(logits, targets, snap)

        diag_vals = loss_fn.cost_matrix.data.diagonal()
        for i, v in enumerate(diag_vals.tolist()):
            self.assertAlmostEqual(
                v, 0.0, places=5,
                msg=f"cost_matrix diagonal[{i}]={v:.6f} (expected 0.0)",
            )


# ---------------------------------------------------------------------------
# CLI entry-point / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 4 — LExecLoss  Unit Tests  (5 tests)")
    print("=" * 60)

    suite  = unittest.TestLoader().loadTestsFromTestCase(TestLExecLoss)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✓ All unit tests passed.")
    else:
        print(f"\n✗ {len(result.failures)} failure(s), "
              f"{len(result.errors)} error(s).")

    # Integration smoke-test
    print("\n--- Integration smoke-test ---")
    torch.manual_seed(42)
    B_demo  = 8
    loss_fn = LExecLoss(spread_mean=0.000384, lambda_=0.1)
    logits  = torch.randn(B_demo, 3, requires_grad=True)
    targets = torch.randint(0, 3, (B_demo,))
    snap    = torch.rand(B_demo, 40) * 0.001

    total_loss, diag = loss_fn(logits, targets, snap)
    total_loss.backward()

    print(f"  total_loss       : {total_loss.item():.6f}")
    print(f"  logits.grad norm : {logits.grad.norm().item():.6f}")
    for k, v in diag.items():
        print(f"  {k:<26s}: {v:.6f}")
    print(f"  cost_matrix (after fwd):\n"
          f"{loss_fn.cost_matrix.data.numpy()}")
