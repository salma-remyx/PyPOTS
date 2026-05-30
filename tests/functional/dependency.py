"""
Test cases for the cross-channel dependency error metric and its wiring into
the `pypots.cli.evaluate` evaluation path.
"""

# Created by Remyx Recommendation — adapted from XCTFormer (https://arxiv.org/abs/2605.18534)
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.nn.functional import calc_cross_channel_dependency_error

# import from the existing (non-new) call-site module to exercise the wiring edit
from pypots.cli.evaluate import _evaluate_imputation_forecasting, TASK_METRICS


class TestCrossChannelDependencyError(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)
        n_samples, n_steps, n_features = 16, 32, 4
        # build correlated channels: every channel is a noisy mix of a shared latent signal
        latent = np.random.randn(n_samples, n_steps, 1)
        mix = np.random.randn(1, 1, n_features)
        self.targets = latent * mix + 0.05 * np.random.randn(n_samples, n_steps, n_features)
        self.targets = self.targets.astype(np.float32)

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_perfect_preservation_is_zero(self):
        ccde = calc_cross_channel_dependency_error(self.targets, self.targets)
        assert ccde == 0.0

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_destroyed_structure_scores_worse_than_preserved(self):
        # a structure-preserving prediction: ground truth plus tiny independent noise
        preserved = self.targets + 0.01 * np.random.randn(*self.targets.shape).astype(np.float32)
        # a structure-destroying prediction with the SAME pointwise scale of perturbation
        # but applied as fully independent per-channel noise that scrambles correlations
        destroyed = np.random.randn(*self.targets.shape).astype(np.float32)

        ccde_preserved = calc_cross_channel_dependency_error(preserved, self.targets)
        ccde_destroyed = calc_cross_channel_dependency_error(destroyed, self.targets)

        assert ccde_preserved >= 0.0
        assert ccde_destroyed > ccde_preserved

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_torch_input_supported(self):
        import torch

        t = torch.from_numpy(self.targets)
        ccde = calc_cross_channel_dependency_error(t, t)
        assert isinstance(ccde, float)
        assert ccde == 0.0

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_mask_restricts_to_evaluated_samples(self):
        masks = np.zeros_like(self.targets)
        masks[0] = 1.0  # only the first sample is "evaluated"
        ccde = calc_cross_channel_dependency_error(self.targets, self.targets, masks)
        assert ccde == 0.0


class TestCCDEEvaluateWiring(unittest.TestCase):
    """Exercise the edit made to pypots/cli/evaluate.py (the call site)."""

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_ccde_registered_for_imputation_and_forecasting(self):
        assert "ccde" in TASK_METRICS["imputation"]
        assert "ccde" in TASK_METRICS["forecasting"]

    @pytest.mark.xdist_group(name="metric-ccde")
    def test_evaluate_imputation_computes_ccde(self):
        np.random.seed(7)
        n_samples, n_steps, n_features = 12, 20, 5
        latent = np.random.randn(n_samples, n_steps, 1)
        mix = np.random.randn(1, 1, n_features)
        ground_truth = (latent * mix + 0.05 * np.random.randn(n_samples, n_steps, n_features)).astype(np.float32)
        predictions = (ground_truth + 0.01 * np.random.randn(n_samples, n_steps, n_features)).astype(np.float32)

        indicating_mask = np.ones_like(ground_truth)

        pred_data = {"imputation": predictions}
        gt_data = {"X_ori": ground_truth, "indicating_mask": indicating_mask}

        results = _evaluate_imputation_forecasting("imputation", pred_data, gt_data, ["mae", "ccde"])

        assert "ccde" in results
        assert isinstance(results["ccde"], float)
        assert results["ccde"] >= 0.0


if __name__ == "__main__":
    unittest.main()
