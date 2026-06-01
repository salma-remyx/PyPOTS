"""
Evaluation metrics for cross-channel dependency preservation in multivariate
time-series predictions (e.g. imputation / forecasting output).

Standard pointwise metrics (MAE, MSE, ...) score each value independently and
are blind to whether a reconstruction respects the relationships *between*
channels. As argued by XCTFormer (Cross-Channel and Cross-Time Transformer,
https://arxiv.org/abs/2605.18534), real-world multivariate series share an
underlying context, so their channels carry latent cross-channel dependencies
that a good model should preserve. This module turns that idea into an
evaluation signal: it compares the inter-channel correlation structure of the
predictions against that of the ground truth. A reconstruction can have low
MAE yet a high cross-channel dependency error if it flattens or scrambles the
correlations between variables.
"""

# Adapted from XCTFormer (https://arxiv.org/abs/2605.18534) — see module docstring.
# License: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import torch

from .error import _check_inputs


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _channel_correlation_matrix(series: np.ndarray, eps: float) -> np.ndarray:
    """Pearson correlation matrix across channels of a single ``(n_steps, n_features)`` series.

    Channels with (near) zero variance have undefined correlation; their entries
    are set to 0, i.e. treated as carrying no linear cross-channel dependency.
    """
    n_features = series.shape[-1]
    centered = series - series.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    cov = (centered.T @ centered) / max(series.shape[0], 1)
    denom = np.outer(std, std)
    valid = denom > eps
    corr = np.zeros((n_features, n_features), dtype=np.float64)
    corr[valid] = cov[valid] / denom[valid]
    return np.clip(corr, -1.0, 1.0)


def calc_cross_channel_dependency_error(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    """Cross-Channel Dependency Error (CCDE) between ``predictions`` and ``targets``.

    For every sample the off-diagonal entries of the channel-by-channel Pearson
    correlation matrix are computed for both the prediction and the target, and
    the metric is the mean absolute difference between the two. The value lies in
    ``[0, 2]``: 0 means the prediction reproduces the ground-truth cross-channel
    correlation structure exactly, larger values mean the inter-variable
    dependencies were distorted.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated, shaped ``(n_samples, n_steps, n_features)``
        (a single ``(n_steps, n_features)`` series is also accepted).

    targets :
        The ground-truth data, same shape as ``predictions``.

    masks :
        Optional mask, same shape as ``targets``. When given, only samples that
        contain at least one position with ``mask == 1`` (i.e. that were actually
        imputed/evaluated) contribute to the metric, mirroring the masked
        evaluation used by the pointwise error metrics in this package.

    eps :
        Numerical floor for channel standard deviations below which a channel is
        treated as constant (no linear dependency).

    Returns
    -------
    ccde :
        The cross-channel dependency error, a Python ``float``.

    Examples
    --------
    >>> import numpy as np
    >>> from pypots.nn.functional import calc_cross_channel_dependency_error
    >>> targets = np.random.randn(8, 24, 5)
    >>> ccde = calc_cross_channel_dependency_error(targets, targets)  # 0.0, perfectly preserved

    """
    lib = _check_inputs(predictions, targets, masks)
    del lib  # only used for validation here; computation runs in numpy

    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        tgt = tgt[np.newaxis, ...]
    assert pred.ndim == 3, (
        f"`predictions`/`targets` must be 2D (n_steps, n_features) or "
        f"3D (n_samples, n_steps, n_features), but got shape {pred.shape}"
    )

    n_features = pred.shape[-1]
    if n_features < 2:
        # cross-channel dependency is undefined with fewer than two channels
        return 0.0

    sample_masks = None
    if masks is not None:
        m = _to_numpy(masks)
        if m.ndim == 2:
            m = m[np.newaxis, ...]
        # a sample contributes only if it has at least one evaluated position
        sample_masks = m.reshape(m.shape[0], -1).sum(axis=1) > 0

    # off-diagonal selector (upper triangle, k=1) — diagonal is always 1
    triu = np.triu_indices(n_features, k=1)

    errors = []
    for i in range(pred.shape[0]):
        if sample_masks is not None and not sample_masks[i]:
            continue
        pred_corr = _channel_correlation_matrix(pred[i], eps)
        tgt_corr = _channel_correlation_matrix(tgt[i], eps)
        diff = np.abs(pred_corr[triu] - tgt_corr[triu])
        errors.append(float(diff.mean()))

    if len(errors) == 0:
        return 0.0
    return float(np.mean(errors))
