"""
The core model of DCdetector for the anomaly detection task.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.dcdetector import BackboneDCdetector
from ...nn.modules.loss import Criterion


def _kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Asymmetric (unnormalized) KL divergence.

    Parameters
    ----------
    p : torch.Tensor, shape [B, H, L, L]
    q : torch.Tensor, shape [B, H, L, L]

    Returns
    -------
    torch.Tensor, shape [B, L]
        Mean KL divergence per batch sample and time step.

    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _normalize_prior(prior: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Normalise a prior attention map to sum to 1 along the last dimension.

    Parameters
    ----------
    prior : torch.Tensor, shape [B, H, L, L]
        Raw (unnormalized) prior attention map.

    n_steps : int
        Length of the time-step dimension (L).

    Returns
    -------
    torch.Tensor, shape [B, H, L, L]
        Row-normalized prior map.

    """
    return prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, n_steps)


class _DCdetector(ModelCore):
    """The core PyTorch model of DCdetector.

    This module wraps :class:`BackboneDCdetector` and adds the minimax
    contrastive loss used during training/validation.

    The training objective is a minimax game between two attention views:
    *series* (patch-wise, inter-patch attention) and *prior* (in-patch,
    intra-patch attention).  The loss ``prior_loss - series_loss`` is
    minimised, which encourages the two views to maximally disagree on
    anomalous patterns.

    Parameters
    ----------
    n_steps :
        Number of time steps in each input window.

    n_features :
        Number of input features.

    patch_sizes :
        List of patch sizes for multi-scale patching.

    d_model :
        Model embedding dimension.

    n_heads :
        Number of attention heads.

    e_layers :
        Number of encoder layers.

    dropout :
        Dropout rate.

    training_loss :
        Loss criterion (used only for its ``lower_better`` attribute; the
        actual training loss is the DCdetector contrastive loss).

    validation_metric :
        Validation metric (same remark as ``training_loss``).

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        patch_sizes: list,
        d_model: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.patch_sizes = patch_sizes

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneDCdetector(
            n_steps=n_steps,
            n_features=n_features,
            patch_sizes=patch_sizes,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        inputs : dict
            Must contain key ``"X"`` with shape ``[B, L, M]``.

        calc_criterion : bool
            If True, the contrastive loss is added to the returned dict as
            ``"loss"`` (training mode) or ``"metric"`` (evaluation mode).

        Returns
        -------
        dict with keys:

        - ``"series"`` – list of patch-wise attention tensors,
          each of shape ``[B, H, L, L]``.
        - ``"prior"``  – list of in-patch attention tensors,
          each of shape ``[B, H, L, L]``.
        - ``"loss"`` / ``"metric"`` – scalar contrastive loss
          (only when ``calc_criterion=True``).

        """
        X = inputs["X"]

        series, prior = self.backbone(X)

        results = {
            "series": series,
            "prior": prior,
        }

        if calc_criterion:
            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                # Normalise prior so it sums to 1 along the last dimension
                prior_norm = _normalize_prior(prior[u], self.n_steps)

                # Symmetric KL between series and normalised prior
                series_loss += torch.mean(
                    _kl_loss(series[u], prior_norm.detach())
                ) + torch.mean(_kl_loss(prior_norm.detach(), series[u]))

                # Symmetric KL in the opposite direction (minimax partner)
                prior_loss += torch.mean(
                    _kl_loss(prior_norm, series[u].detach())
                ) + torch.mean(_kl_loss(series[u].detach(), prior_norm))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            # Minimax training objective: minimise prior_loss - series_loss
            loss = prior_loss - series_loss

            if self.training:
                results["loss"] = loss
            else:
                results["metric"] = loss

        return results
