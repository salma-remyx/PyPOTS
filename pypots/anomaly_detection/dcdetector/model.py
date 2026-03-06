"""
The implementation of DCdetector for the partially-observed time-series anomaly detection task.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...data.dataset.base import BaseDataset
from ...imputation.saits.data import DatasetForSAITS
from ...nn.functional import autocast
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from .core import _DCdetector, _kl_loss, _normalize_prior


class DCdetector(BaseNNDetector):
    """The PyTorch implementation of the DCdetector model :cite:`yang2023dcdetector`
    for the anomaly detection task.

    DCdetector learns dual attention representations (patch-wise and in-patch)
    via a minimax contrastive objective.  Anomaly scores are derived from the
    KL divergence between the two attention views: time steps where the two
    views disagree most are flagged as anomalous.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.
        Must be divisible by every value in ``patch_sizes``.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within the range (0, 1).
        Used for thresholding.

    patch_sizes : list of int
        Patch sizes for multi-scale patching (e.g. ``[3, 5, 7]``).
        Each value must divide ``n_steps`` evenly.

    d_model : int
        Dimension of the model embeddings.

    n_heads : int
        Number of attention heads.

    e_layers : int
        Number of encoder layers.

    dropout : float, optional
        Dropout rate. Default is 0.

    batch_size : int, optional
        Number of samples per training batch. Default is 32.

    epochs : int, optional
        Maximum number of training epochs. Default is 100.

    patience : int or None, optional
        Early-stopping patience. Disabled if None. Default is None.

    training_loss : Criterion or type, optional
        Loss criterion. Used for its ``lower_better`` attribute; the actual
        training loss is the DCdetector contrastive loss. Defaults to MAE.

    validation_metric : Criterion or type, optional
        Validation metric. Same remark as ``training_loss``. Defaults to MSE.

    optimizer : Optimizer or type, optional
        Optimizer for training. Defaults to Adam.

    num_workers : int, optional
        Number of DataLoader worker processes. Default is 0.

    device : str, torch.device, or list, optional
        Device(s) for model training and inference.

    saving_path : str, optional
        Directory to save model checkpoints. No saving if None.

    model_saving_strategy : str or None, optional
        Checkpoint saving strategy: one of ``{None, "best", "better", "all"}``.

    verbose : bool, optional
        Whether to print training progress. Default is True.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        patch_sizes: list,
        d_model: int,
        n_heads: int,
        e_layers: int,
        dropout: float = 0,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            anomaly_rate=anomaly_rate,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        # Validate that n_steps is divisible by each patch size
        for ps in patch_sizes:
            assert n_steps % ps == 0, (
                f"n_steps ({n_steps}) must be divisible by each patch_size, "
                f"but {ps} does not divide {n_steps} evenly."
            )

        self.n_steps = n_steps
        self.n_features = n_features
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout

        self.model = _DCdetector(
            n_steps=n_steps,
            n_features=n_features,
            patch_sizes=patch_sizes,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )

        self._send_model_to_given_device()
        self._print_model_size()

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """Prepare a training batch."""
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        """Prepare a validation batch (same as training)."""
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Prepare an inference batch."""
        indices, X, missing_mask = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
        }

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the model.

        Parameters
        ----------
        train_set : dict or str
            Training dataset.

        val_set : dict or str, optional
            Validation dataset. Must contain ``"X_ori"``.

        file_type : str, optional
            File type for lazy-loading. Default is ``"hdf5"``.

        """
        self.train_set = train_set

        train_dataset = DatasetForSAITS(
            train_set, return_X_ori=False, return_y=False, file_type=file_type
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = DatasetForSAITS(
                val_set, return_X_ori=True, return_y=False, file_type=file_type
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        """Detect anomalies in the test set.

        Anomaly scores are computed as the per-time-step softmax of the
        combined KL divergence between the patch-wise and in-patch attention
        maps, scaled by a temperature of 50 (as in the original paper).
        The threshold is determined from the training set distribution using
        ``anomaly_rate``.

        Parameters
        ----------
        test_set : dict or str
            Test dataset.

        file_type : str, optional
            File type for lazy-loading. Default is ``"hdf5"``.

        Returns
        -------
        dict
            Contains key ``"anomaly_detection"`` with a 1-D binary array of
            length ``n_test_samples * n_steps``.

        """
        self.model.eval()
        temperature = 50

        def _build_dataloader(dataset_arg):
            ds = BaseDataset(
                dataset_arg,
                return_X_ori=False,
                return_X_pred=False,
                return_y=False,
                file_type=file_type,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        def _score_dataloader(dataloader):
            """Return per-time-step anomaly scores, shape [N, L]."""
            score_collector = []
            for data in dataloader:
                inputs = self._assemble_input_for_testing(data)
                with autocast(enabled=self.amp_enabled):
                    results = self.model(inputs)

                series = results["series"]
                prior = results["prior"]

                series_loss = None
                prior_loss = None

                for u in range(len(prior)):
                    prior_norm = _normalize_prior(prior[u], self.n_steps)

                    kl_sp = _kl_loss(series[u], prior_norm.detach()) * temperature
                    kl_ps = _kl_loss(prior_norm, series[u].detach()) * temperature

                    if series_loss is None:
                        series_loss = kl_sp
                        prior_loss = kl_ps
                    else:
                        series_loss = series_loss + kl_sp
                        prior_loss = prior_loss + kl_ps

                # metric: [B, L] — softmax over the time-step dimension
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                score_collector.append(metric.detach().cpu().numpy())

            return np.concatenate(score_collector, axis=0)  # [N, L]

        train_scores = _score_dataloader(_build_dataloader(self.train_set)).reshape(-1)
        test_scores = _score_dataloader(_build_dataloader(test_set)).reshape(-1)

        combined = np.concatenate([train_scores, test_scores], axis=0)
        threshold = np.percentile(combined, 100 - self.anomaly_rate * 100)
        logger.info(f"Threshold: {threshold}")

        anomaly_pred = (test_scores > threshold).astype(int)

        return {"anomaly_detection": anomaly_pred}
