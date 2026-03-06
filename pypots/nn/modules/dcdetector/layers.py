"""
PyTorch modules/layers for DCdetector.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from ..revin import RevIN


class DCdetectorTokenEmbedding(nn.Module):
    """Token embedding for DCdetector using a 1D circular convolution."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> permute -> [B, C, L] -> conv -> [B, d_model, L] -> transpose -> [B, L, d_model]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DCdetectorPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for DCdetector."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class DCdetectorDataEmbedding(nn.Module):
    """Data embedding for DCdetector: token embedding + positional encoding + dropout."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.value_embedding = DCdetectorTokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = DCdetectorPositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DACStructure(nn.Module):
    """Dual Attention Contrastive (DAC) structure.

    Computes two attention views for contrastive representation learning:

    - **Patch-wise (series)**: inter-patch attention that captures long-range dependencies
      across patch positions.
    - **In-patch (prior)**: intra-patch attention that captures local dependencies
      within each patch.

    Both views are upsampled back to the full window size and averaged over the channel
    dimension to produce per-time-step attention matrices of shape ``[B, H, L, L]``.

    Parameters
    ----------
    win_size :
        The full window size (equal to n_steps).

    patch_sizes :
        List of patch sizes used for multi-scale patching.

    n_features :
        Number of input features (channels).

    mask_flag :
        Whether to apply a causal mask. Default is False.

    scale :
        Scaling factor for attention scores. If None, uses ``1/sqrt(d_k)``.

    attention_dropout :
        Dropout rate applied to attention weights.

    output_attention :
        Whether to return attention maps.

    """

    def __init__(
        self,
        win_size: int,
        patch_sizes: list,
        n_features: int,
        mask_flag: bool = False,
        scale=None,
        attention_dropout: float = 0.05,
        output_attention: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_sizes = patch_sizes
        self.n_features = n_features

    def forward(
        self,
        queries_patch_size: torch.Tensor,
        queries_patch_num: torch.Tensor,
        keys_patch_size: torch.Tensor,
        keys_patch_num: torch.Tensor,
        values: torch.Tensor,
        patch_index: int,
        attn_mask=None,
    ):
        """
        Parameters
        ----------
        queries_patch_size : torch.Tensor, shape [B*C, patch_num, H, d_k]
            Queries for the patch-wise attention.

        queries_patch_num : torch.Tensor, shape [B*C, patch_size, H, d_k]
            Queries for the in-patch attention.

        keys_patch_size : torch.Tensor, shape [B*C, patch_num, H, d_k]
            Keys for the patch-wise attention.

        keys_patch_num : torch.Tensor, shape [B*C, patch_size, H, d_k]
            Keys for the in-patch attention.

        values : torch.Tensor, shape [B, L, H, d_v]
            Values (not used directly, kept for interface consistency).

        patch_index : int
            Index into self.patch_sizes for the current scale.

        attn_mask :
            Optional attention mask (not currently applied).

        Returns
        -------
        series_patch_size : torch.Tensor, shape [B, H, L, L]
            Patch-wise attention map upsampled to the full window size.

        series_patch_num : torch.Tensor, shape [B, H, L, L]
            In-patch attention map upsampled to the full window size.

        """
        patchsize = self.patch_sizes[patch_index]
        patch_num = self.window_size // patchsize

        # ---- Patch-wise representation (inter-patch attention) ----
        # queries_patch_size: [B*C, patch_num, H, d_k]
        B, L, H, E = queries_patch_size.shape
        scale_patch_size = self.scale or 1.0 / math.sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size)
        attn_patch_size = scale_patch_size * scores_patch_size
        # series_patch_size: [B*C, H, patch_num, patch_num]
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))

        # ---- In-patch representation (intra-patch attention) ----
        # queries_patch_num: [B*C, patch_size, H, d_k]
        B, L, H, E = queries_patch_num.shape
        scale_patch_num = self.scale or 1.0 / math.sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num, keys_patch_num)
        attn_patch_num = scale_patch_num * scores_patch_num
        # series_patch_num: [B*C, H, patch_size, patch_size]
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))

        # ---- Upsample both maps to full window size [B*C, H, L, L] ----
        # Repeat each attention value patchsize times in both spatial dims
        series_patch_size = repeat(
            series_patch_size,
            "b l m n -> b l (m repeat_m) (n repeat_n)",
            repeat_m=patchsize,
            repeat_n=patchsize,
        )
        # Tile the in-patch attention patch_num times in both spatial dims
        series_patch_num = series_patch_num.repeat(1, 1, patch_num, patch_num)

        # ---- Average over the channel dimension ----
        # [B*C, H, L, L] -> [B, H, L, L]
        series_patch_size = reduce(
            series_patch_size, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.n_features
        )
        series_patch_num = reduce(
            series_patch_num, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.n_features
        )

        return series_patch_size, series_patch_num


class DCdetectorAttentionLayer(nn.Module):
    """Attention layer wrapping the :class:`DACStructure`.

    Projects input embeddings to queries/keys/values and delegates to the
    inner dual-attention module.

    Parameters
    ----------
    attention :
        The inner :class:`DACStructure` module.

    d_model :
        Model dimension.

    patch_sizes :
        List of patch sizes used for multi-scale patching.

    n_features :
        Number of input features (channels).

    n_heads :
        Number of attention heads.

    win_size :
        Full window size (equal to n_steps).

    d_keys :
        Dimension of keys/queries per head. Defaults to ``d_model // n_heads``.

    d_values :
        Dimension of values per head. Defaults to ``d_model // n_heads``.

    """

    def __init__(
        self,
        attention: DACStructure,
        d_model: int,
        patch_sizes: list,
        n_features: int,
        n_heads: int,
        win_size: int,
        d_keys: int = None,
        d_values: int = None,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_sizes = patch_sizes
        self.n_features = n_features
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(
        self,
        x_patch_size: torch.Tensor,
        x_patch_num: torch.Tensor,
        x_ori: torch.Tensor,
        patch_index: int,
        attn_mask=None,
    ):
        """
        Parameters
        ----------
        x_patch_size : torch.Tensor, shape [B*C, patch_num, d_model]
            Patch-size-based embedding of the input.

        x_patch_num : torch.Tensor, shape [B*C, patch_size, d_model]
            Patch-num-based embedding of the input.

        x_ori : torch.Tensor, shape [B, L, d_model]
            Window-level embedding of the full input.

        patch_index : int
            Index into patch_sizes for the current scale.

        attn_mask :
            Optional attention mask.

        Returns
        -------
        series : torch.Tensor, shape [B, H, L, L]
        prior : torch.Tensor, shape [B, H, L, L]

        """
        H = self.n_heads

        # ---- Patch-size branch ----
        B, L, _ = x_patch_size.shape
        queries_patch_size = self.patch_query_projection(x_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_projection(x_patch_size).view(B, L, H, -1)

        # ---- Patch-num branch ----
        B, L, _ = x_patch_num.shape
        queries_patch_num = self.patch_query_projection(x_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_projection(x_patch_num).view(B, L, H, -1)

        # ---- Values from window-level embedding ----
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)

        series, prior = self.inner_attention(
            queries_patch_size,
            queries_patch_num,
            keys_patch_size,
            keys_patch_num,
            values,
            patch_index,
            attn_mask,
        )
        return series, prior


class DCdetectorEncoder(nn.Module):
    """Stack of :class:`DCdetectorAttentionLayer` modules.

    Parameters
    ----------
    attn_layers :
        List of attention layers to stack.

    norm_layer :
        Optional normalization layer applied after all attention layers.

    """

    def __init__(self, attn_layers: list, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x_patch_size: torch.Tensor,
        x_patch_num: torch.Tensor,
        x_ori: torch.Tensor,
        patch_index: int,
        attn_mask=None,
    ):
        """
        Returns
        -------
        series_list : list of torch.Tensor
            One attention map per encoder layer, each of shape [B, H, L, L].

        prior_list : list of torch.Tensor
            One attention map per encoder layer, each of shape [B, H, L, L].

        """
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class BackboneDCdetector(nn.Module):
    """Backbone of the DCdetector model.

    Implements the multi-scale dual-attention contrastive architecture from
    :cite:`yang2023dcdetector`.  For each patch size, the input is split into
    two complementary patch views (patch-wise and in-patch), embedded, and then
    passed through a shared encoder that returns the dual attention maps.
    RevIN normalization is applied to the input before patching.

    Parameters
    ----------
    n_steps :
        The number of time steps (window size).  Must be divisible by every
        element of ``patch_sizes``.

    n_features :
        The number of input features (channels).

    patch_sizes :
        List of patch sizes for multi-scale patching (e.g. ``[3, 5, 7]``).
        Each value must divide ``n_steps`` evenly.

    d_model :
        Dimension of the model embeddings.

    n_heads :
        Number of attention heads.

    e_layers :
        Number of encoder layers.

    dropout :
        Dropout rate applied in embeddings and attention.

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
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.n_features = n_features
        self.n_steps = n_steps

        # Reversible Instance Normalization
        self.revin = RevIN(n_features)

        # Per-patch-size embeddings (two views per scale)
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for patchsize in patch_sizes:
            self.embedding_patch_size.append(
                DCdetectorDataEmbedding(patchsize, d_model, dropout)
            )
            self.embedding_patch_num.append(
                DCdetectorDataEmbedding(n_steps // patchsize, d_model, dropout)
            )

        # Window-level embedding for the value branch
        self.embedding_window_size = DCdetectorDataEmbedding(n_features, d_model, dropout)

        # Shared encoder
        self.encoder = DCdetectorEncoder(
            [
                DCdetectorAttentionLayer(
                    DACStructure(
                        n_steps,
                        patch_sizes,
                        n_features,
                        mask_flag=False,
                        attention_dropout=dropout,
                        output_attention=True,
                    ),
                    d_model,
                    patch_sizes,
                    n_features,
                    n_heads,
                    n_steps,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, L, M]
            Input time-series data (B=batch, L=n_steps, M=n_features).

        Returns
        -------
        series_list : list of torch.Tensor
            Flattened list of patch-wise attention maps (one per scale × layer).
            Each tensor has shape ``[B, H, L, L]``.

        prior_list : list of torch.Tensor
            Flattened list of in-patch attention maps (one per scale × layer).
            Each tensor has shape ``[B, H, L, L]``.

        """
        # RevIN normalization
        x = self.revin(x, mode="norm")

        # Window-level embedding (used as values in the attention layer)
        x_ori = self.embedding_window_size(x)

        series_patch_mean = []
        prior_patch_mean = []

        for patch_index, patchsize in enumerate(self.patch_sizes):
            # ---- Patch-size view: [B, L, M] -> [B, M, L] -> [B*M, L//p, p] ----
            x_patch_size = rearrange(x, "b l m -> b m l")
            x_patch_size = rearrange(x_patch_size, "b m (n p) -> (b m) n p", p=patchsize)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)

            # ---- Patch-num view: [B, L, M] -> [B, M, L] -> [B*M, p, L//p] ----
            x_patch_num = rearrange(x, "b l m -> b m l")
            x_patch_num = rearrange(x_patch_num, "b m (p n) -> (b m) p n", p=patchsize)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series)
            prior_patch_mean.append(prior)

        # Flatten the nested lists (each element was a list of e_layers tensors)
        series_list = [item for sublist in series_patch_mean for item in sublist]
        prior_list = [item for sublist in prior_patch_mean for item in sublist]

        return series_list, prior_list
