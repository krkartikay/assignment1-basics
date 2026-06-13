import torch
from torch import nn
from torch import Tensor
from jaxtyping import Int, Float


class EmbeddingLayer(nn.Module):
    """
    Handwritten implementation of Embedding Layer.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        device_: torch.device = device or torch.device("cpu")
        self.W = torch.empty((vocab_size, dim), device=device_, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... dim"]:
        # One way to do this would be to apply one-hot encoding
        # and then matrix multiply. But it would be inefficient.
        # Instead we directly pick the relevant indices from the
        # embedding 'dictionary'.
        return self.W[token_ids]
