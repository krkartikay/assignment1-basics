import torch
from torch import nn
from einops import einsum


class LinearLayer(nn.Module):
    """
    Handwritten implementation of Linear Layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        device_: torch.device = device or torch.device("cpu")
        self.W = torch.empty((out_features, in_features), device=device_, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = einsum(self.W, X, "xout xin, ... xin -> ... xout")
        return output
