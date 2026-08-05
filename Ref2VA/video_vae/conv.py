# SPDX-License-Identifier: Apache-2.0
# Spatial-parallel 3D convolution for the MiniMax H3 visual VAE.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import get_parallel_state, exchange_borders




class BaseConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        padding_mode_t = "constant" if padding_mode_t == "zeros" else padding_mode_t
        self.pad_mode = padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal

    def _apply_temporal_padding(self, x):
        B, C, D, H, W = x.shape
        if D > 1:
            pad_size = (
                0,
                0,
                0,
                0,
                self.padding[0] * 2 if self.causal else self.padding[0],
                0 if self.causal else self.padding[0],
            )
            return F.pad(x, pad_size, mode=self.pad_mode_t)
        else:
            if self.pad_mode_t == "constant":
                assert self.causal, "Zeros padding is only supported for causal mode"
                zeros = torch.zeros_like(x[:, :, :1, :, :]).expand(
                    -1, -1, self.kernel_size[0] - 1, -1, -1
                )
                return torch.cat([zeros, x], dim=2)
            else:
                return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x

        x = F.pad(
            x,
            (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
            mode=self.pad_mode,
        )

        x = self._apply_temporal_padding(x)
        return x

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)

        x = self._apply_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )


class SpatialParallelConv3d(BaseConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
        self.spatial_parallel = False
        self.chunk_dim = -1

    def _exchange_borders(self, x, sp_rank, sp_size):
        if self.chunk_dim == -1:
            pad = self.padding[2]
        elif self.chunk_dim == -2:
            pad = self.padding[1]
        else:
            raise ValueError(f"Invalid chunk dimension: {self.chunk_dim}")

        if pad == 0:
            return x

        local_process_group = get_parallel_state()["sp_process_group"]
        return exchange_borders(
            x,
            pad,
            self.pad_mode,
            sp_rank,
            sp_size,
            local_process_group,
            dim=self.chunk_dim,
        )

    def _apply_padding(self, x):
        if not self.spatial_parallel:
            return super()._apply_padding(x)

        state = get_parallel_state()

        x = self._exchange_borders(x, state["sp_rank"], state["sp_size"])

        if self.chunk_dim == -1:
            x = F.pad(
                x, (0, 0, self.padding[1], self.padding[1], 0, 0), mode=self.pad_mode
            )
        elif self.chunk_dim == -2:
            x = F.pad(
                x, (self.padding[2], self.padding[2], 0, 0, 0, 0), mode=self.pad_mode
            )
        else:
            raise ValueError(f"Invalid chunk dimension: {self.chunk_dim}")

        x = self._apply_temporal_padding(x)
        return x
