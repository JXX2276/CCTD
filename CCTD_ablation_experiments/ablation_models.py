"""Network variants used solely by the CCTD ablation experiments."""

import math

import numpy as np
import torch
from torch import nn


class UncoupledSpatialCTN(nn.Module):
    """CCTD variant with independent HSI and guide spatial INRs.

    This corresponds to removing the spatial-factor coupling in the article's
    loss/coupling ablation.
    """

    def __init__(self, hsi_shape, guide_shape, tv_shape, rank, omega=2, mid_channel=300, device="cuda"):
        super().__init__()
        r1, r2, _ = hsi_shape
        self.centre = nn.Parameter(torch.empty(r1, r2, rank, device=device))
        self.omega = omega

        for prefix, out_channels in (("U", r1), ("V", r2), ("W", rank), ("O", r1), ("P", r2), ("Q", rank)):
            setattr(self, f"{prefix}_net1", nn.Parameter(torch.empty(mid_channel, 1, device=device)))
            setattr(self, f"{prefix}_net2", nn.Parameter(torch.empty(mid_channel, mid_channel, device=device)))
            setattr(self, f"{prefix}_net3", nn.Parameter(torch.empty(out_channels, mid_channel, device=device)))

        self._initialize_parameters()
        self._set_coordinates(hsi_shape, guide_shape)

    def _initialize_parameters(self):
        bound = 1 / math.sqrt(self.centre.size(0))
        nn.init.uniform_(self.centre, -bound, bound)
        for name, parameter in self.named_parameters():
            if name != "centre":
                nn.init.kaiming_normal_(parameter, a=math.sqrt(5))

    def _set_coordinates(self, hsi_shape, guide_shape):
        n1, n2, n3 = hsi_shape
        device = self.centre.device
        self.coor_hsi = [
            torch.arange(1, n1 + 1, device=device, dtype=self.centre.dtype).reshape(1, n1),
            torch.arange(1, n2 + 1, device=device, dtype=self.centre.dtype).reshape(1, n2),
            torch.arange(1, n3 + 1, device=device, dtype=self.centre.dtype).reshape(1, n3),
        ]
        g1, g2, g3 = guide_shape
        self.coor_guide = [
            torch.arange(1, g1 + 1, device=device, dtype=self.centre.dtype).reshape(1, g1) / (g1 / n1),
            torch.arange(1, g2 + 1, device=device, dtype=self.centre.dtype).reshape(1, g2) / (g2 / n2),
            torch.arange(1, g3 + 1, device=device, dtype=self.centre.dtype).reshape(1, g3) / (g3 / n3),
        ]

    def _factor(self, prefix, coordinate):
        first = getattr(self, f"{prefix}_net1")
        hidden = getattr(self, f"{prefix}_net2")
        output = getattr(self, f"{prefix}_net3")
        return output @ torch.sin(self.omega * hidden @ torch.sin(self.omega * first @ coordinate))

    def _reconstruct(self, factors):
        u, v, w = factors
        output = self.centre.permute(1, 2, 0) @ u
        output = output.permute(1, 2, 0) @ v
        return output.permute(1, 2, 0) @ w

    def forward(self):
        hsi = self._reconstruct([self._factor(prefix, coordinate) for prefix, coordinate in zip("UVW", self.coor_hsi)])
        guide = self._reconstruct([self._factor(prefix, coordinate) for prefix, coordinate in zip("OPQ", self.coor_guide)])
        return hsi, guide


class SharedSpectralCTN(nn.Module):
    """CCTD variant that shares one spectral INR between HSI and guide."""

    def __init__(self, hsi_shape, guide_shape, tv_shape, rank, omega=2, mid_channel=300, device="cuda"):
        super().__init__()
        r1, r2, _ = hsi_shape
        self.centre = nn.Parameter(torch.empty(r1, r2, rank, device=device))
        self.omega = omega

        for prefix, out_channels in (("U", r1), ("V", r2), ("S", rank)):
            setattr(self, f"{prefix}_net1", nn.Parameter(torch.empty(mid_channel, 1, device=device)))
            setattr(self, f"{prefix}_net2", nn.Parameter(torch.empty(mid_channel, mid_channel, device=device)))
            setattr(self, f"{prefix}_net3", nn.Parameter(torch.empty(out_channels, mid_channel, device=device)))

        self._initialize_parameters()
        self._set_coordinates(hsi_shape, guide_shape)

    def _initialize_parameters(self):
        bound = 1 / math.sqrt(self.centre.size(0))
        nn.init.uniform_(self.centre, -bound, bound)
        for name, parameter in self.named_parameters():
            if name != "centre":
                nn.init.kaiming_normal_(parameter, a=math.sqrt(5))

    def _set_coordinates(self, hsi_shape, guide_shape):
        n1, n2, n3 = hsi_shape
        device = self.centre.device
        self.coor_hsi = [
            torch.arange(1, n1 + 1, device=device, dtype=self.centre.dtype).reshape(1, n1),
            torch.arange(1, n2 + 1, device=device, dtype=self.centre.dtype).reshape(1, n2),
            torch.arange(1, n3 + 1, device=device, dtype=self.centre.dtype).reshape(1, n3),
        ]
        g1, g2, g3 = guide_shape
        self.coor_guide = [
            torch.arange(1, g1 + 1, device=device, dtype=self.centre.dtype).reshape(1, g1) / (g1 / n1),
            torch.arange(1, g2 + 1, device=device, dtype=self.centre.dtype).reshape(1, g2) / (g2 / n2),
            torch.arange(1, g3 + 1, device=device, dtype=self.centre.dtype).reshape(1, g3) / (g3 / n3),
        ]

    def _factor(self, prefix, coordinate):
        first = getattr(self, f"{prefix}_net1")
        hidden = getattr(self, f"{prefix}_net2")
        output = getattr(self, f"{prefix}_net3")
        return output @ torch.sin(self.omega * hidden @ torch.sin(self.omega * first @ coordinate))

    def _reconstruct(self, coordinates):
        u = self._factor("U", coordinates[0])
        v = self._factor("V", coordinates[1])
        spectral = self._factor("S", coordinates[2])
        output = self.centre.permute(1, 2, 0) @ u
        output = output.permute(1, 2, 0) @ v
        return output.permute(1, 2, 0) @ spectral

    def forward(self):
        return self._reconstruct(self.coor_hsi), self._reconstruct(self.coor_guide)
