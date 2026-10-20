"""Main CCTD network used by CCTD.py.

Only the architecture used in the primary experiment is retained here.  The
CTN2 and CTN3 ablation/variant architectures from the original project are
intentionally excluded.
"""

import math

import numpy as np
import torch
from torch import nn


class CTN(nn.Module):
    """Coupled Tucker decomposition with sinusoidal implicit neural factors."""

    def __init__(self, hsi_shape, guide_shape, tv_shape, rank, omega=2, mid_channel=300, device="cuda"):
        super().__init__()
        r1, r2, _ = hsi_shape
        r3 = rank
        self.centre = nn.Parameter(torch.Tensor(r1, r2, r3).to(device))

        self.U_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).to(device))
        self.U_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).to(device))
        self.U_net3 = nn.Parameter(torch.Tensor(r1, mid_channel).to(device))

        self.V_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).to(device))
        self.V_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).to(device))
        self.V_net3 = nn.Parameter(torch.Tensor(r2, mid_channel).to(device))

        self.W_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).to(device))
        self.W_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).to(device))
        self.W_net3 = nn.Parameter(torch.Tensor(r3, mid_channel).to(device))

        self.P_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).to(device))
        self.P_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).to(device))
        self.P_net3 = nn.Parameter(torch.Tensor(r3, mid_channel).to(device))

        self.omega = omega
        self.init_params()
        self.set_coordinate(hsi_shape, guide_shape, tv_shape)

    def init_params(self):
        stdv = 1 / math.sqrt(self.centre.size(0))
        self.centre.data.uniform_(-stdv, stdv)

        std = 5
        for parameter in (
            self.U_net1, self.U_net2, self.U_net3,
            self.V_net1, self.V_net2, self.V_net3,
            self.W_net1, self.W_net2, self.W_net3,
            self.P_net1, self.P_net2, self.P_net3,
        ):
            torch.nn.init.kaiming_normal_(parameter, a=math.sqrt(std))

    def set_coordinate(self, hsi_shape, guide_shape, tv_shape):
        """Create normalized coordinate vectors for HSI and guide reconstruction."""
        n_1, n_2, n_3 = hsi_shape
        U_input = torch.tensor(np.array(range(1, n_1 + 1))).reshape(1, n_1).to(self.centre)
        V_input = torch.tensor(np.array(range(1, n_2 + 1))).reshape(1, n_2).to(self.centre)
        W_input = torch.tensor(np.array(range(1, n_3 + 1))).reshape(1, n_3).to(self.centre)
        self.coor_hsi = [U_input, V_input, W_input]

        g_1, g_2, g_3 = guide_shape
        ratio1, ratio2, ratio3 = g_1 / n_1, g_2 / n_2, g_3 / n_3
        U_input = torch.tensor(np.array(range(1, g_1 + 1))).reshape(1, g_1).to(self.centre) / ratio1
        V_input = torch.tensor(np.array(range(1, g_2 + 1))).reshape(1, g_2).to(self.centre) / ratio2
        W_input = torch.tensor(np.array(range(1, g_3 + 1))).reshape(1, g_3).to(self.centre) / ratio3
        self.coor_guide = [U_input, V_input, W_input]

    def recon_hsi(self, coordinate):
        U_input, V_input, W_input = coordinate
        U = self.U_net3 @ torch.sin(self.omega * self.U_net2 @ torch.sin(self.omega * self.U_net1 @ U_input))
        V = self.V_net3 @ torch.sin(self.omega * self.V_net2 @ torch.sin(self.omega * self.V_net1 @ V_input))
        W = self.W_net3 @ torch.sin(self.omega * self.W_net2 @ torch.sin(self.omega * self.W_net1 @ W_input))
        output = self.centre.permute(1, 2, 0) @ U
        output = output.permute(1, 2, 0) @ V
        output = output.permute(1, 2, 0) @ W
        return output, [U, V, W]

    def recon_guide(self, coordinate):
        U_input, V_input, P_input = coordinate
        U = self.U_net3 @ torch.sin(self.omega * self.U_net2 @ torch.sin(self.omega * self.U_net1 @ U_input))
        V = self.V_net3 @ torch.sin(self.omega * self.V_net2 @ torch.sin(self.omega * self.V_net1 @ V_input))
        P = self.P_net3 @ torch.sin(self.omega * self.P_net2 @ torch.sin(self.omega * self.P_net1 @ P_input))
        output = self.centre.permute(1, 2, 0) @ U
        output = output.permute(1, 2, 0) @ V
        output = output.permute(1, 2, 0) @ P
        return output, [U, V, P]

    def forward(self):
        out_HSI, _ = self.recon_hsi(self.coor_hsi)
        out_Guide, _ = self.recon_guide(self.coor_guide)
        return out_HSI, out_Guide
