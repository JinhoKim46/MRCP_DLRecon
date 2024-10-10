"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn

import dlrecon.data.data_consistency as dc
from dlrecon.models import ResNet
from dlrecon.models.utils import sens_expand, sens_reduce


class MoDLNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 10,
        chans: int = 64,
        num_resblocks: int = 15,
        cg_iter: int = 10,
    ):

        super().__init__()
        self.regularizer = ResNet(in_ch=2, chans=chans, num_of_resblocks=num_resblocks)
        self.num_cascades = num_cascades
        self.cg_iter = cg_iter
        self.lam = nn.Parameter(torch.tensor([0.05]))

    def forward(self, input_k, trn_mask, sens_maps):
        input_x = sens_reduce(input_k, sens_maps)
        x = input_x
        for _ in range(self.num_cascades):
            x = self.complex_to_chan_dim(x)
            x = self.regularizer(x.float())
            x = self.chan_complex_to_last_dim(x)

            rhs = input_x + self.lam * x
            x = dc.conjgrad(rhs, sens_maps, trn_mask, self.lam, self.cg_iter)

        kspace_pred = sens_expand(x, sens_maps)
        x = x.squeeze(1)  # shape from (batch, coil, h, w, 2) to (batch, h, w, 2)
        return x, kspace_pred

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
