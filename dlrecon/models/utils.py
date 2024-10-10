"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import dlrecon


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return dlrecon.fft2c(dlrecon.complex_mul(x, sens_maps))


def sens_reduce(
    x: torch.Tensor, sens_maps: torch.Tensor, dim=1, keepdim=True
) -> torch.Tensor:
    return dlrecon.complex_mul(dlrecon.ifft2c(x), dlrecon.complex_conj(sens_maps)).sum(
        dim=dim, keepdim=keepdim
    )
