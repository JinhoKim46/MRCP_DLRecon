"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Optional, Sequence

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE)"""
    return torch.mean((gt - pred) ** 2)


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def psnr(
    gt: torch.Tensor, pred: torch.Tensor, maxval: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = torch.max(gt)

    mse_val = mse(gt, pred)
    return 20 * torch.log10(maxval) - 10 * torch.log10(mse_val)


def ssim(
    gt: torch.Tensor, pred: torch.Tensor, maxval: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = torch.max(gt) if maxval is None else maxval

    # expand dimensions to [B=1, C=1, H, W]
    gt = gt[None, None]
    pred = pred[None, None]

    if torch.is_complex(gt):
        gt = torch.abs(gt)
    if torch.is_complex(pred):
        pred = torch.abs(pred)

    # torchmetrics.functional.structural_similarity_index_measure returns wrong results for dtype float32 inputs on cuda
    pred = pred.to(torch.float64)
    gt = gt.to(torch.float64)

    # choose parameters to match default parameters of skimage.metrics.structural_similarity
    ssim = StructuralSimilarityIndexMeasure(
        data_range=maxval.item(), gaussian_kernel=False, kernel_size=7
    ).to(pred.device)
    return ssim(pred, gt)


class _VolumeAverageMetric(Metric):
    """
    Abstract class for distributed metrics averaged over image volumes.
    Each given value is associated to a volume. Upon computing, values
    are averaged for each volume before an average is calculated over
    these resultant values.
    """

    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state(
            "fname_hashes", default=[], dist_reduce_fx="cat"
        )  # use hash because str is not allowed
        self.add_state("slice_nums", default=[], dist_reduce_fx="cat")

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method. This method should internally call the `update_keys` method and append a value to the
        `values` state."""

    def update_keys(self, fname: str, slice_num: torch.Tensor):
        assert slice_num.ndim == 0, "`slice_num` has to be a pytorch scalar"
        self.fname_hashes.append(torch.as_tensor(hash(fname), device=self.device))
        self.slice_nums.append(torch.as_tensor(slice_num, device=self.device))

    def compute(self):
        values_dict = defaultdict(dict)
        assert len(self.values) == len(self.fname_hashes) == len(self.slice_nums)
        for value, fname_hash, slice_num in zip(
            self.values, self.fname_hashes, self.slice_nums
        ):
            values_dict[fname_hash][slice_num] = value
        values_sum = 0
        for values in values_dict.values():
            values_sum += torch.mean(torch.cat([v.view(-1) for v in values.values()]))
        if len(values_dict) == 0:
            return torch.as_tensor(values_sum)
        else:
            return torch.as_tensor(values_sum / len(values_dict))


class MSEMetric(_VolumeAverageMetric):
    def update(
        self,
        fnames: Sequence[str],
        slice_nums: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ):
        assert (
            len(fnames)
            == slice_nums.shape[0]
            == targets.shape[0]
            == predictions.shape[0]
        )
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(mse(targets[i], predictions[i]))


class NMSEMetric(_VolumeAverageMetric):
    def update(
        self,
        fnames: Sequence[str],
        slice_nums: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ):
        assert (
            len(fnames)
            == slice_nums.shape[0]
            == targets.shape[0]
            == predictions.shape[0]
        )
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(nmse(targets[i], predictions[i]))


class PSNRMetric(_VolumeAverageMetric):
    def update(
        self,
        fnames: Sequence[str],
        slice_nums: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        maxvals: Sequence[Optional[torch.Tensor]],
    ):
        assert (
            len(fnames)
            == slice_nums.shape[0]
            == targets.shape[0]
            == predictions.shape[0]
            == len(maxvals)
        )
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(psnr(targets[i], predictions[i], maxval=maxvals[i]))


class SSIMMetric(_VolumeAverageMetric):
    def update(
        self,
        fnames: Sequence[str],
        slice_nums: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        maxvals: Sequence[Optional[torch.Tensor]],
    ):
        assert (
            len(fnames)
            == slice_nums.shape[0]
            == targets.shape[0]
            == predictions.shape[0]
            == len(maxvals)
        )
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(ssim(targets[i], predictions[i], maxval=maxvals[i]))
