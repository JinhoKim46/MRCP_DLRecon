"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .ssdu_subsample import SSDUMaskFunc
from .subsample import MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    # To compute with FloateTensor, the data should be float32
    return torch.from_numpy(data).to(torch.float32)


def apply_ssdu_mask(
    data: np.ndarray,
    mask_func: SSDUMaskFunc,
    seed: Optional[float | Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample the Omega mask into the Theta mask and Lambda mask.

    Args:
        data:           The input k-space data. This is already undersampled data with
                        the Omega mask. The shape should be (Coil, rows, cols)
        mask_func:      A function that takes undersampled kdata and Omega mask and returns
                        the Theta mask for training and Lambda mask for loss.
        seed:           Seed for the random number generator.

    Returns:
        tuple containing:
            trn_mask:   SSDU Theta mask for training
            loss_mask:  SSDU Lambda mask for loss
    """
    us_direction = get_us_direction(data)

    if us_direction == "UD":
        data = data.transpose(0, 2, 1)

    mask_omega = data[0].astype(bool).astype(int)[None, ..., None]  # [1, rows, cols, 1]

    trn_mask, loss_mask = mask_func(data, mask_omega, seed=seed)

    if us_direction == "UD":
        trn_mask = trn_mask.transpose(0, 2, 1, 3)
        loss_mask = loss_mask.transpose(0, 2, 1, 3)

    return trn_mask.astype(np.int32), loss_mask.astype(np.int32)


def apply_mask(
    data: np.ndarray,
    mask_func: MaskFunc,
    seed: Optional[float | Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data:           The input k-space data. This should have at least 3 dimensions,
                        where dimensions -3 and -2 are the spatial dimensions.
        mask_func:      A function that takes a shape (tuple of ints) and a random
                        number seed and returns a mask.
        seed:           Seed for the random number generator.

    Returns:
        tuple containing:
            masked data:            Subsampled k-space data. (Coil, RO, PE)
            mask:                   The generated mask.
            num_low_frequencies:    The number of low-resolution frequency samples
                in the mask.
    """
    us_direction = get_us_direction(data)

    if us_direction == "UD":
        data = data.transpose(0, 2, 1)

    offset = get_offset(data)

    if data.ndim == 3:
        # From (Coil, RO, PE) to (Coil, RO, PE, 1)
        data = data[..., None]

    # the mask is computed on the k-space shape without padding and with readout and complex dimensions reduced
    shape = (1, *data.shape[1:3], 1)

    mask, _ = mask_func(
        shape,
        offset=offset,
        seed=seed,
    )

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    masked_data = masked_data.squeeze()

    if us_direction == "UD":
        masked_data = masked_data.transpose(0, 2, 1)
        mask = mask.transpose(0, 2, 1, 3)

    return masked_data, mask


def get_offset(kspace):
    # take first 10 PE coefficients to determine the offset
    PE_coef = kspace[0, 0, :10]

    # find the first non-zero element
    try:
        offset = np.where(PE_coef != 0)[0][0]
    except IndexError:
        # For knee data, set the offset to 0
        offset = 0

    return offset


def get_us_direction(kspace):
    assert kspace.ndim == 3, "kspace should be 3D."
    _, _, ky = kspace.shape
    us_direction = "LR" if kspace[0, :, ky // 2].all() else "UD"

    return us_direction


def undersample_(kspace, rate):
    idx = np.where(kspace[0, :, 0].astype(bool))[0][::rate]

    kspace_us = np.zeros_like(kspace)
    kspace_us[:, idx] = kspace[:, idx]

    return kspace_us


def get_acs_index(kspace):
    mask = kspace[0, :, 0].astype(bool)
    slices = np.ma.clump_masked(np.ma.masked_where(mask, mask))
    acs_ind = [(s.start, s.stop - 1) for s in slices if s.start < (s.stop - 1)]
    assert (
        acs_ind != [] or len(acs_ind) > 1
    ), "Couldn't extract center lines mask from k-space - is there pat2 undersampling?"
    acs_start = acs_ind[0][0]
    acs_end = acs_ind[0][1]

    return acs_start, acs_end


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (not x.shape[0] == mask_to.shape[0]):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def get_maxval(img: torch.Tensor) -> torch.Tensor:
    """

    Args:
        img: size of (row, col, 2)

    Returns:
        maxval: maxval
    """

    img = mrcp_crop(img.unsqueeze(0))
    return torch.max(img.abs())


def mrcp_crop(image):
    return image[..., image.shape[-2] // 5 * 1 : image.shape[-2] // 5 * 4, :]


def middle_crop(images):
    """
    Crop over-sampled images along the RO direction.

    Args:
        images: list of recon_images

    Returns:
        output: list of cropped images
    """
    output = [mrcp_crop(img) for img in images]
    return output


class BatchSample(NamedTuple):
    """
    A sample of the batch data for DL reconstruction module.

    Args:
        input:              Input k-space .
        target:             Target data (image for ssim loss, kspace for l1l2 loss)
        sens_maps:          Sensitivity maps.
        grappa_img:         ESPIRiT-based coil-combined GRAPPA reconstruction.
        kspace_raw:         Two-fold raw k-space data.
        fname:              File name.
        slice_num:          The slice index.
        gt_max_value:       Maximum value of the gt image.
        training_mask:      Mask_theta for training.
        loss_mask:          Mask_gamma for loss.
    """

    input: torch.Tensor
    target: torch.Tensor
    sens_maps: torch.Tensor
    grappa_img: torch.Tensor
    kspace_raw: torch.Tensor
    fname: str
    slice_num: list
    gt_max_value: torch.Tensor
    training_mask: torch.Tensor
    loss_mask: torch.Tensor


class DataTransform:
    """
    Data Transformer for training models.
    """

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        ssdu_mask_func: Optional[SSDUMaskFunc] = None,
        training_manner: str = "sv",
        use_seed: bool = True,
    ) -> None:
        self.mask_func = mask_func
        self.ssdu_mask_func = ssdu_mask_func
        self.training_manner = training_manner
        self.use_seed = use_seed

        self.trn_mask = {}
        self.loss_mask = {}

    def __call__(
        self,
        kspace_raw: np.ndarray,
        kspace_fs: np.ndarray,
        grappa_img: np.ndarray,
        fname: str,
        slice_num: int,
        sens_maps: Union[torch.Tensor, bool],
    ) -> BatchSample:
        """
        Args:
            kspace_raw: Two-fold raw k-space data.
            kspace_fs: Fully-sampled k-space data with GRAPPA on 2x MRCP k-space data.
            grappa_img: ESPIRiT-based coil-combined GRAPPA reconstruction.
            fname: File name.
            slice_num: Serial number of the slice.
            sens_map: sensitivity map

        Returns:
            VarNetSample
        """
        # random number generator used for augmentation
        seed = None if not self.use_seed else list(map(ord, fname))

        kspace_raw_torch = to_tensor(kspace_raw)  # Coil, row, col, 2
        grappa_img = torch.from_numpy(grappa_img)  # grappa_img
        sens_maps_torch = to_tensor(sens_maps)

        kspace_us, mask = apply_mask(kspace_raw, self.mask_func, seed=seed)

        #### NETWORK SETTING ####
        if self.training_manner == "sv":
            mask_orig = np.zeros_like(mask)
            mask_orig[0, ..., 0] = (kspace_fs[0] != 0).astype(int)

            trn_mask, loss_mask = (mask, mask_orig)

            input = to_tensor(kspace_us)
            target = to_tensor(kspace_fs)

        elif self.training_manner == "ssv":
            # For TRAINING
            if self.ssdu_mask_func is not None:
                trn_mask, loss_mask = apply_ssdu_mask(kspace_us, self.ssdu_mask_func, seed=seed)

                input = to_tensor(kspace_us) * torch.from_numpy(trn_mask)  # (coil, row, col, 2)
                target = to_tensor(kspace_us) * torch.from_numpy(loss_mask)  # (coil, row, col, 2)
            # For Testing
            else:
                # use the equi-mask with 1 padding for knee data
                trn_mask = mask
                loss_mask = mask  # Will not used

                input = to_tensor(kspace_us)
                target = to_tensor(kspace_raw)  # Will not used

        gt_max_value = get_maxval(grappa_img)
        sample = BatchSample(
            input=input,  # (coil, row, col, 2) / kspace
            target=target,  # Target image for vn and modl or kspace for ssdu
            sens_maps=sens_maps_torch,
            grappa_img=grappa_img,  # Ground truth image
            kspace_raw=kspace_raw_torch,
            fname=fname,
            slice_num=slice_num,
            training_mask=trn_mask,
            loss_mask=loss_mask,
            gt_max_value=gt_max_value,
        )

        return sample
