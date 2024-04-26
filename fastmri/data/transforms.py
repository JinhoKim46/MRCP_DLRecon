"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

===============================================================================
Copyright (c) 2024 Jinho Kim (jinho.kim@fau.de)

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
===============================================================================
"""

from typing import Dict, NamedTuple, Union
import numpy as np
import torch
import fastmri.utils as utils


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

    return torch.from_numpy(data)

def create_mask(kspace):
    """

    :param kspace: shape of (coil, PE, RO)
    :return: mask => shape of (1,PE,1,1)
    """
    mask = kspace[0, :, 0].astype(bool)[None, :, None, None]
    return to_tensor(mask)


def undersample(kspace, rate):
    '''
    :param kspace: kspace to be undersampled
    :param rate: Rate of undersample
    :return:
        kspace_us: Undersampled kspace
    '''

    was_torch = False
    if torch.is_tensor(kspace):
        kspace = torch.view_as_complex(kspace)
        kspace = kspace.detach().cpu().numpy()
        was_torch = True

    acs_start, acs_end = _get_acs_index(kspace)
    acs = kspace[:, acs_start:acs_end + 1]

    kspace_us = _undersample(kspace, rate)
    kspace_us[:, acs_start:acs_end + 1] = acs

    if was_torch:
        kspace_us = to_tensor(kspace_us)

    return kspace_us


def _undersample(kspace, rate):
    idx = np.where(kspace[0, :, 0].astype(bool))[0][::rate]

    kspace_us = np.zeros_like(kspace)
    kspace_us[:, idx] = kspace[:, idx]

    return kspace_us


def _get_acs_index(kspace):
    mask = kspace[0, :, 0].astype(bool)
    slices = np.ma.clump_masked(np.ma.masked_where(mask, mask))
    acs_ind = [(s.start, s.stop - 1) for s in slices if s.start < (s.stop - 1)]
    assert (acs_ind != [] or len(
        acs_ind) > 1), "Couldn't extract center lines mask from k-space - is there pat2 undersampling?"
    acs_start = acs_ind[0][0]
    acs_end = acs_ind[0][1]

    return acs_start, acs_end

def _mrcp_crop(image):
    _, h, _ = image.shape
    return image[..., h // 5 * 1:h // 5 * 4, :]

def get_maxval(img: torch.Tensor) -> torch.Tensor:
    '''

    Args:
        img: size of (batch, row, col)

    Returns:
        maxval: maxval
    '''
    img = _mrcp_crop(img.unsqueeze(0))
    return torch.max(img) - torch.min(img)

def center_crop(images):
    """
    Center crop images.
    
    Args:
        images: list of recon_images

    Returns:
        output: list of cropped images
    """
    output = [_mrcp_crop(img) for img in images]
    
    return output


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        input:      Undersampled k-space.
        target:     The target image.
        target_k:   The ground truth k-space.
        sens_maps:  Sensitivity maps.
        fname:      File name.
        slice_num:  The slice index.
        max_value:  Maximum image value.
        mask:       The applied sampling mask.
        rate:       The rate of undersampling for input.
    """

    input: torch.Tensor
    target: torch.Tensor
    target_k: torch.Tensor
    sens_maps: torch.Tensor
    fname: str
    slice_num: list
    max_value: torch.Tensor
    mask: torch.Tensor
    rate: int

class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """
    def __call__(
            self,
            kspace: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
            sens_maps: Union[torch.Tensor, bool],
            dinfo: list
    ) -> VarNetSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, PE, RO, slice) for
                multi-coil data.
            target: Target image of shape (y, x, slice).
            attrs: Acquisition related information.
            fname: File name.
            slice_num: Serial number of the slice.
            sens_map: sensitivity map of shape (num_coils, y, x, slice).
            dinfo: Data info to store SENSE initializations. (filename, foldername)

        Returns:
            A VarNetSample.
        """
        GT_kspace_torch = to_tensor(kspace)

        kspace_us = undersample(kspace, rate=attrs["acceleration"])
        
        train_torch = utils.sense(kspace_us, sens_maps, dinfo)
        mask = create_mask(kspace_us)

        train_torch = to_tensor(train_torch)
        target = torch.tensor(np.abs(target))
        max_value = get_maxval(target)
        
        sample = VarNetSample(
            input=train_torch,
            target=target,
            target_k=GT_kspace_torch,
            fname=fname,
            slice_num=slice_num,
            mask=mask,
            max_value=max_value,
            sens_maps=to_tensor(sens_maps),
            rate=attrs['target_acc']
        )

        return sample
