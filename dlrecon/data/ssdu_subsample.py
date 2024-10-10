"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
"""

import contextlib
from typing import Optional, Tuple, Union

import numpy as np


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class SSDUMaskFunc:
    """
    A parent class for SSDU sampling masks.
    """

    def __init__(
        self,
        center_block: Tuple[int, int] = (5, 5),
        rho: float = 0.4,
        std_scale: int = 4,
        seed: Optional[int] = None,
    ):
        self.center_block = center_block
        self.rho = rho
        self.std_scale = std_scale
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        input_data: np.ndarray,
        mask_omega: np.ndarray,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        assert input_data.ndim == 3, "input_data should have the shape of 3"

        with temp_seed(self.rng, seed):
            mask_theta, mask_lambda = self.sample_mask(input_data, mask_omega)

        trn_mask = mask_theta
        loss_mask = mask_lambda
        return trn_mask, loss_mask

    def sample_mask(self, input_data: np.ndarray, mask_omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SSDUGaussianMask1D(SSDUMaskFunc):
    """Generate vairable density Gaussian SSDU sampling mask."""

    def sample_mask(self, input_data: np.ndarray, mask_omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        nrow, ncol = input_data.shape[1], input_data.shape[2]
        center_kx = int(find_center_ind(input_data, axes=(0, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 1)))

        col_profile = mask_omega[0, nrow // 2, :, 0]  # for safe reason, take the middle profile not the first or last.
        total_count = int(len(np.where(col_profile)[0]) * self.rho)

        total_element_multiplier = 5
        while True:
            try:
                # Generate a large number of random samples
                indy = np.random.normal(
                    loc=center_ky,
                    scale=(ncol - 1) / self.std_scale,
                    size=total_count * total_element_multiplier,
                ).astype(int)

                # Clip the values to be within the valid range
                indy = np.clip(indy, 0, ncol - 1)

                # Combine and deduplicate indices
                sampled_indices = np.unique(indy)

                # Filter out indices that do not meet the conditions
                valid_mask = mask_omega[0, nrow // 2, sampled_indices, 0] == 1
                valid_indices = sampled_indices[valid_mask]

                # Select the required number of valid indices
                loss_indices = valid_indices[
                    np.random.choice(
                        valid_indices.shape[0],
                        size=total_count,
                        replace=False,
                    )
                ]
                break

            except:
                total_element_multiplier += 5

        # Create loss mask
        loss_mask = np.zeros_like(mask_omega)
        loss_mask[0, :, loss_indices, 0] = 1
        small_acs_height_start = self.center_block[0] // 2
        small_acs_height_end = self.center_block[0] - small_acs_height_start
        small_acs_width_start = self.center_block[1] // 2
        small_acs_width_end = self.center_block[1] - small_acs_width_start
        loss_mask[
            0,
            center_kx - small_acs_height_start : center_kx + small_acs_height_end,
            center_ky - small_acs_width_start : center_ky + small_acs_width_end,
            0,
        ] = 0

        # Create training mask
        trn_mask = mask_omega - loss_mask

        return trn_mask, loss_mask


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims:
        return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]
