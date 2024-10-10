"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

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


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        num_acs: int,
        acceleration: int,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_acs: The number of auto-calibration signal (ACS) lines.
            accelerations: Amount of under-sampling.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        self.num_acs = num_acs
        self.acceleration = acceleration
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, int]:
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
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_acs = self.sample_mask(shape, offset)

        # combine masks together
        return np.maximum(center_mask, accel_mask), num_acs

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, self.num_acs), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                shape, self.acceleration, offset, self.num_acs
            ),
            shape,
        )

        return center_mask, acceleration_mask, self.num_acs

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Reshape mask to desired output shape."""
        num_rows, num_cols = shape[-3:-1]
        mask_shape = [1 for _ in shape]
        mask_shape[-3] = num_rows
        mask_shape[-2] = num_cols

        return mask.reshape(*mask_shape).astype(np.float32)

    def calculate_acceleration_mask(
        self,
        shape: Sequence[int],
        acceleration: int,
        offset: Optional[int],
        num_acs: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_acs: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_rows, num_cols = shape[-3:-1]  # take [-3] and [-2]
        mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[:, pad : pad + num_low_freqs] = 1
        assert (mask.sum() // num_rows) == num_low_freqs

        return mask


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        shape: Sequence[int],
        acceleration: int,
        offset: Optional[int],
        num_acs: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_acs: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        num_rows, num_cols = shape[-3:-1]  # take [-3] and [-2]
        mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        mask[:, offset::acceleration] = 1

        return mask


def get_mask_type(mask_type_str: str):
    mask_type_str = mask_type_str.lower()
    if mask_type_str == "equispaced":
        return EquiSpacedMaskFunc
    else:
        raise ValueError(f"{mask_type_str} not supported in dynamic mode")
