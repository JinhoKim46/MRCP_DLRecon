"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import pandas as pd
import torch


class BaseDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
        data_partition: str = "train",
        volume_sample_rate: Optional[float] = None,
        data_selection: Optional[list] = None,
    ):
        """
        Args:
            root:                       Path to the dataset.
            transform:                  Optional; A callable object that pre-processes the raw
                                        data into appropriate form. The transform function should take
                                        'kspace', 'target', 'attributes', 'filename', and 'slice' as
                                        inputs. 'target' may be null for test data.
            data_partition:             Optional; A string that specifies which data partition to use.
                                        One of 'train', 'val', or 'test'. Defaults to 'train'.
            gt_acc:                     Optional; Pseudo GT acceleration rate. Default is 2.
            is_prototype:               Optional; If True, only use the first 1 file in the dataset.
            volume_sample_rate:         Fraction of volumes of the training data split to use. Can be
                                        set to less than 1.0 for rapid prototyping. If not set, it defaults to 1.0.
                                        To subsample the dataset either set sample_rate (sample by slice) or
                                        volume_sample_rate (sample by volume), but not both.
            data_selection:             Optional; A list of strings that specifies the data to use.
        """
        self.transform = transform
        self.examples = []
        self.dataset_csv = pd.read_csv(root / "dataset.csv")

        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        files = self.get_file_list(root, data_partition, data_selection)

        for fname in sorted(files):
            num_slices = self._retrieve_shape(fname)
            self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]

        if volume_sample_rate < 1.0:
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            num_volumes = max(num_volumes, 1)  # Make sure there is at least 1 volume sample.
            sampled_vols = vol_names[:num_volumes]
            self.examples = [example for example in self.examples if example[0].stem in sampled_vols]

        # filter example items only for train and validation.
        if data_partition == "train":
            self.examples_flter(root)

    def examples_flter(self, root):
        pass

    def _retrieve_shape(self, fname):
        with h5py.File(fname, "r") as hf:
            shapes = hf["kdata_raw"].shape

        return shapes[-1]

    def get_file_list(self, root, data_partition, data_selection):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        raise NotImplementedError


class MRCPDataset(BaseDataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        is_prototype: bool = False,
        **kwargs,
    ):
        """
        Args:
            is_prototype:   Optional; If True, only use the first 1 file in the dataset.

        """
        super().__init__(**kwargs)

        # Fill the data file names to be used for prototyping.
        prototyping_data = []

        if is_prototype:
            self.examples = [f for f in self.examples if f[0].stem in prototyping_data]

    def examples_flter(self, root):
        """
        Filter the samples as you wish
        """
        pass

    def get_file_list(self, root, data_partition, data_selection):
        acc_list = ["2x"] if data_partition in ["train", "val"] else ["2x", "6x"]

        filtered_dataset = self.dataset_csv[
            (self.dataset_csv["Acc"].isin(acc_list)) & (self.dataset_csv["Split"] == data_partition)
        ]
        # If data_selection is not None, filter the dataset based on the data_selection list
        if data_selection is not None:
            filtered_dataset = self.dataset_csv[self.dataset_csv["Name"].isin(data_selection)]

        return [root / f"{fname}.h5" for fname in filtered_dataset.Name]

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kdata_raw"][..., dataslice]  # shape: (ncoil, kx, ky, slice)
            kspace_grappa = hf[f"kdata_fs"][..., dataslice]  # shape: (ncoil, kx, ky, slice)
            grappa_img = hf[f"grappa"][..., dataslice]  # shape: (x, y, slice)
            sens_maps = hf["sm_espirit"][..., dataslice]  # shape: (ncoil, x, y, slice)

        sample = self.transform(kspace, kspace_grappa, grappa_img, fname.name, dataslice, sens_maps)
        return sample
