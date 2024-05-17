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

import os
from pathlib import Path
from typing import Callable, Optional, Union
import h5py
import torch
import fastmri.data.utils as utils


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root: Union[str, Path, os.PathLike], transform: Optional[Callable] = None, data_partition: str = "train", target_acc: int = 6, is_prototype: bool = False):
        """
        Args:
            root:               Path to the dataset.
            transform:          Optional; A callable object that pre-processes the raw
                                data into appropriate form. The transform function should take
                                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                                inputs. 'target' may be null for test data.
            data_partition:     Optional; A string that specifies which data partition to use.
                                One of 'train', 'val', or 'test'. Defaults to 'train'.            
            target_acc:         Optional; Target acceleration rate. Default is 6.
            is_prototype:       Optional; If True, only use the first 1 file in the dataset.
        """
        self.transform = transform
        self.target_acc = target_acc
        self.examples = []
        
        files = utils.get_file_list(root, data_partition)

        # It is useful to use small amount of data for development and debugging.
        # if is_prototype:
        #     prototyping_data = # define the prototyping data here
        #     files = [f for f in files if f.stem in prototyping_data]
        
        for fname in sorted(files):
            num_slices = self._retrieve_metadata(fname)
            self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]

        print(f"Size: {len(files)}\n")

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["kdata"].shape[-1]

        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kdata"][..., dataslice]  # shape: (ncoil, PE, RO, slice)
            target = hf["grappa"][..., dataslice]  # shape: (y, x, slice)
            sens_maps = hf["sm_espirit"][..., dataslice]  # shape: (ncoil, y, x, slice)
            
            attrs = dict(hf.attrs)
            base_acc = attrs['base_acc'] # Either 2 or 6 (Base acceleration factor)
            target_acc = self.target_acc

        dinfo = [f"{fname.stem}_{dataslice}", fname.parent.stem] # data info: [filename, foldername] to store SENSE initializations
        sample = self.transform(kspace, target, target_acc, base_acc, fname.name, dataslice, sens_maps, dinfo)

        return sample
