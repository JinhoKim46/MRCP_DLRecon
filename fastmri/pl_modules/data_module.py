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

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
import fastmri
import pytorch_lightning as pl
import torch
from fastmri.data import SliceDataset

class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
            self,
            data_path: Path,
            data_transform: Callable,
            batch_size: int = 1,
            num_workers: int = 4,
            target_acc: int = 6,
            is_prototype: bool = False,
    ):
        """
        Args:
            data_path:                  Path to root data directory. For example, if knee/path
                                        is the root directory with subdirectories multicoil_train and
                                        multicoil_val, you would input knee/path for data_path.
            data_transform:             A transform object for the training split.            
            batch_size:                 Batch size.
            num_workers:                Number of workers for PyTorch dataloader.
            target_acc:                 Target acceleration rate. Default is 6. 
            is_prototype:               Whether to use prototype data.
        """
        super().__init__()

        self.data_path = data_path
        self.data_transform = data_transform        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_acc = target_acc
        self.is_prototype = is_prototype

    def _create_data_loader(
            self,
            data_partition: str
    ) -> torch.utils.data.DataLoader:
        is_train = True if data_partition == "train" else False

        dataset = SliceDataset(
            root=self.data_path,
            transform=self.data_transform,
            data_partition=data_partition,
            target_acc=self.target_acc,
            is_prototype=self.is_prototype,
        )

        # Batch sampler 
        batch_sampler = fastmri.data.BatchSampler(dataset, batch_size=self.batch_size, is_train=is_train)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_sampler=batch_sampler,
            pin_memory=True,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(data_partition='test')

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )
        parser.add_argument(
            '--is_prototype',
            default=False,
            type=bool,
            help='Whether to use prototype data or not'
        )
        
        return parser
