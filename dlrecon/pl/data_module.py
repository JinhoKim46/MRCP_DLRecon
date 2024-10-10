"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch

from dlrecon.data import MRCPDataset
from dlrecon.data.batch_sampler import ClusteredBatchSampler
from dlrecon.data.volume_sampler import VolumeSampler


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func and ssdu_mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: MRCPDataset = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))

    if data.transform.ssdu_mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.ssdu_mask_func.rng.seed(seed % (2**32 - 1))


class DataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.
    """

    def __init__(
        self,
        data_path: Path = Path(),
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        batch_size: int = 1,
        is_prototype: bool = False,
        distributed_sampler: bool = False,
        volume_sample_rate: Optional[float] = None,
        data_selection: Union[None, list, str] = None,
    ):
        """
        Args:
            data_path:                  Path to root data directory. For example, if knee/path
                                        is the root directory with subdirectories multicoil_train and
                                        multicoil_val, you would input knee/path for data_path.
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            batch_size:                 Batch size.
            is_prototype:               Whether to use prototype data.
            distributed_sampler:        Whether to use a distributed sampler.
                                        This should be set to True if training with ddp.
            volume_sample_rate:         Fraction of volumes of the training data split to use. Can be
                                        set to less than 1.0 for rapid prototyping. If not set, it defaults to 1.0.
                                        To subsample the dataset either set sample_rate (sample by slice) or
                                        volume_sample_rate (sample by volume), but not both.
            data_selection:             A list of strings that specifies the data to use.

        """
        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.is_prototype = is_prototype
        self.distributed_sampler = distributed_sampler
        self.volume_sample_rate = volume_sample_rate
        self.data_selection = data_selection

    def _create_data_loader(
        self,
        data_transform,
        data_partition: str,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:

        if data_partition == "train":
            is_train = True
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
        else:
            is_train = False
            volume_sample_rate = None

        dataset = MRCPDataset(
            root=self.data_path,
            transform=data_transform,
            data_partition=data_partition,
            is_prototype=self.is_prototype,
            volume_sample_rate=volume_sample_rate,
            data_selection=self.data_selection,
        )

        sampler = None
        if self.batch_size > 1:
            # ensure that batches contain only samples of the same size
            sampler = ClusteredBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,
                distributed=self.distributed_sampler,
            )
        elif self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                # ensure that entire volumes go to the same GPU in the ddp setting
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            batch_size=(self.batch_size if not isinstance(sampler, ClusteredBatchSampler) else 1),
            sampler=sampler if not isinstance(sampler, ClusteredBatchSampler) else None,
            batch_sampler=(sampler if isinstance(sampler, ClusteredBatchSampler) else None),
            pin_memory=True,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition="test")