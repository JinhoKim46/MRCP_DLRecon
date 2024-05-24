"""
===============================================================================
Copyright (c) 2024 Jinho Kim (jinho.kim@fau.de)

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
===============================================================================
"""

from random import shuffle
from torch.utils.data import Sampler
from fastmri.data.mri_data import SliceDataset


class BatchSampler(Sampler):
    """
    This BatchSampler samples batches of indices for MRCP data, not individual samples.
    This sampler does not discard last even though the last batch may be smaller than the specified batch size.

    """

    def __init__(
        self,
        dataset: SliceDataset,
        batch_size: int,
        is_train: bool = True,
    ):
        self.dataset = [(i[0].stem, i[1]) for i in dataset.examples]
        self.batch_size = batch_size
        self.is_train = is_train
        self.batches = self.get_batches()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def get_batches(self):
        batches, batch = [], []
        i = 0
        batch_cnt = 0
        cur_fname = None
        while i < len(self.dataset):
            if batch_cnt == self.batch_size:
                batches.append(batch)
                batch = []
                cur_fname = None
                batch_cnt = 0

            if cur_fname is None:
                cur_fname = self.dataset[i][0]
                batch.append(self.dataset[i][1])
                i += 1
                batch_cnt += 1
            else:
                if self.dataset[i][1] == cur_fname:
                    batch.append(self.dataset[i][1])
                    i += 1
                    batch_cnt += 1
                else:
                    batches.append(batch)
                    batch = []
                    cur_fname = None
                    batch_cnt = 0

        if len(batch) > 0:
            batches.append(batch)

        if self.is_train:
            shuffle(batches)

        return batches
