"""
===============================================================================
Copyright (c) 2024 Jinho Kim (jinho.kim@fau.de)

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
===============================================================================
"""
from fastmri.data.mri_data import SliceDataset
from torch.utils.data import Sampler
from random import shuffle
from collections import defaultdict


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
        self.dataset = [dataset[i] for i in range(len(dataset))]
        self.batch_size = batch_size
        self.is_train = is_train
        self.batches = self.get_batches()
        
    def __iter__(self):        
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def get_batches(self):
        # Create a dictionary to group indices by their shapes
        shape_groups = defaultdict(list)
        for index, data in enumerate(self.dataset):
            shape_groups[data.input.shape].append(index)

        # If in training mode, shuffle the groups for randomness
        grouped_indices = list(shape_groups.values())
        if self.is_train:            
            for group in grouped_indices:
                shuffle(group)
            shuffle(grouped_indices)

        # Generate batches
        batches = []
        for group in grouped_indices:
            for i in range(0, len(group), self.batch_size):
                batches.append(group[i:i + self.batch_size])
        
        return batches