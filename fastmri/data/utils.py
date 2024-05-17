"""
===============================================================================
Copyright (c) 2024 Jinho Kim (jinho.kim@fau.de)

Modifications and additional features by Jinho Kim are licensed under the MIT license, 
as detailed in the accompanying LICENSE file.
===============================================================================
"""
from pathlib import Path
import pandas as pd


def get_file_list(root:Path, data_partition):
    dataset = pd.read_csv(root / "dataset.csv")
    file_list = filter_data(dataset, root, data_partition)        
    
    return file_list

def filter_data(dataset, root, data_partition):
    filtered_dataset = dataset[(dataset['Split'] == data_partition)]
    return [root / f"{i}.h5"  for i in filtered_dataset['Name']] # [PATH_TO_DATASET/FILE_NAME.h5, ...]