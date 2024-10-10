"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim <jinho.kim@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2, copytree
from typing import Dict, Optional, Union

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from dlrecon import complex_abs

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")


class TQDMProgressBarWithoutVersion(TQDMProgressBar):
    """
    Progress bar that does not display a version number
    """

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if "v_num" in metrics:
            del metrics["v_num"]
        return metrics


def log_image(trainer: pl.Trainer, name, image):
    if image.ndim == 2:
        image = image[None]  # [channel, y, x]
    trainer.logger.experiment.add_image(name, image, global_step=trainer.global_step)


def log_images(trainer: pl.Trainer, images: dict, one_time=False):
    if one_time and trainer.current_epoch > 1:
        return

    for name, image in images.items():
        # Save two log images for the target (due to the progress bar in tensorboard.)
        if trainer.current_epoch > 1 and "target" in name:
            continue

        log_image(trainer, name, image)


def normalize(img):
    img = np.abs(img)
    return img / img.max()


def outputs_torch2np(outputs: dict):
    """
    Convert outputs from torch.Tensor to numpy.ndarray.
    """
    outputs_np = outputs.copy()
    for key, value in outputs_np.items():
        if isinstance(value, torch.Tensor):
            outputs_np[key] = value.detach().cpu().numpy()

    return outputs_np


def save_recon(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function saves reconstructed images to the out_dir

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    npys_path = out_dir / "npys"

    npys_path.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        file_dir = npys_path / fname
        file_dir.mkdir(exist_ok=True, parents=True)
        npy_path = file_dir / f"{fname}.npy"
        with open(npy_path, "wb") as f:
            np.save(f, recons)


def read_path_yaml(target: str):
    yaml_path = Path.cwd() / "configs/paths.yaml"  # Relative path based on main.py

    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)

        target_path = Path(yaml_dict[target])

    return target_path


def set_name(c):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    common_part = f"{date_str}_{c.model.training_manner}"
    run_name = f"{common_part}_{SLURM_JOB_ID}" if SLURM_JOB_ID is not None else common_part

    return run_name


def save_file_dump(c, subcommand):
    save_path = c.trainer.default_root_dir / "script_dump"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    version = "train" if subcommand in ["fit", "validate"] else "test"

    save_path = save_path / version

    dirs = ["configs", "dlrecon"]
    for dir in dirs:
        try:
            copytree(dir, os.path.join(save_path, dir))
        except:
            pass
    files = ["main.py"]
    for file in files:
        try:
            copy2(file, os.path.join(save_path, file))
        except:
            print(f"{file} does not exist.")


def add_main_arguments(parser: Union[ArgumentParser, LightningArgumentParser]):
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name. If --optuna is given, this is the study name and the experiment "
        "names will be composed of the study name and a consecutive number.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="The path to the training data directory. By default, it is set in the paths.yaml file. Set it manually by passing --data_path parameter at run.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="The path to the log directory. By default, it is set in the paths.yaml file. Set it manually by passing --log_path parameter at run.",
    )


def validate_args(c):
    """
    Add validation for the arguments.
    """
    pass


def device_check(devices: Union[str, int]):
    if devices == "auto":
        return True
    elif devices == "-1":
        return True
    elif isinstance(devices, str):
        devices = [i for i in devices.split(",") if i != ""]
        return True if len(devices) > 1 else False
    else:  # isinstance(devices, int)
        return True if int > 1 else False


def print_training_info(c):
    print(
        f"""
================
HYPERPARAMETERS
================
- NAME:                 {c.name}
- Trainer
    - MAX_EPOCHS:       {c.trainer.max_epochs}
- Model
    - TRAINING_MANNER:  {c.model.training_manner}
    - NUM_CASCADES:     {c.model.num_cascades}
    - NUM_RESBLOCKS:    {c.model.num_resblocks}
    - LR:               {c.model.lr}
    - CHANS:            {c.model.chans}
- Data
    - IS_PROTOTYPE:     {c.data.is_prototype}
- Transform
    - NUM_ACS:          {c.transform.num_acs}
    - MASK_TYPE:        {c.transform.mask_type}   
    - SSDU_MASK_TYPE:   {c.transform.ssdu_mask_type}
- Path
    - DATA_PATH:        {c.data_path}
    - LOG_PATH:         {c.log_path}
"""
    )
