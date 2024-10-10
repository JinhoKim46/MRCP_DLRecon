"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim <jinho.kim@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
from typing import Dict, Optional, Union

import optuna
import torch.cuda
from jsonargparse import Namespace
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

import dlrecon.pl.utils as pl_utils

from ..data.subsample import get_mask_type
from . import DataModule, DLRModule, TQDMProgressBarWithoutVersion


class DLReconCLI(LightningCLI):
    """
    Customized LightningCLI for MRCP DLRecon
    """

    def __init__(
        self,
        overwrite_args: Optional[Dict] = None,
        trial: Optional[optuna.Trial] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.overwrite_args = overwrite_args
        self.trial = trial
        self.name = name

        parser_kwargs = kwargs.pop("parser_kwargs", {})
        save_config_kwargs = kwargs.pop("save_config_kwargs", {})
        save_config_kwargs.update({"overwrite": True})

        # check if a config file has been passed via command line
        for i, arg in enumerate(sys.argv):
            if arg in ["-h", "--help", "--print_config"]:
                break
            if arg in ["-c", "--config"]:
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                    break
        else:
            raise RuntimeError("No config file given")

        super().__init__(
            DLRModule,
            DataModule,
            save_config_kwargs=save_config_kwargs,
            parser_kwargs=parser_kwargs,
            trainer_defaults={"callbacks": [TQDMProgressBarWithoutVersion()]},
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        pl_utils.add_main_arguments(parser)

        group = parser.add_argument_group("Data transform and undersampling options:")
        group.add_argument(
            "--transform.mask_type",
            type=str,
            default="equispaced",
            help="Type of mask function for undersampling",
        )
        group.add_argument(
            "--transform.acceleartion_rate",
            type=int,
            default=6,
            help="Acceleration rate for undersampling",
        )
        group.add_argument(
            "--transform.num_acs",
            type=int,
            default=24,
            help="The number of autocalibration signal(ACS) lines",
        )
        group.add_argument(
            "--transform.ssdu_mask_center_block",
            type=int,
            default=5,
            help="The size of the SSDU center block.",
        )
        group.add_argument(
            "--transform.ssdu_mask_rho",
            type=float,
            default=0.4,
            help="A ratio of available points between the lambda mask (loss mask) and the Omega mask (original undersampling mask) (||Mask_lambda||/||Mask_omega||)",
        )
        group.add_argument(
            "--transform.ssdu_mask_std_scale",
            type=int,
            default=4,
            help="A standard deviation scale for the SSDU Gaussian mask.",
        )
        group.add_argument(
            "--transform.ssdu_mask_type",
            type=str,
            default="gaussian1d",
            choices=["gaussian1d"],
            help="A type of the SSDU sampling distribution.",
        )

        group = parser.add_argument_group("Callback shortcut options:")
        group.add_argument(
            "--callback.val_log_images",
            type=int,
            default=16,
            help="Number of images to log during validation",
        )
        group.add_argument(
            "--callback.val_log_interval",
            type=int,
            default=10,
            help="Interval for logging validation images",
        )
        group.add_argument(
            "--callback.checkpoint_monitor",
            type=str,
            default="val_loss",
            help="Metric to monitor for checkpointing",
        )
        group.add_argument(
            "--callback.checkpoint_mode",
            type=str,
            default="max",
            choices=["min", "max"],
            help="Mode for checkpointing",
        )

        parser.add_argument(
            "--float32_matmul_precision",
            type=str,
            default="highest",
            choices=["highest", "high", "medium"],
            help="Precision of float32 matrix multiplications",
        )

    def before_instantiate_classes(self) -> None:
        subcommand = self.config["subcommand"]
        if subcommand == "predict":
            raise NotImplementedError("Prediction is not supported, please use `test` subcommand for inference")

        c = self.config[subcommand]
        if c.trainer.callbacks is None:
            c.trainer.callbacks = []  # initialize with empty list

        c.name = pl_utils.set_name(c) if self.name is None else self.name

        # optuna
        if self.trial is not None:
            c.trainer.num_sanity_val_steps = 0
            c.trainer.callbacks.append(
                Namespace(
                    {
                        "class_path": "dlrecon.pl.MetricMonitor",
                        "init_args": {"monitor": "val_loss", "mode": "max"},
                    }
                )
            )

        # set default paths based on directory config
        c.data_path = pl_utils.read_path_yaml("data_path") if c.data_path is None else c.data_path
        c.log_path = pl_utils.read_path_yaml("log_path") if c.log_path is None else c.log_path

        c.data.data_path = c.data_path
        c.trainer.default_root_dir = c.log_path / c.name

        c.trainer.log_every_n_steps = 50

        # overwrite args given via constructor
        if self.overwrite_args:
            for k, v in self.overwrite_args.items():
                c[k] = v

        # for Optuna, runs are saved in date_str / {date_str}_{Trial}.
        if self.overwrite_args is not None:
            c.trainer.default_root_dir = c.trainer.default_root_dir / c.name

        # configure checkpointing in checkpoint_dir
        checkpoint_dir = c.trainer.default_root_dir / "checkpoints"
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for callback in c.trainer.callbacks:
            if callback["class_path"] == "pytorch_lightning.callbacks.ModelCheckpoint":
                callback["init_args"]["dirpath"] = checkpoint_dir

        # set default checkpoint if one exists in our checkpoint directory
        if checkpoint_dir.exists():
            ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
            if ckpt_list:
                c.ckpt_path = str(ckpt_list[-1])
        if self.subcommand in ["test", "predict"] and c.ckpt_path is None:
            raise RuntimeError("No checkpoint available")

        # logger
        c.trainer.logger = Namespace(
            {
                "class_path": "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
                "init_args": {
                    "save_dir": c.trainer.default_root_dir,
                    "version": ("train" if self.subcommand in ["fit", "validate"] else "test"),
                },
            }
        )

        # logging callback
        c.trainer.callbacks.append(
            Namespace(
                {
                    "class_path": "dlrecon.pl.ValImageLogger",
                    "init_args": {
                        "num_log_images": c.callback.val_log_images,
                        "logging_interval": c.callback.val_log_interval,
                    },
                }
            )
        )

        # save reconstructions callback
        c.trainer.callbacks.append(
            Namespace(
                {
                    "class_path": "dlrecon.pl.TestImageLogger",
                    "init_args": {},
                }
            )
        )

        # data.data_selection should be a list when it is given
        if c.data.data_selection is not None:
            if isinstance(c.data.data_selection, str):
                c.data.data_selection = c.data.data_selection.split(",")
                c.data.data_selection = [x.strip() for x in c.data.data_selection]

        # initialize DDP if all of the following conditions are met:
        # 1. more than one device available
        # 2. `devices` is 'auto', -1 or greater than 1
        if torch.cuda.device_count() > 1 and pl_utils.device_check(c.trainer.devices):
            c.trainer.strategy = Namespace(
                {
                    "class_path": "pytorch_lightning.strategies.DDPStrategy",
                    "init_args": {
                        "find_unused_parameters": False,
                        "static_graph": True,
                    },
                }
            )
            c.data.distributed_sampler = True

        # mask function and transform objects
        mask_class = get_mask_type(c.transform.mask_type)
        mask_func = Namespace(
            {
                "class_path": mask_class.__module__ + "." + mask_class.__qualname__,
                "init_args": {
                    "num_acs": c.transform.num_acs,
                    "acceleration": c.transform.acceleartion_rate,
                },
            }
        )

        # SSDU mask setting when architecture is "SSDU" or "VN with SSV training manner" for "fit" or "validation" subcommand
        if c.model.training_manner == "ssv" and self.subcommand in ["fit", "validate"]:
            block_size = c.transform.ssdu_mask_center_block
            ssdu_mask_type = c.transform.ssdu_mask_type

            if ssdu_mask_type == "gaussian1d":
                ssdu_mask_class = "SSDUGaussianMask1D"
            else:
                raise ValueError(f"Invalid SSDU mask type {ssdu_mask_type}. Please implement your own SSDU mask class.")

            ssdu_mask_func = Namespace(
                {
                    "class_path": f"dlrecon.data.ssdu_subsample.{ssdu_mask_class}",
                    "init_args": {
                        "center_block": (block_size, block_size),
                        "rho": c.transform.ssdu_mask_rho,
                        "std_scale": c.transform.ssdu_mask_std_scale,
                    },
                }
            )
        else:
            ssdu_mask_func = None

        # use random masks for train transform, fixed masks for val transform
        if c.data.train_transform is None and self.subcommand == "fit":
            c.data.train_transform = Namespace(
                {
                    "class_path": "dlrecon.data.transforms.DataTransform",
                    "init_args": {
                        "mask_func": mask_func,
                        "ssdu_mask_func": ssdu_mask_func,
                        "training_manner": c.model.training_manner,
                        "use_seed": False,
                    },
                }
            )

        if c.data.val_transform is None and self.subcommand in ["fit", "validate"]:
            c.data.val_transform = Namespace(
                {
                    "class_path": "dlrecon.data.transforms.DataTransform",
                    "init_args": {
                        "mask_func": mask_func,
                        "ssdu_mask_func": ssdu_mask_func,
                        "training_manner": c.model.training_manner,
                        "use_seed": True,
                    },
                }
            )
        if c.data.test_transform is None and self.subcommand == "test":
            c.data.test_transform = Namespace(
                {
                    "class_path": "dlrecon.data.transforms.DataTransform",
                    "init_args": {
                        "mask_func": mask_func,
                        "ssdu_mask_func": ssdu_mask_func,
                        "training_manner": c.model.training_manner,
                        "use_seed": True,
                    },
                }
            )

        # float32 matrix multiplication precision
        torch.set_float32_matmul_precision(c.float32_matmul_precision)

        pl_utils.validate_args(c)
        pl_utils.save_file_dump(c, self.subcommand)
        pl_utils.print_training_info(c)
