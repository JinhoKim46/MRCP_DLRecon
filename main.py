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
import pathlib
import json
import datetime
import pytorch_lightning as pl
from argparse import ArgumentParser
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule
from fastmri.utils import save_file_dump
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        data_transform=VarNetDataTransform(),        
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_acc=args.target_acc,
        is_prototype=args.is_prototype
    )   

    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay
    )

    # Add additional callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    args.callbacks.extend([lr_monitor, earlystopping])

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trained_model = VarNetModule.load_from_checkpoint(args.resume_from_checkpoint)
        trainer.test(trained_model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args(config_json):
    parser = ArgumentParser()

    # basic args
    parser.add_argument(
        '--memo',
        type=str,
        default=config_json["memo"],
        help='Memo')

    parser.add_argument(
        '--ckpt_date',
        type=str,
        default=config_json["ckpt_date"],
        help='Data string of the checkpoint to run'
    )

    config = config_json["path"]
    parser.add_argument(
        "--data_path",
        default=pathlib.Path(config["data_path"]),
        type=str,
        help="data_path",
    )
    parser.add_argument(
        "--log_path",
        default=pathlib.Path(config["log_path"]),
        type=str,
        help="log_path",
    )
    
    # client arguments
    config = config_json["client_arguments"]
    parser.add_argument(
        "--mode",
        default=config["mode"],
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    # data transform params
    config = config_json["data_transform"]
    parser.add_argument(
        "--target_acc",
        nargs="+",
        default=config["target_acc"],
        type=int,
        help="Target acceleration factor",
    )

    # data config
    config = config_json["data_config"]
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(        
        batch_size=config["batch_size"],  # number of samples per batch
        is_prototype=config['is_prototype'],
        num_workers=config["num_workers"],  # number of data loading workers
    )
    
    # module config
    config = config_json["module_config"]
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=config["num_cascades"],  # number of unrolled iterations, 12
        pools=config["pools"],  # number of pooling layers for U-Net, 4
        chans=config["chans"],  # number of top-level channels for U-Net, 18        
        lr=config["lr"],  # Adam learning rate
        lr_step_size=config["lr_step_size"],  # epoch at which to decrease learning rate
        lr_gamma=config["lr_gamma"],  # extent to which to decrease learning rate
        weight_decay=config["weight_decay"]  # weight regularization strength
    )

    # trainer config
    config = config_json["trainer_config"]
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        devices=config["devices"],  # number of gpus to use
        accelerator=config['accelerator'],
        seed=config["seed"],  # random seed
        deterministic=config["deterministic"],  # makes things slower, but deterministic
        max_epochs=config["max_epochs"],  # max number of epochs
        enable_checkpointing=config["enable_checkpointing"],  # saving checkpoint or not
    )

    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_id = "." if slurm_job_id == None else f"{slurm_job_id}.tinygpu"

    args = parser.parse_args()

    args.data_path = args.data_path.parent / slurm_job_id / args.data_path.name


    data_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    data_str = f"{data_str}_{slurm_job_id}" if slurm_job_id is not None else data_str
    data_str = data_str if args.ckpt_date == "" else args.ckpt_date

    args.default_root_dir = args.log_path / data_str
    
    # Only set callbacks.ModelCheckpoint when the checkpoint is set to be saved
    if args.enable_checkpointing:
        config = config_json["enable_checkpointing"]
        args.callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=args.default_root_dir / "checkpoints",
                save_top_k=config["save_top_k"],
                verbose=config["verbose"],
                monitor=config["monitor"],
                mode=config["mode"],
            )
        ]

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # set default checkpoint if one exists in our checkpoint directory
    if args.mode == 'test':
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    args.log_every_n_steps=1
    return args


def run_cli():
    config_name = "config.json"
    config = json.load(open(pathlib.Path(__file__).parent.absolute() / config_name))
    args = build_args(config)
    save_file_dump(args, config_name)
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
