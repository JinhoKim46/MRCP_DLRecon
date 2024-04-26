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
import fastmri
import torch

from fastmri.data import transforms
from fastmri.models import VarNet
from .mri_module import MriModule
from fastmri.utils import recon


class VarNetModule(MriModule):
    """
    VarNet training module inspired by the paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades:       Number of cascades (i.e., layers) for variational network.
            pools:              Number of downsampling and upsampling layers for cascade  U-Net.
            chans:              Number of channels for cascade U-Net.
            lr:                 Learning rate.
            lr_step_size:       Learning rate step size.
            lr_gamma:           Learning rate gamma decay.
            weight_decay:       Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            chans=self.chans,
            pools=self.pools
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, sens_maps=None):
        return self.varnet(masked_kspace, mask, sens_maps)

    def training_step(self, batch, _):
        output, sens_maps = self(batch.input, batch.mask, sens_maps=batch.sens_maps)

        output_recon = recon(output, sens_maps)
        target_recon, output_recon = transforms.center_crop([batch.target, output_recon])

        loss = self.loss(target_recon.unsqueeze(1), output_recon.unsqueeze(1), batch.max_value)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output, sens_maps = self.forward(batch.input, batch.mask, sens_maps=batch.sens_maps)

        output_recon = recon(output, sens_maps)
        target_recon, output_recon = transforms.center_crop([batch.target, output_recon])
        
        loss = self.loss(target_recon.unsqueeze(1), output_recon.unsqueeze(1), batch.max_value)

        self.log("val_loss", loss)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "target": target_recon,
            "output": output_recon,
            "input_k": batch.input,
            "output_k": output,
            "target_k": batch.target_k,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        output, sens_maps = self(batch.input, batch.mask, sens_maps=batch.sens_maps)

        input_recon = recon(batch.input)
        output_recon = recon(output, sens_maps)
        input_recon, output_recon, target_recon = transforms.center_crop([input_recon, output_recon, batch.target])

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "input": input_recon,
            "output": output_recon,
            "target": target_recon,
            "input_k": batch.input,
            "output_k": output,
            "target_k": batch.target_k,
            "rate": batch.rate,
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites
        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", 
            default=0.001, 
            type=float, 
            help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
