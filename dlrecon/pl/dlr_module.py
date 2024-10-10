"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchmetrics

from dlrecon.data.transforms import middle_crop
from dlrecon.losses import MixL1L2Loss
from dlrecon.metrics import NMSEMetric, PSNRMetric, SSIMMetric
from dlrecon.models import MoDLNet


class DLRModule(pl.LightningModule):
    """
    DL Recon training module.
    VN / SSDU / MODL
    """

    def __init__(
        self,
        training_manner: str = "sv",
        num_cascades: int = 12,
        num_resblocks: int = 4,
        chans: int = 18,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        cgdc_iter: int = 10,
        **kwargs,
    ):
        """
        Args:
            training_manner:    Training manner [sv, ssv])
            num_cascades:       Number of cascades (i.e., layers) for unrolled network.
            num_resblocks:      Number of residual blocks for ResNet.
            chans:              Number of channels for ResNet.
            lr:                 Learning rate.
            lr_step_size:       Learning rate step size.
            lr_gamma:           Learning rate gamma decay.
            weight_decay:       Parameter for penalizing weights norm.
            cgdc_iter:          Number of CG iterations for data consistency.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.training_manner = training_manner

        net_params = {
            "num_cascades": num_cascades,
            "chans": chans,
            "num_resblocks": num_resblocks,
            "cg_iter": cgdc_iter,
        }

        self.net = MoDLNet(**net_params)
        self.loss = MixL1L2Loss()

        # evaluation metrics
        self.val_loss = torchmetrics.MeanMetric()
        self.nmse = NMSEMetric()
        self.ssim = SSIMMetric()
        self.psnr = PSNRMetric()

    def forward(self, masked_kspace, train_mask, sens_maps, **kwargs):
        """
        Return:
            output_recon: Reconstructed image (..., 2) real vaued 2 chans.
            output_k: Predicted k-space (...) complex valued

        """
        output_recon, output_k = self.net(masked_kspace, train_mask, sens_maps, **kwargs)
        return torch.view_as_complex(output_recon), output_k

    def common_forward(self, batch):
        output_recon, output_k = self(
            masked_kspace=batch.input,
            train_mask=batch.training_mask,
            sens_maps=batch.sens_maps,
        )

        out_k = output_k * batch.loss_mask
        out_loss = torch.view_as_complex(out_k)
        target_loss = torch.view_as_complex(batch.target)

        loss = self.loss(out_loss, target_loss)

        gt_recon, output_recon = middle_crop([batch.grappa_img, output_recon])

        return (
            output_recon,
            gt_recon,
            loss,
        )

    def training_step(self, batch, _):
        _, _, loss = self.common_forward(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output_recon, gt_recon, loss = self.common_forward(batch)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "grappa_img": gt_recon,
            "output": output_recon,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        output_recon, _ = self(
            masked_kspace=batch.input,
            train_mask=batch.training_mask,
            sens_maps=batch.sens_maps,
        )
        gt_recon, output_recon = middle_crop([batch.grappa_img, output_recon])

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "grappa_img": gt_recon,
            "output": output_recon,
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError("outputs must be a dict")
        # update metrics
        grappa_img = outputs["grappa_img"].abs()
        output = outputs["output"].abs()
        maxval = [v.reshape(-1) for v in batch.gt_max_value]

        self.val_loss.update(outputs["val_loss"])
        self.nmse.update(batch.fname, batch.slice_num, grappa_img, output)
        self.ssim.update(batch.fname, batch.slice_num, grappa_img, output, maxvals=maxval)
        self.psnr.update(batch.fname, batch.slice_num, grappa_img, output, maxvals=maxval)

    def on_validation_epoch_end(self):
        # logging
        self.log("val_loss", self.val_loss, prog_bar=True)
        self.log("validation_metrics/nmse", self.nmse)
        self.log("validation_metrics/ssim", self.ssim, prog_bar=True)
        self.log("validation_metrics/psnr", self.psnr)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError("outputs must be a dict")
        # update metrics
        grappa_img = outputs["grappa_img"].abs()
        output = outputs["output"].abs()
        maxval = [v.reshape(-1) for v in batch.gt_max_value]

        self.nmse.update(batch.fname, batch.slice_num, grappa_img, output)
        self.ssim.update(batch.fname, batch.slice_num, grappa_img, output, maxvals=maxval)
        self.psnr.update(batch.fname, batch.slice_num, grappa_img, output, maxvals=maxval)

    def on_test_epoch_end(self):
        # logging
        self.log("test_metrics/nmse", self.nmse)
        self.log("test_metrics/ssim", self.ssim, prog_bar=True)
        self.log("test_metrics/psnr", self.psnr)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)

        return [optim], [scheduler]
