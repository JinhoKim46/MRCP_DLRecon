"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
Copyright (c) Jinho Kim (jinho.kim@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback

import dlrecon.pl.utils as pl_utils


class ValImageLogger(pl.Callback):

    def __init__(
        self,
        num_log_images: int = 16,
        logging_interval: int = 10,
        log_always_after: float = 0.8,
    ):
        """
        Args:
            num_log_images: Number of validation images to log. Defaults to 16.
            logging_interval: After how many epochs to log validation images. Defaults to 10.
            log_always_after: After what percentage of trainer.max_epochs to log images in every epoch.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.logging_interval = logging_interval
        self.log_always_after = log_always_after

        self.output_imgs = defaultdict(dict)
        self.grappa_imgs = defaultdict(dict)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            num_log_images=self.num_log_images,
            val_log_indices=self.val_log_indices,
            logging_interval=self.logging_interval,
            log_always_after=self.log_always_after,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError("Expected outputs to be a dict")

        if trainer.sanity_checking:
            return

        # check inputs
        for k in ("batch_idx", "fname", "slice_num", "grappa_img", "output"):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step.")

        outputs = pl_utils.outputs_torch2np(outputs)

        if outputs["output"].ndim == 2:
            outputs["output"] = outputs["output"].unsqueeze(0)
        elif outputs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if outputs["grappa_img"].ndim == 2:
            outputs["grappa_img"] = outputs["grappa_img"].unsqueeze(0)
        elif outputs["grappa_img"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            if isinstance(trainer.val_dataloaders, list):
                examples = [j for i in trainer.val_dataloaders.dataset.examples for j in i]
            elif trainer.val_dataloaders is not None:
                examples = trainer.val_dataloaders.dataset.examples
            else:
                raise RuntimeError("Could not determine number of validation samples")

            examples_enum = list(enumerate(examples))
            middle_slice_idx = [i[0] for i in examples_enum if 12 < i[1][1] < 25]
            self.val_log_indices = list(np.random.permutation(middle_slice_idx)[: self.num_log_images])

        # log images to tensorboard
        if isinstance(outputs["batch_idx"], int):
            batch_indices = [outputs["batch_idx"]]
        else:
            batch_indices = outputs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                fname = outputs["fname"][i].split(".")[0] + "_" + str(outputs["slice_num"][i])
                key = f"images_val"
                grappa = np.abs(outputs["grappa_img"][i])
                output = np.abs(outputs["output"][i])
                output = pl_utils.normalize(output)
                grappa = pl_utils.normalize(grappa)
                error = grappa - output
                images = {
                    f"{key}/{fname}/GRAPPA": grappa,
                    f"{key}/{fname}/recon": output,
                    f"{key}/{fname}/error": error,
                }
                pl_utils.log_images(trainer, images)

        for i, (fname, slice_num) in enumerate(zip(outputs["fname"], outputs["slice_num"])):
            # [:-3]: to remove .h5 from the filename
            self.output_imgs[fname[:-3]][slice_num.item()] = np.abs(outputs["output"][i])
            self.grappa_imgs[fname[:-3]][slice_num.item()] = np.abs(outputs["grappa_img"][i])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        # stack all the slices for each file
        stack_imgs = lambda x: np.stack([out[1] for out in sorted(x.items())])
        output_imgs = defaultdict()
        grappa_imgs = defaultdict()
        for fname in self.output_imgs.keys():
            output_imgs[fname] = stack_imgs(self.output_imgs[fname])
            grappa_imgs[fname] = stack_imgs(self.grappa_imgs[fname])

        for fname in self.output_imgs.keys():
            recon = output_imgs[fname].copy()
            grappa_img = grappa_imgs[fname].copy()

            mip_recon = np.max(recon, axis=0)
            mip_recon = pl_utils.normalize(mip_recon)

            mip_grappa_img = np.max(grappa_img, axis=0)
            mip_grappa_img = pl_utils.normalize(mip_grappa_img)

            mips = {
                f"MIP_val/{fname}/recon": mip_recon,
                f"MIP_val/{fname}/GRAPPA": mip_grappa_img,
            }
            pl_utils.log_images(trainer, mips)


class TestImageLogger(pl.Callback):

    def __init__(
        self,
    ):
        super().__init__()
        self.num_log_images = 10
        self.test_log_indices = None

        self.save_img = True

        self.output_imgs = defaultdict(dict)
        self.grappa_imgs = defaultdict(dict)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(test_log_indices=self.test_log_indices, save_img=self.save_img)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError("Expected outputs to be a dict")

        outputs = pl_utils.outputs_torch2np(outputs)

        for i, (fname, slice_num) in enumerate(zip(outputs["fname"], outputs["slice_num"])):
            # [:-3]: to remove .h5 from the filename
            self.output_imgs[fname[:-3]][slice_num.item()] = np.abs(outputs["output"][i])
            self.grappa_imgs[fname[:-3]][slice_num.item()] = np.abs(outputs["grappa_img"][i])

    def on_test_epoch_end(self, trainer, pl_module):
        save_path = Path(trainer.default_root_dir)

        # stack all the slices for each file
        stack_imgs = lambda x: np.stack([out[1] for out in sorted(x.items())])
        output_imgs = defaultdict()
        grappa_imgs = defaultdict()
        for fname in self.output_imgs.keys():
            output_imgs[fname] = stack_imgs(self.output_imgs[fname])
            grappa_imgs[fname] = stack_imgs(self.grappa_imgs[fname])

        for fname in self.output_imgs.keys():
            recon = output_imgs[fname].copy()
            grappa_img = grappa_imgs[fname].copy()

            mip_recon = np.max(recon, axis=0)
            mip_recon = pl_utils.normalize(mip_recon)

            mip_grappa_img = np.max(grappa_img, axis=0)
            mip_grappa_img = pl_utils.normalize(mip_grappa_img)

            mips = {
                f"MIP_test/{fname}/recon": mip_recon,
                f"MIP_test/{fname}/GRAPPA": mip_grappa_img,
            }
            pl_utils.log_images(trainer, mips)

        pl_utils.save_recon(output_imgs, save_path)


class MyPyTorchLightningPruningCallback(PyTorchLightningPruningCallback):
    """
    Version of the PyTorchLightningPruningCallback with a quick fix for
    an incompatibility issue between optuna and PyTorch Lightning
    """

    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        trainer.training_type_plugin = trainer.strategy
        super().on_validation_end(trainer, pl_module)
        del trainer.training_type_plugin


class MetricMonitor(pl.Callback):
    """
    Callback that monitors a metric during training
    """

    def __init__(self, monitor: str, mode: str):
        super().__init__()
        self.monitor = monitor
        assert mode in ["max", "min"], f"Unknown mode {mode}"
        self.mode = mode
        self.best = None

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode, best=self.best)

    def on_validation_end(self, trainer, pl_module):
        current_metric = trainer.callback_metrics[self.monitor]
        if (
            self.best is None
            or (self.mode == "max" and current_metric > self.best)
            or (self.mode == "min" and current_metric < self.best)
        ):
            self.best = current_metric


class EarlyStoppingOnLossThreshold(pl.Callback):
    def __init__(self, loss_threshold=None):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.loss_threshold is not None:
            train_loss = outputs["loss"].item()
            self._run_early_stopping_check(trainer, train_loss, "train")

    def on_validation_end(self, trainer, pl_module):
        if self.loss_threshold is not None:
            val_loss = trainer.callback_metrics.get("val_loss").item()
            self._run_early_stopping_check(trainer, val_loss, "validation")

    def _run_early_stopping_check(self, trainer, loss, subcommand):
        if loss > self.loss_threshold:
            print(f"Stopping training as {subcommand} loss {loss} exceeds threshold {self.loss_threshold}")
            trainer.should_stop = True
