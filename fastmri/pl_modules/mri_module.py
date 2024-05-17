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

import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import fastmri
import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from torchmetrics.metric import Metric
import fastmri.utils as utils


class DistributedMetricSum(Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 10, **kwargs):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 10.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None
        self.test_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "target",
            "output",
            "input_k",
            "output_k",
            "target_k",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step.")
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            if len(self.trainer.val_dataloaders[0]) < self.num_log_images:
                self.val_log_indices = list(range(len(self.trainer.val_dataloaders[0])))
            else:
                self.val_log_indices = list(np.linspace(10, len(self.trainer.val_dataloaders[0]), self.num_log_images).astype(int))

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                fname = val_logs["fname"][i].split(".")[0] + "_" + str(val_logs["slice_num"][i].cpu().numpy())
                key = f"images/val_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                output = output / output.max()
                target = target / target.max()
                error = torch.abs(target - output)
                error = error / error.max()
                self.log_image(f"{key}/target/{fname}", target)
                self.log_image(f"{key}/reconstruction/{fname}", output)
                self.log_image(f"{key}/error/{fname}", error)

                key = f"kspace/val_idx_{batch_idx}"
                input_k = utils.kdata2tensorboard(val_logs["input_k"], batch_idx=i).unsqueeze(0)
                output_k = utils.kdata2tensorboard(val_logs["output_k"], batch_idx=i).unsqueeze(0)
                target_k = utils.kdata2tensorboard(val_logs["target_k"], batch_idx=i).unsqueeze(0)
                self.log_image(f"{key}/input/{fname}", input_k)
                self.log_image(f"{key}/output/{fname}", output_k)
                self.log_image(f"{key}/target/{fname}", target_k)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        gt_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            ground_truth = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(evaluate.mse(ground_truth, output)).view(1)
            gt_norms[fname][slice_num] = torch.tensor(evaluate.mse(ground_truth, np.zeros_like(ground_truth))).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(evaluate.ssim(ground_truth[None, ...], output[None, ...], maxval=maxval)).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"], 
            "mse_vals": dict(mse_vals), 
            "gt_norms": dict(gt_norms), 
            "ssim_vals": dict(ssim_vals), 
            "max_vals": max_vals, 
            "slice_num": val_logs["slice_num"].cpu(), 
            "fname": val_logs["fname"], 
            "output": val_logs["output"].cpu(), 
            "target": val_logs["target"].cpu()
            }

    def test_step_end(self, test_log):
        # pick a set of images to log if we don't have one already
        if self.test_log_indices is None:
            if len(self.trainer.test_dataloaders[0]) < self.num_log_images:
                self.test_log_indices = list(range(len(self.trainer.test_dataloaders[0])))
            else:
                self.test_log_indices = list(np.linspace(10, len(self.trainer.test_dataloaders[0]), self.num_log_images).astype(int))

        # log images to tensorboard
        if isinstance(test_log["batch_idx"], int):
            batch_indices = [test_log["batch_idx"]]
        else:
            batch_indices = test_log["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.test_log_indices:
                fname = test_log["fname"][i].split(".")[0] + "_" + str(test_log["slice_num"][i].cpu().numpy())
                key = f"images_test/test_idx_{batch_idx}"
                input_us = test_log["input"][i].unsqueeze(0)
                input_us = input_us / input_us.max()

                output = test_log["output"][i].unsqueeze(0)
                output = output / output.max()

                target = test_log["target"][i].unsqueeze(0)
                target = target / target.max()

                self.log_image(f"{key}/input/{fname}", input_us)
                self.log_image(f"{key}/output/{fname}", output)
                self.log_image(f"{key}/target/{fname}", target)

                key = f"kdata_test/test_idx_{batch_idx}"
                input_k = utils.kdata2tensorboard(test_log["input_k"], batch_idx=i).unsqueeze(0)
                output_k = utils.kdata2tensorboard(test_log["output_k"], batch_idx=i).unsqueeze(0)
                target_k = utils.kdata2tensorboard(test_log["target_k"], batch_idx=i).unsqueeze(0)
                self.log_image(f"{key}/input_k/{fname}", input_k)
                self.log_image(f"{key}/output_k/{fname}", output_k)
                self.log_image(f"{key}/target_k/{fname}", target_k)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        gt_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(test_log["fname"]):
            slice_num = int(test_log["slice_num"][i].cpu())
            maxval = test_log["max_value"][i].cpu().numpy()
            output = test_log["output"][i].cpu().numpy()
            target = test_log["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(evaluate.mse(target, output)).view(1)
            gt_norms[fname][slice_num] = torch.tensor(evaluate.mse(target, np.zeros_like(target))).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)).view(1)
            max_vals[fname] = maxval

        return {"mse_vals": dict(mse_vals), 
        "gt_norms": dict(gt_norms), 
        "ssim_vals": dict(ssim_vals), 
        "max_vals": max_vals, 
        "slice_num": test_log["slice_num"].cpu(), 
        "fname": test_log["fname"], 
        "output": test_log["output"].cpu()
        }

    def log_image(self, name, image, dformats="CHW"):
        self.logger.experiment.add_image(name, image, global_step=self.global_step, dataformats=dformats)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        gt_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
                gt_norms[k].update(val_log["gt_norms"][k])
                ssim_vals[k].update(val_log["ssim_vals"][k])
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert mse_vals.keys() == gt_norms.keys() == ssim_vals.keys() == max_vals.keys()

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            gt_norm = torch.mean(torch.cat([v.view(-1) for _, v in gt_norms[fname].items()]))
            metrics["nmse"] = metrics["nmse"] + mse_val / gt_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(torch.tensor(len(losses), dtype=torch.float))

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

        outputs = defaultdict(dict)
        targets = defaultdict(dict)
        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in val_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice_num"])):
                # fname is xx.h5. So just take xx from the fname
                outputs[fname.split(".")[0]][int(slice_num.cpu())] = log["output"][i]
                targets[fname.split(".")[0]][int(slice_num.cpu())] = log["target"][i]

        # stack all the slices for each file
        stack_imgs = lambda x: np.stack([out[1].cpu().numpy() for out in sorted(x.items())])
        for fname in outputs.keys():
            outputs[fname] = stack_imgs(outputs[fname])
            targets[fname] = stack_imgs(targets[fname])

        for fname in outputs.keys():
            recon = outputs[fname].copy()
            target = targets[fname].copy()

            mip_recon = torch.tensor(np.max(recon, axis=0)).unsqueeze(0)
            mip_recon -= mip_recon.min()
            mip_recon /= mip_recon.max()

            mip_target = torch.tensor(np.max(target, axis=0)).unsqueeze(0)
            mip_target -= mip_target.min()
            mip_target /= mip_target.max()
            self.log_image(f"MIP_val/{fname}-Recon", mip_recon)
            self.log_image(f"MIP_val/{fname}-Target", mip_target)

    def test_epoch_end(self, test_logs):
        # aggregate losses
        mse_vals = defaultdict(dict)
        gt_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for test_log in test_logs:
            for k in test_log["mse_vals"].keys():
                mse_vals[k].update(test_log["mse_vals"][k])
            for k in test_log["gt_norms"].keys():
                gt_norms[k].update(test_log["gt_norms"][k])
            for k in test_log["ssim_vals"].keys():
                ssim_vals[k].update(test_log["ssim_vals"][k])
            for k in test_log["max_vals"]:
                max_vals[k] = test_log["max_vals"][k]

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            gt_norm = torch.mean(torch.cat([v.view(-1) for _, v in gt_norms[fname].items()]))
            metrics["nmse"] = metrics["nmse"] + mse_val / gt_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples)

        """Saving recon_images"""
        outputs = defaultdict(dict)
        # Use dicts for aggregation to handle duplicate slices
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice_num"])):
                # Take file name without extensions.
                outputs[fname.split(".")[0]][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack([out[1].cpu().numpy() for out in sorted(outputs[fname].items())])

        for fname, recon in outputs.items():
            key = f"MIP_test/{fname}"
            recon = recon.copy()
            mip = torch.tensor(np.max(recon, axis=0)).unsqueeze(0)
            mip -= mip.min()
            mip /= mip.max()
            self.log_image(f"{key}", mip)

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "npys"
        else:
            save_path = pathlib.Path.cwd() / "npys"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_recon(outputs, save_path)


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
