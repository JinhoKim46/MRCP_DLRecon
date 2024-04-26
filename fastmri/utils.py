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
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
from numpy.fft import ifftshift, fftshift, fftn

import fastmri

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
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        file_dir = out_dir / fname
        file_dir.mkdir(exist_ok=True, parents=True)
        npy_path = file_dir / f"{fname}_DLRcon.npy"
        with open(npy_path, "wb") as f:
            np.save(f, recons)

def save_file_dump(args, config_name):
    from shutil import copy2, copytree

    save_path = args.default_root_dir / "script_dump"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id == None:
        slurm_job_id = _get_next_version(save_path)

    save_path = save_path / f"version_{slurm_job_id}"

    dirs = ["fastmri"]
    for dir in dirs:
        try:
            copytree(dir, os.path.join(save_path, dir))
        except:
            pass
    files = ["main.py", "fastmri/utils.py", config_name]
    for file in files:
        try:
            copy2(file, os.path.join(save_path, file))
        except:
            print(f"{file} does not exist.")


def _get_next_version(root_dir):
    existing_versions = os.listdir(root_dir)
    existing_versions_cp = existing_versions.copy()

    for i in existing_versions:
        if "test" in i or "train" in i:
            existing_versions_cp.remove(i)

    if len(existing_versions_cp) == 0:
        return 0

    existing_versions_cp = [int(i.split("_")[1]) for i in existing_versions_cp]
    return max(existing_versions_cp) + 1


def kdata2tensorboard(
    kspace_torch: Union[torch.Tensor, list], batch_idx: int = 0, coil_idx: int = 0, smoothing_factor: int = 8
):
    """
    Function for making k-space visualizations for Tensorboard.
    """

    if isinstance(kspace_torch, list):
        kspace_grids = torch.zeros(
            (len(kspace_torch), 1, kspace_torch[0].shape[0], *kspace_torch[0].shape[-3:-1])
        )
        for i in range(len(kspace_torch)):
            kspace_grids[i] = kdata2tensorboard(kspace_torch[i], batch_idx=batch_idx, coil_idx=coil_idx)
        return torch.permute(kspace_grids, (2, 0, 1, 3, 4))
    else:
        kspace_torch = kspace_torch[batch_idx]

        kspace_torch = kspace_torch.detach()[coil_idx, ...]
        # Assumes that the smallest values will be close enough to 0 as to not matter much.
        kspace_grid = fastmri.complex_abs(kspace_torch)
        # Scaling & smoothing.
        # smoothing_factor converted to float32 tensor. expm1 and log1p require float32 tensors.
        # They cannot accept python integers.
        sf = torch.tensor(smoothing_factor, dtype=torch.float32)
        kspace_grid *= torch.expm1(sf) / kspace_grid.max()
        kspace_grid = torch.log1p(kspace_grid)  # Adds 1 to input for natural log.
        kspace_grid /= kspace_grid.max()  # Standardization to 0~1 range.

        return kspace_grid.to(device="cpu", non_blocking=True)


def recon(kspace_torch: Union[torch.tensor, list], sens_maps: Optional[torch.tensor] = None):
    """

    Args:
        kspace_torch: kspace to be recon
                torch.tensor: size of (batch, coil, row, col, 2)
                list: size of cascades
                        each element: size of (batch, coil, row, col, 2)

    Returns:
        reconstructed image
    """
    if isinstance(kspace_torch, list):
        recon_torch = torch.zeros(
            (len(kspace_torch), 1, kspace_torch[0].shape[0], *kspace_torch[0].shape[-3:-1])
        )
        for i in range(len(kspace_torch)):
            recon_torch[i] = recon(kspace_torch[i])
        return torch.permute(recon_torch, (2, 0, 1, 3, 4), sens_maps)
    else:
        coil_imgs = fastmri.ifft2c(kspace_torch)

        if sens_maps is not None:
            cb_imgs =  fastmri.complex_mul(coil_imgs, fastmri.complex_conj(sens_maps)).sum(dim=1)
            return fastmri.complex_abs(cb_imgs)
        else:            
            return torch.sqrt((fastmri.complex_abs(coil_imgs) ** 2).sum(1))


def sense(kspace, sens_maps, dinfo):
    def fft(img):
        data = ifftshift(img, axes=[-2, -1])
        data = fftn(data, axes=(-2, -1), norm="ortho")
        data = fftshift(data, axes=[-2, -1])
        return data.astype(np.complex64)

    root = Path(os.getcwd())
    dname, dfolder = dinfo
    folder = root / "sense" / dfolder
    if (folder / f"{dname}.npy").exists():
        return np.load(open(folder / f"{dname}.npy", "rb"))
    else:
        folder.mkdir(parents=True, exist_ok=True)

        sense_recon = (
            mr.app.SenseRecon(
                kspace.copy(), sens_maps.copy(), lamda=0.01, device=sp.Device(0), show_pbar=False
            )
            .run()
            .get()
        )
        kspace_sense = fft(sense_recon * sens_maps)

        # data consistency
        kspace_sense = np.where(kspace, kspace, kspace_sense)

        # save kspace
        np.save(open(folder / f"{dname}.npy", "wb"), kspace_sense)

        return kspace_sense