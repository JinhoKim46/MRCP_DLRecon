:exclamation: These implementations are based on the original fastMRI repository by Facebook Research. The original repository can be found [here](https://github.com/facebookresearch/fastMRI).

# Deep Learning-based MRCP Reconstruction 

A Variational Network (VN)-based reconstruction framework for Magnetic Resonance Cholangiopancreatography (MRCP) imaging. We train a VN model on the 3T six-fold retrospectively undersampled MRCP as input and 3T two-fold accelerated MRCP as the target. We evaluate the model for six-fold retrospective and prospective undersampling acquired at 3T and 0.55T. 


## Features of the Framework
- Input: SENSE-based synthesized k-space data on six-fold retrospective undersampling instead of zero-filled k-space data
- Target: GRAPPA reconstruction of two-fold accelerated MRCP (clinical standard)
- Sensitivity maps: Predefined using an ESPIRiT algorithm
- Unrolled network: The Variational Network model

## Installation and Usage
### Installation 
1. Create a new conda environment
```sh
conda create -n mrcp_dlrecon python=3.8.17
```
```sh
conda activate mrcp_dlrecon
```
2. Install `PyTorch` on the conda environment
```sh
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
3. Clone the repository then navigate to the `MRCP_DLRecon` root directory. Run
```sh
pip install -e . 
```

### Usage
#### Data Preparation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11209901.svg)](https://doi.org/10.5281/zenodo.11209901)

- MRI data are stored in the `HDF5` format containing the following structures:
  - Datasets:
    - `grappa`: target data (y $\times$ x $\times$ Slice)
    - `kdata`: $k$-space data (nCoil $\times$ PE $\times$ RO $\times$ Slice)
    - `sm_espirit`: ESPIRiT-based sensitivity maps (nCoil $\times$ y $\times$ x $\times$ Slice)
  - Attributes:
    - `base_acc`: acceleration factor of the raw data. 
- The `sample_data` directory contains sample MRCP data for training, validation, and testing. We provide **two two-fold (2x) 3D MRCP** and **one six-fold (6x) 3D MRCP**. In the `sample_data/dataset.csv`, the two 2x MRCP data are defined for training and validation, and the 6x MRCP data is marked for testing. You can find the data [here](https://doi.org/10.5281/zenodo.11209901). Additional information, such as header information, is ignored in the sample data. 
- Data splitting for training, validation, and testing is done by the `sample_data/dataset.csv` file.
- Replace `path > data_path` in the `config.json` file with the actual path to the data.
#### Run
To run the framework, first define the correct configurations in the `config.json` file. 
##### Train
1. Set the value of the `client_arguments > mode` field to `train` in the `config.json` file.
2. Run `main.py` script
3. `SENSE` reconstructions for input are generated and saved in the `CURRENT_PATH/sense/XXX` directory as `xxx.npy`. Therefore, the first iteration will be slower than the following iterations. 
4. Log files containing `checkpoints/`, `lightning_logs/`, and `script_dump/` are stored in `path > log_path/yyyymmdd_hhmmss`. 
##### Test
1. Set the `client_arguments > mode` field to `test` in the `config.json` file.
2. Set the `ckpt_data` field to the log directory, i.e., `yyyymmdd_hhmmss`, containing the checkpoint file. The log directory should be placed in the `path > log_path`.
3. The output files are saved in the `path > log_path/ckpt_data/npys/FILENAME`.
- To run the framework with the provided sample data, set the `ckpt_data` field to `trained_model` and `client_arguments > mode` field to `test`. 