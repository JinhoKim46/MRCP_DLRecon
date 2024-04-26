:exclamation: These implementations are based on the original fastMRI repository by Facebook Research. The original repository can be found [here](https://github.com/facebookresearch/fastMRI).

# Deep Learning-based MRCP Reconstruction 

A Variational Network (VN)-based reconstruction framework for Magnetic Resonance Cholangiopancreatography (MRCP) imaging. We train a VN model on the 3T six-fold retrospectively undersampled MRCP as input and 3T two-fold accelerated MRCP as target. We evaluate the model for six-fold retrospective and prosepctive undersampling acuiqred at 3T and 0.55T. 


## Features of the Framework
- Input: SENSE-based synthesized k-space data on six-fold retrospective undersampling instead of zero-filled k-space data
- Target: GRAPPA reconstruction of two-fold accelerated MRCP (clinical standard)
- Sensitivity maps: Predefined using an ESPIRiT algorithm
- Unrolled network: The Variational Network model

## Installation and Usage
### Installation 
1. Create a new conda environment
   > - conda create -n mrcp_dlrecon python=3.8.17
   > - conda activate mrcp_dlrecon
2. Install `PyTorch` on the conda environment
   > - conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
3. Clone the repository then navigate to the `MRCP_DLRecon` root directory. Run
   > - pip install -e . 

### Usage
#### Data Preparation
- MRI data are stored in the `hdf5` format containing following datasets:
  - `grappa`: target data (y $\times$ x $\times$ Slice)
  - `kdata`: k-space data (nCoil $\times$ PE $\times$ RO $\times$ Slice)
  - `sm_espirit`: sensitivity maps (nCoil $\times$ y $\times$ x $\times$ Slice)
- The `sample_data` directory contains sample data for training, validation, and testing. These data are provided to give an idea of the data structure. You can find the data in LINK_TO_DOI. **In-vivo data for traial run will be available soon**.
- Data splition for training, validation, and testing is done by the `sample_data/dataset.csv` file. Feel free to add additional information about data in this file.
- Replace `path > data_path` in the `config.json` file with the path to the data directory.
#### Run
To run the framework, first define correct configurations in the `config.json` file. 
##### Train
1. Set the value of the `client_arguments > mode` field to `train` in the `config.json` file.
2. Run `main.py` script
3. `SENSE` reconstructions for input are generated and saved in the `CURRENT_PATH/sense/XXX` directory as `xxx.npy`. Therefore, the first run will be slower than the following runs. 
4. Log files containing `checkpoints`, `lightning_logs` for tensorboard, and `script_dump` are stored in `path > log_path` you set in the `config.json` file. 
##### Test
1. Set the value of the `client_arguments > mode` field to `test` in the `config.json` file.
2. Input the log directory name in the correct `ckpt_data`, i.e., `yyyymmdd_hhmmss`. The log directory should be placed in the `path > log_path` directory.
3. The output files are saved in the `path > log_path/npys/FILENAME` directory.

