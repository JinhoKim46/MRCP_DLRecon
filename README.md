# Deep Learning-based MRCP Reconstruction 

Deep Learning-based (DL) reconstruction framework for Magnetic Resonance Cholangiopancreatography (MRCP) imaging. We use ResNet-based DL models for supervised (SV) and self-supervised training. We train DL models on the 3T six-fold retrospectively undersampled MRCP. We evaluate the model for six-fold retrospective and prospective undersampling acquired at 3T and 0.55T. 


## Installation
1. Clone the repository then navigate to the `MRCP_DLRecon` root directory.
```sh
git clone git@github.com:JinhoKim46/MRCP_DLRecon.git
cd MRCP_DLRecon
```
2. Create a new conda environment
```sh
conda env create -f environment.yml
```
1. Activate the conda environment
```sh
conda activate mrcp_dlrecon
```
1. Install a dlrecon package
```sh
pip install -e . 
```

## Usage
### Data Preparation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13912092.svg)](https://doi.org/10.5281/zenodo.13912092)

- MRI data are stored in the `HDF5` format containing the following structures:
  - Datasets:
    - `grappa`: target data (y $\times$ x $\times$ Slice)
    - `kdata_raw`: Raw $k$-space data (x2) (nCoil $\times$ PE $\times$ RO $\times$ Slice)
    - `kdata_fs`: Fully-sampled $k$-space data from `kdata_raw` using GRAPPA (nCoil $\times$ PE $\times$ RO $\times$ Slice)
    - `sm_espirit`: ESPIRiT-based sensitivity maps (nCoil $\times$ y $\times$ x $\times$ Slice)
- The `sample_data` directory contains sample MRCP data for training, validation, and testing. We provide **two two-fold (2x) 3D MRCP** and **one six-fold (6x) 3D MRCP**. In the `sample_data/dataset.csv`, the two 2x MRCP data are defined for training and validation, and the 6x MRCP data is marked for testing. You can find the data [here](https://doi.org/10.5281/zenodo.13912092). Additional information, such as header information, is ignored in the sample data. 
- Data splitting for training, validation, and testing is done by the `sample_data/dataset.csv` file.
- Replace `data_path` in the `configs/paths.yaml` file with the actual path to the data.
### Run
#### Train
1. Define the training configurations in the `configs/dlrecon.yaml` file.
2. Run `main.py` by
```sh
python main.py fit --config configs/dlrecon.yaml
```
1. You can define the run name by adding the `--name` argument at run. Unless you define the run name, it is set to `%Y%m%d_%H%M%S_{training_manner}` with the current date and time. 
  ```sh
  python main.py fit --config configs/dlrecon.yaml --name test_run
  ```
1. You can overwrite the configurations in the `configs/dlrecon.yaml` file by adding arguments at run. 
  ```sh
  python main.py fit --config configs/dlrecon.yaml --model.training_manner ssv
  ```
5. Log files containing `checkpoints/`, `lightning_logs/`, and `script_dump/` are stored in `log_path/run_name`. `log_path` is defined in the `configs/paths.yaml` file.
6. You can resume the training by giving `run_name` with `fit` command. `*.ckpt` file should be placed in `run_name/checkpoints/` to resume the model.
  ```sh
  python main.py fit --config configs/dlrecon.yaml --name run_name
  ```
#### Test
1. Run `main.py` with `run_name` by
```sh
python main.py test --config configs/dlrecon.yaml --name run_name
```
1. `*.ckpt` file should be placed in `run_name/checkpoints/` to test the model.
2. The output files are saved in `log_path/run_name/npys/FILENAME`.

## Citation
Please cite the following [paper](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/nbm.70002) if this repository is helpful for your research :)
  
  ```
  Kim, J., Nickel, M. and Knoll, F. (2025), Deep Learning-Based Accelerated MR Cholangiopancreatography Without Fully-Sampled Data. NMR in Biomedicine, 38: e70002. https://doi.org/10.1002/nbm.70002
  ```
