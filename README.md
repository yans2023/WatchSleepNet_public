# WatchSleepNet

## Description

## Directory Structure

## Installation 

1. Clone the repository

```
git clone https://github.com/WillKeWang/WatchSleepNet_public.git
```

2. Setup virtual environment (recommended)
```
conda create -n watchsleepnet python=3.10
conda activate watchsleepnet
```
> [!Note]
> Develpment was conducted in an conda environment, but you may use another package manager of your choice.

2. Install required dependencies
```
pip install -r requirements.txt
```
> [!TIP]
> The versions in `requirements.txt` have been tested with our compute setup. Please update the versions if they are not compatible with your CUDA setup.


## Usage

### Dataset Preperation

1. Confirm that the dataset root directory contains the following subdirectories with the same name:

```
.
├── DREAMT_EIBI
├── DREAMT_PIBI
├── MESA_EIBI
├── MESA_PIBI
├── SHHS_IBI
└── SHHS_MESA_IBI
```

2. Update the variable `DATASET_DIR` in `modeling/config.py` to your selected dataset root directory before proceeding to replicate the experiments.

1. Set the `DATASET_DIR` variable in `modeling/config.py` to your dataset root directory:
```
DATASET_DIR = # Enter your dataset root directory
```

### Experiment 1: Transfer Learning

You can perform transfer learning experiments (pre-train on IBI from SHHS+MESA and test on DREAMT IBI) using the `modeling/train_transfer.py`. To run WatchSleepNet, simply run:
```
python train_transfer.py
```
To perform the experiment with other benchmark models (i.e. InsightSleepNet, SleepConvNet), indicate selected model using the `--model` parser argument:
```
python train_transfer.py --model=insightsleepnet
```
```
python train_transfer.py --model=sleepconvnet
```

### Experiment 2: WatchSleepNet Ablation Study

### Hyperparameter Tuning

