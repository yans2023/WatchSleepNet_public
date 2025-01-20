# WatchSleepNet

## Description

## Directory Structure

## Installation 

### Dependencies

The following dependencies are required for this project and have been tested for the listed versions. Please ensure these dependencies are installed and compatible with your GPU setup:

```
numpy=1.23.5
scikit-learn=1.6.1
torch=2.5.1
optuna=4.1.0
neurokit2=0.2.10
psutil=6.1.1
```
## Usage

### Dataset Preperation
Confirm that the root directory containing experiment datasets are structured as follows (and that the naming of each directory is consistent with the example):

```
```
Update the variable `DATASET_DIR` in `modeling/config.py` to your selected dataset root directory before proceeding to replicate the experiments.
```
# Example
DATASET_DIR = /mnt/nvme2/
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

## Acknowledgement