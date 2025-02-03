# WatchSleepNet

![Static Badge](https://img.shields.io/badge/License%20-%20MIT%20-%20blue)

<img src="./media/WatchSleepNet Logo.png" height="100"/>

## Introduction

WatchSleepNet is a novel deep learning model designed to improve sleep staging using wrist-worn wearables. The model integrates ResNet, Temporal Convolutional Networks (TCN), and LSTM with attention to enhance temporal feature extraction. A key innovation is the pretraining strategy, which first trains on large ECG-derived Inter-Beat Interval (IBI) datasets and then fine-tunes on smaller wrist-worn PPG-derived IBI datasets, significantly improving generalization. WatchSleepNet achieves state-of-the-art performance on the DREAMT dataset, surpassing existing wearable sleep staging models. This repository provides the full implementation and benchmarking tools to support reproducible research and future advancements in sleep monitoring.

## Directory Structure

```bash
.
├── dataset_preparation
│   ├── dreamt
│   │   ├── dreamt_extract_ibi_se.py # Run program to extract IBI from downloaded DREAMT dataset
│   │   └── utils.py
│   ├── public_datasets
│   │   ├── mesa_extract_ppg.py # Run program to extract PPG from MESA dataset
│   │   ├── shhs_extract_ecg.py # Run program to extract ECG from SHHS dataset
│   │   ├── extract_and_save_ibi.py # Run program to save IBI extracted from both MESA and SHHS datasets.
│   │   └── utils.py
│   └── visualize_raw_signal.ipynb # (Optional) Example code to visualize raw signals
├── modeling
│   ├── config.py # Configure parameters related to dataset and models
│   ├── data_setup.py
│   ├── engine.py
│   ├── insightsleepnet_hpt.py # Run program to perform hyperparameter tuning for InsightSleepNet
│   ├── sleepconvnet_hpt.py # Run program to perform hyperparameter tuning for SleepConvNet
│   ├── watchsleepnet_hpt.py # Run program to perform hyperparameter tuning for WatchSleepNet
│   ├── models # Model architectures
│   │   ├── insightsleepnet.py
│   │   ├── sleepconvnet.py
│   │   └── watchsleepnet.py
│   ├── notebooks # (Optional) Example code to visualize experiment results
│   ├── optuna_studies # Hyperparameter tuning results
│   ├── train_cv.py
│   ├── train_transfer.py # Run program to perform transfer learning experiments
│   ├── utils.py
│   ├── watchsleepnet_cv_ablation.py # Run program to perform ablation experiments (DREAMT) on WatchSleepNet
│   └── watchsleepnet_transfer_ablation.py # Run program to perform ablation experiments (Transfer Learning) on WatchSleepNet
├── README.md
└── requirements.txt
```
> [!TIP]
> The files without description contain utility functions that are used in the main programs.

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

1. Download the open source datasets ([DREAMT](https://physionet.org/content/dreamt/1.0.0/), [MESA](https://sleepdata.org/datasets/mesa), [SHHS](https://sleepdata.org/datasets/shhs)) to your designated directory.

2. Extract IBI from DREAMT dataset
```
python dataset_preperation/dreamt/dreamt_extract_ibi_se.py
```
> [!TIP]
> Update the `root_path` variable in the file to where you stored the DREAMT dataset. The directory should be named `DREAMT_PIBI_SE`.

3. Extract PPG from MESA and ECG from SHHS
```
python dataset_preperation/public_dataset/mesa_extract_ppg.py
python dataset_preperation/public_dataset/shhs_extract_ecg.py
```
> [!TIP]
> Update the `out_dir` path in the files to where you stored the MESA and SHHS datasets.

4. Extract IBI from MESA and SHHS
```
python dataset_preperation/public_dataset/extract_and_save_ibi.py
```
> [!TIP]
> Update `shhs_input_dir` and `mesa_input_dir` to where you stored the extracted PPG and ECG datasets. Update `output_dir` to where you want to store the combined IBI. The directory should be named `SHHS_MESA_IBI`.

5. Confirm that the dataset root directory contains the following subdirectories with the same names:

```bash
.
├── DREAMT_PIBI_SE
├── MESA_PIBI
└── SHHS_MESA_IBI
```

6. Set the `DATASET_DIR` variable in `modeling/config.py` to your dataset root directory:
```
DATASET_DIR = # Enter your dataset root directory
```

### Experiment 1: Transfer Learning

You can perform transfer learning experiments (pre-train on IBI from SHHS+MESA and test on DREAMT IBI) using the `modeling/train_transfer.py`. Run the experiment with WatchSleepNet:
```
python modeling/train_transfer.py
```
To perform the experiment with other benchmark models (i.e. InsightSleepNet, SleepConvNet), indicate selected model using the `--model` parser argument:
```
python modeling/train_transfer.py --model=insightsleepnet
```
```
python modeling/train_transfer.py --model=sleepconvnet
```

### Experiment 2: WatchSleepNet Ablation Study

You can perform ablation experiments on WatchSleepNet using `modeling/watchsleepnet_cv_ablation.py`. Run WatchSleepNet without the TCN and Attention components
```
python modeling/watchsleepnet_cv_ablation.py
```
or 
```
python modeling/watchsleepnet_transfer_ablation.py
```
> [!TIP]
> `watchsleepnet_cv_ablation.py` tests performance on using only DREAMT while `watchsleepnet_transfer_ablation.py` performs transfer learning on SHHS+MESA and then tests on DREAMT.

Utilize the argument flags to run either/both the TCN and Attention components
```
python modeling/watchsleepnet_cv_ablation.py --use_tcn --use_attention
```

### Hyperparameter Tuning

You can perform hyperparameter tuning for WatchSleepNet, InsightSleepNet, and SleepConvNet. For example, to tune WatchSleepNet run
```
python modeling/watchsleepnet_hpt.py
```