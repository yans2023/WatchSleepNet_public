# WatchSleepNet

![Static Badge](https://img.shields.io/badge/License%20-%20MIT%20-%20blue)

## Introduction

WatchSleepNet is a novel deep learning model designed to improve sleep staging using wrist-worn wearables. The model integrates ResNet, Temporal Convolutional Networks (TCN), and LSTM with attention to enhance temporal feature extraction. A key innovation is the pretraining strategy, which first trains on large ECG-derived IBI datasets and then fine-tunes on smaller wrist-worn PPG-derived IBI datasets, significantly improving generalization. WatchSleepNet achieves state-of-the-art performance on the DREAMT dataset, surpassing existing wearable sleep staging models. This repository provides the full implementation and benchmarking tools to support reproducible research and future advancements in sleep monitoring.

## Directory Structure

```bash
.
├── dataset_preparation
│   ├── dreamt
│   │   ├── dreamt_bvp_preprocessing.py
│   │   ├── dreamt_extract_ibi_se.py
│   │   ├── prepare_dreamt_ppg.py
│   │   ├── prepare_dreamt_ppg_whole_sequence.py
│   │   ├── read_raw_e4.py
│   │   ├── read_raw_PSG.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── public_datasets
│   │   ├── extract_ibi_ppg.py
│   │   ├── extract_ibi.py
│   │   ├── __init__.py
│   │   ├── mesa_add_ahi.py
│   │   └── shhs_add_ahi.py
│   └── visualize_raw_signal.ipynb
├── __init__.py
├── modeling
│   ├── checkpoints
│   │   ├── insightsleepnet
│   │   │   ├── dreamt_pibi
│   │   │   └── shhs_mesa_ibi
│   │   │       └── best_saved_model_vtrial.pt
│   │   └── watchsleepnet
│   │       ├── dreamt_pibi
│   │       │   ├── best_saved_model_vtrial_fold1.pt
│   │       │   ├── best_saved_model_vtrial_fold2.pt
│   │       │   ├── best_saved_model_vtrial_fold3.pt
│   │       │   ├── best_saved_model_vtrial_fold4.pt
│   │       │   └── best_saved_model_vtrial_fold5.pt
│   │       └── shhs_mesa_ibi
│   │           └── best_saved_model_vtrial.pt
│   ├── cleaned_files_apnea_severity.json
│   ├── config.py
│   ├── data_setup.py
│   ├── engine.py
│   ├── __init__.py
│   ├── insightsleepnet_hpt.py
│   ├── models
│   │   ├── checkpoints
│   │   │   └── watchsleepnet
│   │   │       ├── dreamt_pibi
│   │   │       ├── shhs_ibi
│   │   │       └── shhs_mesa_ibi
│   │   │           └── best_saved_model_vtrial.pt
│   │   ├── __init__.py
│   │   ├── insightsleepnet.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── insightsleepnet.cpython-310.pyc
│   │   │   ├── sleepconvnet.cpython-310.pyc
│   │   │   └── watchsleepnet.cpython-310.pyc
│   │   ├── sleepconvnet.py
│   │   └── watchsleepnet.py
│   ├── notebooks
│   │   ├── epoch_proportion.ipynb
│   │   ├── figures
│   │   │   ├── confusion_matrix_apnea_category_finetune_all_layers.png
│   │   │   ├── confusion_matrix_apnea_category_finetune_tcn+lstm.png
│   │   │   ├── confusion_matrix_apnea_category.png
│   │   │   ├── confusion_matrix_overall_finetune_all_layers.png
│   │   │   └── sampling_experiment_performance_difference.png
│   │   ├── __init__.py
│   │   ├── plot_ablation_performances.ipynb
│   │   ├── plot_cm.ipynb
│   │   ├── population_characteristics.ipynb
│   │   ├── sampling_experiments_plotting.ipynb
│   │   └── visualize_prediction.ipynb
│   ├── optuna_studies
│   │   └── insightsleepnet_hpo_results.csv
│   ├── print_models.py
│   ├── __pycache__
│   │   ├── config.cpython-310.pyc
│   │   ├── data_setup.cpython-310.pyc
│   │   └── engine.cpython-310.pyc
│   ├── sleepconvnet_hpt.py
│   ├── torch_installation_check.py
│   ├── train_cv.py
│   ├── train_transfer.py
│   ├── utils.py
│   ├── watchsleepnet_cv_ablation.py
│   ├── watchsleepnet_hpt.py
│   └── watchsleepnet_transfer_ablation.py
├── README.md
└── requirements.txt
```

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

```bash
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
