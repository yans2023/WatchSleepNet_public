from pathlib import Path
import os
import torch
import sys
import torch.nn as nn


# Define the dynamic file save path generator
def generate_model_save_path(model_name, dataset_name, version=None, suffix=None):
    """
    Generate a customizable model save path and create the directory if it doesn't exist.
    """
    file_name = f"best_saved_model"

    if version:
        file_name += f"_{version}"
    if suffix:
        file_name += f"_{suffix}"

    directory = f"checkpoints/{model_name}/{dataset_name}"
    os.makedirs(directory, exist_ok=True)

    return os.path.join(directory, f"{file_name}.pt")


# Define configuration settings for each dataset in a dictionary
dataset_configurations = {
    "shhs_ibi": {
        "directory": Path("/mnt/nvme2/SHHS_IBI"),
        "downsampling_rate": 1,
        "multiplier": 1,
        "get_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
    "mesa_eibi": {
        "directory": Path("/mnt/nvme2/MESA_EIBI"),
        "downsampling_rate": 1,
        "multiplier": 1,
        "get_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
    "mesa_pibi": {
        "directory": Path("/mnt/nvme2/MESA_PIBI"),
        "downsampling_rate": 1,
        "multiplier": 1,
        "get_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
    "shhs_mesa_ibi": {
        "directory": Path("/mnt/nvme2/SHHS_MESA_IBI"),
        "downsampling_rate": 1,
        "multiplier": 1,
        "get_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
        "dd_analysis_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
    "dreamt_pibi": {
        "directory": Path("/mnt/nvme2/DREAMT_PIBI_SE"),
        "downsampling_rate": 1,
        "multiplier": 1,
        "get_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
        # this may need to be reworked
        "dd_analysis_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
}


# WatchSleepNet Configuration
class WatchSleepNetConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 200
    # PATIENCE = 50
    PATIENCE = 20
    NUM_INPUT_CHANNELS = 1
    FEATURE_CHANNELS = 256
    NUM_CHANNELS = 256
    KERNEL_SIZE = 5
    HIDDEN_DIM = 256
    NUM_HEADS = 16
    TCN_LAYERS = 3
    NUM_LAYERS = 4
    NUM_CLASSES = 3
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1) 
    WEIGHT_DECAY = 1e-4

# InsightSleepNet Configuration (hardcoded parameters based on original paper)
# original paper: NUM_EPOCHS=100, BATCH_SIZE=8, LEARNING_RATE=1e-4, PATIENCE=40
class InsightSleepNetConfig:
    BATCH_SIZE = 4
    OUTPUT_SIZE = 3
    INPUT_SIZE = 750
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 5
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

# SleepConvNet Configuration (hardcoded parameters based on original paper)
class SleepConvNetConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 20
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)
    WEIGHT_DECAY = 1e-3
    OUTPUT_SIZE = 3
