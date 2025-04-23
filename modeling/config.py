from pathlib import Path
import os
import torch
import sys
import torch.nn as nn

### Enter path to your dataset (formatted per README)
DATASET_DIR = "/home/willkewang/Datasets/"

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
        "dd_analysis_model_save_path": lambda model_name=None, dataset_name=None, version=None, suffix=None: generate_model_save_path(
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
            suffix=suffix,
        ),
    },
    "shhs_ibi": {
        "directory": Path("{}SHHS_IBI".format(DATASET_DIR)),
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
    # TODO
    "mesa_ppg": {
        # "directory": Path("{}SHHS_MESA_IBI".format(DATASET_DIR)),
        # "directory": Path("{}MESA_PPG".format(DATASET_DIR)),
        "directory": Path("/mnt/nvme2/MESA_PPG_preprocessed/"),
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
        "directory": Path("/mnt/nvme2/DREAMT_PIBI_SE_updated"),
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
    "dreamt_ppg": {
        "directory": Path("/mnt/nvme2/DREAMT_PPG_preprocesse/"),
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
}


class WatchSleepNetConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 200
    PATIENCE = 20
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    NUM_INPUT_CHANNELS = 1         # e.g., raw input channels
    NUM_CHANNELS = 256             # 'num_channels'
    KERNEL_SIZE = 5
    HIDDEN_DIM = 256
    NUM_HEADS = 16
    TCN_LAYERS = 3
    NUM_LAYERS = 4
    NUM_CLASSES = 3                 # W / N / R
    # NUM_CLASSES = 4                 # W / L / D / R

    USE_TCN = True
    USE_ATTENTION = True
    USE_LSTM = True

    @classmethod
    def to_dict(cls):
        return {
            # --- General model constructor fields ---
            "num_features": cls.NUM_INPUT_CHANNELS,
            "num_channels": cls.NUM_CHANNELS,
            "kernel_size": cls.KERNEL_SIZE,
            "hidden_dim": cls.HIDDEN_DIM,
            "num_heads": cls.NUM_HEADS,
            "num_layers": cls.NUM_LAYERS,
            "tcn_layers": cls.TCN_LAYERS,
            "use_tcn": cls.USE_TCN,
            "use_attention": cls.USE_ATTENTION,
            "num_classes": cls.NUM_CLASSES,
            # --- Training hyperparams ---
            "BATCH_SIZE": cls.BATCH_SIZE,
            "LEARNING_RATE": cls.LEARNING_RATE,
            "WEIGHT_DECAY": cls.WEIGHT_DECAY,
            "NUM_EPOCHS": cls.NUM_EPOCHS,
            "PATIENCE": cls.PATIENCE,
            "LOSS_FN": cls.LOSS_FN,
        }
    
class InsightSleepNetConfig:
    # --- Training hyperparams ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 200
    PATIENCE = 10
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)
    NUM_CLASSES = 3

    # --- Old Architecture Defaults (multi-block) ---
    INPUT_SIZE = 750    # Each segment length
    OUTPUT_SIZE = 3     # e.g., number of classes
    DROPOUT_RATE = 0.5
    ACTIVATION = nn.ReLU()

    BLOCK_CONFIGS = [
        {
            "in_channels": 32,   "n_filters": 8,
            "bottleneck_channels": 8,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
        {
            "in_channels": 32,   "n_filters": 16,
            "bottleneck_channels": 16,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
        {
            "in_channels": 64,   "n_filters": 16,
            "bottleneck_channels": 16,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
        {
            "in_channels": 64,   "n_filters": 32,
            "bottleneck_channels": 32,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
        {
            "in_channels": 128,  "n_filters": 64,
            "bottleneck_channels": 64,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
        {
            "in_channels": 256,  "n_filters": 128,
            "bottleneck_channels": 128,
            "kernel_sizes": [5,11,23],
            "use_residual": True
        },
    ]

    INITIAL_CONV_OUT = 32
    FINAL_POOL_SIZE = 1100

    @classmethod
    def to_dict(cls):
        return {
            # For the model constructor
            "input_size":       cls.INPUT_SIZE,
            "output_size":      cls.OUTPUT_SIZE,
            "dropout_rate":     cls.DROPOUT_RATE,
            "activation":       cls.ACTIVATION,
            "block_configs":    cls.BLOCK_CONFIGS,
            "initial_conv_out": cls.INITIAL_CONV_OUT,
            "final_pool_size":  cls.FINAL_POOL_SIZE,
            "num_classes":      cls.NUM_CLASSES,

            # --- Training hyperparams ---
            "BATCH_SIZE":    cls.BATCH_SIZE,
            "LEARNING_RATE": cls.LEARNING_RATE,
            "WEIGHT_DECAY":  cls.WEIGHT_DECAY,
            "NUM_EPOCHS":    cls.NUM_EPOCHS,
            "PATIENCE":      cls.PATIENCE,
            "LOSS_FN":       cls.LOSS_FN,
        }
    
class SleepConvNetConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    PATIENCE = 20
    WEIGHT_DECAY = 1e-3
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    INPUT_SIZE = 750
    TARGET_SIZE = 256
    NUM_SEGMENTS = 1100
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.2
    CONV_LAYERS_CONFIGS = [(1, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)]
    DILATION_LAYERS_CONFIGS = None
    USE_RESIDUAL = True

    @classmethod
    def to_dict(cls):
        """
        Convert the class fields into a dict for setup_model_and_optimizer(...).
        """
        return {
            # For SleepConvNet constructor
            "input_size": cls.INPUT_SIZE,
            "target_size": cls.TARGET_SIZE,
            "num_segments": cls.NUM_SEGMENTS,
            "num_classes": cls.NUM_CLASSES,
            "dropout_rate": cls.DROPOUT_RATE,
            "conv_layers_configs": cls.CONV_LAYERS_CONFIGS,
            "dilation_layers_configs": cls.DILATION_LAYERS_CONFIGS,
            "use_residual": cls.USE_RESIDUAL,
            # --- Training hyperparams ---
            "BATCH_SIZE": cls.BATCH_SIZE,
            "LEARNING_RATE": cls.LEARNING_RATE,
            "WEIGHT_DECAY": cls.WEIGHT_DECAY,
            "NUM_EPOCHS": cls.NUM_EPOCHS,
            "PATIENCE": cls.PATIENCE,
            "LOSS_FN": cls.LOSS_FN,
        }


class SleepPPGNetConfig:
    """
    Configuration for SleepPPG-Net, implementing residual convolutional layers and a TCN for sleep staging.
    """
    # Training hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 40
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Model hyperparameters
    NUM_CLASSES = 3
    INPUT_CHANNELS = 1
    NUM_RES_BLOCKS = 4
    TCN_LAYERS = 2
    HIDDEN_DIM = 32
    DROPOUT_RATE = 0.2
    
    @classmethod
    def to_dict(cls):
        """
        Convert the class fields into a dictionary for model initialization.
        """
        return {
            "input_channels": cls.INPUT_CHANNELS,
            "num_classes": cls.NUM_CLASSES,
            "num_res_blocks": cls.NUM_RES_BLOCKS,
            "tcn_layers": cls.TCN_LAYERS,
            "hidden_dim": cls.HIDDEN_DIM,
            "dropout_rate": cls.DROPOUT_RATE,
            "BATCH_SIZE": cls.BATCH_SIZE,
            "LEARNING_RATE": cls.LEARNING_RATE,
            "WEIGHT_DECAY": cls.WEIGHT_DECAY,
            "NUM_EPOCHS": cls.NUM_EPOCHS,
            "PATIENCE": cls.PATIENCE,
            "LOSS_FN": cls.LOSS_FN,
        }