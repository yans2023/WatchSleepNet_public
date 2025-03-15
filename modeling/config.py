from pathlib import Path
import os
import torch
import sys
import torch.nn as nn


### Enter path to your dataset (formatted per README)
DATASET_DIR = "/mnt/nvme2/"


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
        "directory": Path("{}SHHS_IBI".format(DATASET_DIR)),
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
        "directory": Path("{}MESA_EIBI".format(DATASET_DIR)),
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
        "directory": Path("{}MESA_PIBI".format(DATASET_DIR)),
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
        "directory": Path("{}SHHS_MESA_IBI".format(DATASET_DIR)),
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
        "directory": Path("{}DREAMT_PIBI_SE".format(DATASET_DIR)),
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


class WatchSleepNetConfig:
    """
    Configuration for the new WatchSleepNet (previously watchsleepnet2) 
    with ablation flags for TCN/attention built in.
    """
    # Training hyperparams
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 200
    PATIENCE = 50
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    # Model hyperparams (best from your tuning or existing defaults)
    NUM_INPUT_CHANNELS = 1         # e.g., raw input channels
    NUM_CHANNELS = 256             # 'num_channels'
    KERNEL_SIZE = 5
    HIDDEN_DIM = 256
    NUM_HEADS = 32
    TCN_LAYERS = 3
    NUM_LAYERS = 4
    NUM_CLASSES = 3
    # For ablation flags, set them to True by default (or parameterize them)
    USE_TCN = True
    USE_ATTENTION = True

    @classmethod
    def to_dict(cls):
        """
        Convert the class fields into a dict for setup_model_and_optimizer(...) usage.
        """
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
    """
    Configuration for InsightSleepNet, defaulting to the old multi-block architecture.
    """
    # --- Training hyperparams ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 20
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)
    NUM_CLASSES = 3

    # --- Old Architecture Defaults (multi-block) ---
    INPUT_SIZE = 750    # Each segment length
    OUTPUT_SIZE = 3     # e.g., number of classes
    DROPOUT_RATE = 0.2
    ACTIVATION = nn.ReLU()

    # Instead of num_inception_blocks/n_filters, we define block-by-block:
    # (in_channels, n_filters, bottleneck, kernel_sizes, use_residual)
    # matching your old code’s 6-block progression from 32->64->64->128->256->512 channels.
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

    # For the initial conv “Conv1d(1,32, kernel_size=40, stride=20)” => out_channels=32
    INITIAL_CONV_OUT = 32
    # Then final pooling => "AdaptiveAvgPool1d(1100)" 
    FINAL_POOL_SIZE = 1100

    @classmethod
    def to_dict(cls):
        """
        Convert the class fields into a dict for setup_model_and_optimizer(...).
        The key 'block_configs' will tell our new parametric InsightSleepNet 
        to build the 6-block progression (old architecture).
        """
        return {
            # For the model constructor
            "input_size":       cls.INPUT_SIZE,
            "output_size":      cls.OUTPUT_SIZE,
            "dropout_rate":     cls.DROPOUT_RATE,
            "activation":       cls.ACTIVATION,
            "block_configs":    cls.BLOCK_CONFIGS,       # Old architecture blocks
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
    """
    Configuration for SleepConvNet, with updated best hyperparams from HPO.
    """
    # Training hyperparams
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 20
    WEIGHT_DECAY = 1e-3
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    # Best model hyperparams from your tuning
    INPUT_SIZE = 750
    TARGET_SIZE = 256
    NUM_SEGMENTS = 1100
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.2
    CONV_LAYERS_CONFIGS = [(1, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)]
    DILATION_LAYERS_CONFIGS = None   # or your custom best config
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
    PATIENCE = 10
    WEIGHT_DECAY = 1e-3
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
