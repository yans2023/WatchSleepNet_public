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
        "directory": Path("/home/willkewang/Datasets/SHHS_MESA_IBI"),
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
        "directory": Path("/home/willkewang/Datasets/DREAMT_PIBI_SE"),
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
    PATIENCE = 20
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    # Model hyperparams (best from your tuning or existing defaults)
    NUM_INPUT_CHANNELS = 1         # e.g., raw input channels
    NUM_CHANNELS = 256             # 'num_channels'
    KERNEL_SIZE = 5
    HIDDEN_DIM = 256
    NUM_HEADS = 16
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
    Configuration for InsightSleepNet, with updated best hyperparams from HPO.
    """
    # Training hyperparams
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 5
    WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.CrossEntropyLoss(ignore_index=-1)

    # Best model hyperparams from your tuning
    INPUT_SIZE = 750
    OUTPUT_SIZE = 3
    N_FILTERS = 32
    BOTTLENECK_CHANNELS = 16
    KERNEL_SIZES = [9, 19, 39]
    NUM_INCEPTION_BLOCKS = 3
    USE_RESIDUAL = False
    DROPOUT_RATE = 0.2
    # Possibly define activation in code, or keep as a field
    ACTIVATION = nn.ReLU()  

    @classmethod
    def to_dict(cls):
        """
        Convert the class fields into a dict for setup_model_and_optimizer(...).
        """
        return {
            # For the model constructor
            "input_size": cls.INPUT_SIZE,
            "output_size": cls.OUTPUT_SIZE,
            "n_filters": cls.N_FILTERS,
            "bottleneck_channels": cls.BOTTLENECK_CHANNELS,
            "kernel_sizes": cls.KERNEL_SIZES,
            "num_inception_blocks": cls.NUM_INCEPTION_BLOCKS,
            "use_residual": cls.USE_RESIDUAL,
            "dropout_rate": cls.DROPOUT_RATE,
            "activation": cls.ACTIVATION,
            # --- Training hyperparams ---
            "BATCH_SIZE": cls.BATCH_SIZE,
            "LEARNING_RATE": cls.LEARNING_RATE,
            "WEIGHT_DECAY": cls.WEIGHT_DECAY,
            "NUM_EPOCHS": cls.NUM_EPOCHS,
            "PATIENCE": cls.PATIENCE,
            "LOSS_FN": cls.LOSS_FN,
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