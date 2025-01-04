import os
import torch
import argparse
import warnings
import random
import numpy as np
import optuna

warnings.filterwarnings("ignore", category=UserWarning)

from data_setup import create_dataloaders_kfolds
from models.sleepconvnet2 import SleepConvNet
from config import (
    SleepConvNetConfig,
    dataset_configurations,
)
from engine import train_cross_validate, train_cross_validate_hpo
from optuna.exceptions import TrialPruned

# Seed settings for reproducibility
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# System Settings
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dataset",
    type=str,
    default="shhs_mesa_ibi",
    choices=[
        "shhs_ibi",
        "mesa_eibi",
        "mesa_pibi",
        "shhs_mesa_ibi",
    ],
)
parser.add_argument(
    "--task", type=str, default="sleep_staging", choices=["sleep_staging", "sleep_wake"]
)
args = parser.parse_args()

# Retrieve configuration based on the dataset argument
train_config = dataset_configurations.get(args.train_dataset, None)
model_config = SleepConvNetConfig

# Create folds for cross-validation
dataloader_folds = create_dataloaders_kfolds(
    dir=train_config["directory"],
    dataset=args.train_dataset,
    num_folds=5,
    val_ratio=0.2,
    batch_size=model_config.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    multiplier=train_config["multiplier"],
    downsampling_rate=train_config["downsampling_rate"],
)

loss_fn = model_config.LOSS_FN


def objective(trial):
    # For now, do not tune dropout or dilation. Just pick dropout = 0.2
    dropout_rate = 0.2

    # Choose conv_layers_configs
    conv_layers_configs = trial.suggest_categorical(
        "conv_layers_configs",
        [
            [(1, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)],  # final_in_channels=128
            [(1, 32, 3, 1), (32, 48, 3, 1), (48, 64, 3, 1)],  # final_in_channels=64
        ],
    )

    final_in_channels = conv_layers_configs[-1][1]

    # Choose dilation_layers_configs deterministically based on final_in_channels
    # No `suggest_categorical` here, just pick the matching set directly.
    if final_in_channels == 128:
        dilation_layers_configs = [(128, 128, 7, d) for d in [2, 4, 8, 16, 32]]
    else:
        dilation_layers_configs = [(64, 64, 7, d) for d in [2, 4, 8]]

    def model_init():
        from models.sleepconvnet2 import SleepConvNet

        return SleepConvNet(
            input_size=750,
            target_size=256,
            num_segments=1100,
            num_classes=3,
            dropout_rate=dropout_rate,
            conv_layers_configs=conv_layers_configs,
            dilation_layers_configs=dilation_layers_configs,
        ).to(DEVICE)

    # Run cross-validation
    results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = (
        train_cross_validate_hpo(
            model_init=model_init,
            dataloader_folds=dataloader_folds,
            learning_rate=model_config.LEARNING_RATE,
            weight_decay=model_config.WEIGHT_DECAY,
            loss_fn=loss_fn,
            num_epochs=model_config.NUM_EPOCHS,
            patience=model_config.PATIENCE,
            device=DEVICE,
            checkpoint_path=train_config["get_model_save_path"](
                model_name="sleepconvnet",
                dataset_name=args.train_dataset,
                version="optuna_trial",
            ),
        )
    )

    # Use overall_kappa as the metric to maximize
    return overall_kappa


if __name__ == "__main__":
    # Create a study object and start optimizing
    study = optuna.create_study(direction="maximize")
    # Try a limited number of trials, e.g., 5
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
