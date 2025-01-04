import os
import torch
import argparse
import warnings
import random
import numpy as np
import torch.nn as nn
import optuna

warnings.filterwarnings("ignore", category=UserWarning)

from data_setup import create_dataloaders_kfolds
# from models.insightsleepnet import InsightSleepNet  # Make sure to import the correct path to your InsightSleepNet
from config import (
    # You may have an analogous config for InsightSleepNet, or reuse SleepConvNetConfig
    SleepConvNetConfig,
    dataset_configurations,
)
from engine import train_cross_validate_hpo
from optuna.exceptions import TrialPruned

# ---------------------
# SEED SETTINGS
# ---------------------
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------
# SYSTEM SETTINGS
# ---------------------
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# ARGUMENT PARSING
# ---------------------
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

# Retrieve config from the dataset argument
train_config = dataset_configurations.get(args.train_dataset, None)
model_config = SleepConvNetConfig  # or a dedicated config for InsightSleepNet

# ---------------------
# DATALOADER FOLDS
# ---------------------
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
    n_filters = trial.suggest_categorical("n_filters", [24, 32])
    bottleneck_channels = trial.suggest_categorical("bottleneck_channels", [16, 32])
    kernel_sizes = trial.suggest_categorical(
        "kernel_sizes",
        [
            [5, 11, 23],
            [9, 19, 39],
        ],
    )
    num_inception_blocks = trial.suggest_int("num_inception_blocks", 2, 4)
    use_residual = trial.suggest_categorical("use_residual", [True, False])
    dropout_rate = 0.2  # or trial.suggest_float("dropout_rate", 0.1, 0.3)

    def model_init():
        from models.insightsleepnet2 import InsightSleepNet
        model = InsightSleepNet(
            input_size=750,
            output_size=3,
            n_filters=n_filters,
            bottleneck_channels=bottleneck_channels,
            kernel_sizes=kernel_sizes,
            num_inception_blocks=num_inception_blocks,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            activation=nn.ReLU(),
        ).to(DEVICE)
        return model


    # Run cross-validation
    results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_cross_validate_hpo(
        model_init=model_init,
        dataloader_folds=dataloader_folds,
        learning_rate=model_config.LEARNING_RATE,
        weight_decay=model_config.WEIGHT_DECAY,
        loss_fn=loss_fn,
        num_epochs=model_config.NUM_EPOCHS,
        patience=model_config.PATIENCE,
        device=DEVICE,
        checkpoint_path=train_config["get_model_save_path"](
            model_name="insightsleepnet",
            dataset_name=args.train_dataset,
            version="optuna_trial",
        ),
    )

    # Return whichever metric you want to maximize
    return overall_kappa


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Kappa): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")