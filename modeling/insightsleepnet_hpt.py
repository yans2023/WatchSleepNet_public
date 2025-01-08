import os
import torch
import argparse
import warnings
import random
import numpy as np
import torch.nn as nn
import optuna
from optuna.exceptions import TrialPruned
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------
# Your custom modules
# ---------------------
from data_setup import create_dataloaders_kfolds
from config import (
    # Replace with your actual config if you have a dedicated one for InsightSleepNet
    # Or reuse SleepConvNetConfig if you prefer, but rename it for clarity:
    InsightSleepNetConfig,
    dataset_configurations,
)
from engine import train_cross_validate_hpo  # <= Ensure this import references the NEW function
# E.g., from engine_new import train_cross_validate_hpo

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
    help="Dataset for which we run hyperparameter optimization on InsightSleepNet."
)
parser.add_argument(
    "--task", 
    type=str, 
    default="sleep_staging", 
    choices=["sleep_staging", "sleep_wake"],
    help="Classification task: sleep_staging or sleep_wake."
)
args = parser.parse_args()

# ---------------------
# RETRIEVE CONFIG
# ---------------------
train_config = dataset_configurations.get(args.train_dataset, None)
if train_config is None:
    raise ValueError(f"No config found for dataset '{args.train_dataset}'.")

# Suppose you have an actual `InsightSleepNetConfig` with default LR, epochs, etc.
model_config = InsightSleepNetConfig

# ---------------------
# CREATE DATA LOADER FOLDS
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

# ---------------------
# OBJECTIVE FUNCTION FOR OPTUNA
# ---------------------
def objective(trial):
    # 1) Sample hyperparams
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

    # 2) Define model_init to create an InsightSleepNet with the chosen hyperparams
    def model_init():
        from models.insightsleepnet import InsightSleepNet
        model = InsightSleepNet(
            input_size=750,
            output_size=3,  # or 2 if "sleep_wake"
            n_filters=n_filters,
            bottleneck_channels=bottleneck_channels,
            kernel_sizes=kernel_sizes,
            num_inception_blocks=num_inception_blocks,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            activation=nn.ReLU(),
        )
        return model  # We'll move the .to(device) step inside the HPO function or the engine if we want

    # 3) Cross-validate using train_cross_validate_hpo
    #    We pass 'model_init' so the engine can create a new model for each fold.
    checkpoint_path = train_config["get_model_save_path"](
        model_name="insightsleepnet",
        dataset_name=args.train_dataset,
        version="optuna_trial",
    )

    (
        results,
        overall_acc,
        overall_f1,
        overall_kappa,
        rem_f1,
        auroc,
    ) = train_cross_validate_hpo(
        model_init=model_init,
        dataloader_folds=dataloader_folds,
        learning_rate=model_config.LEARNING_RATE,
        weight_decay=model_config.WEIGHT_DECAY,
        loss_fn=loss_fn,
        num_epochs=model_config.NUM_EPOCHS,
        patience=model_config.PATIENCE,
        device=DEVICE,
        checkpoint_path=checkpoint_path,
    )

    # Return the metric you want to maximize; let's say overall_kappa
    return overall_kappa

# ---------------------
# RUNNING THE STUDY
# ---------------------
if __name__ == "__main__":
    # Create the study; choose "maximize" if we want to maximize Kappa
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Kappa): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
