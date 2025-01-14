"""
Hyperparameter Optimization for WatchSleepNet
--------------------------------------------

This script uses Optuna to tune hyperparameters primarily related to the TCN
and LSTM (e.g., number of layers). It performs cross-validation on your dataset
to evaluate each trial. Modify dataset creation and training functions as needed.

Usage:
  python hpo_watchsleepnet.py --train_dataset shhs_mesa_ibi
"""

import os
import argparse
import warnings
import logging
import random

import torch
import numpy as np
import optuna
from optuna.exceptions import TrialPruned

# Replace these with your actual data loading and engine modules
from data_setup import create_dataloaders_kfolds
from engine import train_cross_validate_hpo  # Adjust to your actual training loop for WatchSleepNet

# Import WatchSleepNet and/or related configuration
from models.watchsleepnet import WatchSleepNet  # Adjust import path as needed
from config import dataset_configurations  # Adjust if you have a config dict for each dataset

# ----------------------- Logging Configuration -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ----------------------- Utility Functions -----------------------
def set_seed(seed: int = 0):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed} for reproducibility.")

def get_device() -> torch.device:
    """
    Determine the computation device (CPU or CUDA).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device set to: {device}")
    return device

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for WatchSleepNet"
    )

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
        help="Dataset to train on."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sleep_staging",
        choices=["sleep_staging", "sleep_wake"],
        help="Task to perform."
    )
    args = parser.parse_args()
    logger.info("Command-line arguments parsed successfully.")
    return args

# ----------------------- Objective Function for Optuna -----------------------
def objective(
    trial,
    train_config: dict,
    dataloader_folds: list,
    device: torch.device,
    model_name: str,
):
    """
    Objective function for Optuna hyperparameter tuning of WatchSleepNet.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        train_config (dict): Dataset and training configurations.
        dataloader_folds (list): List of dataloader tuples for cross-validation.
        device (torch.device): Computation device.
        model_name (str): Name for saving or referencing the model.

    Returns:
        float: The metric to maximize (e.g., Cohen's Kappa).
    """

    # ------------------ Hyperparameter Suggestions ------------------
    # Number of TCN layers (depth of TCN block)
    tcn_layers = trial.suggest_int("tcn_layers", 1, 4)

    # Number of LSTM layers
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)

    # Optionally, tune whether to use TCN or attention
    use_tcn = trial.suggest_categorical("use_tcn", [True, False])
    use_attention = trial.suggest_categorical("use_attention", [True, False])

    # Optionally, tune learning_rate and weight_decay
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # ------------------ Model Initialization ------------------
    def model_init():
        # Build the WatchSleepNet with the suggested hyperparameters
        # Adjust input dimensions as per your setup
        model = WatchSleepNet(
            num_features=1,        # e.g., IBI has 1 channel
            num_channels=64,       # TCN channel size
            kernel_size=3,         # TCN kernel size
            hidden_dim=128,        # LSTM hidden dimension
            num_heads=4,           # Multi-head attention heads
            num_layers=lstm_layers, # LSTM layers
            num_classes=3,         # e.g., 3-class sleep staging
            tcn_layers=tcn_layers,
            use_tcn=use_tcn,
            use_attention=use_attention,
        ).to(device)
        return model

    try:
        # ------------------ Cross-Validation Training ------------------
        # Adjust train_cross_validate_hpo to your actual function for training WatchSleepNet
        results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_cross_validate_hpo(
            model_init=model_init,
            dataloader_folds=dataloader_folds,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_fn=torch.nn.CrossEntropyLoss(),  # or any other loss you use
            num_epochs=train_config.get("NUM_EPOCHS", 10),
            patience=train_config.get("PATIENCE", 5),
            device=device,
            checkpoint_path=train_config["get_model_save_path"](
                model_name=model_name,
                dataset_name=train_config["dataset"],
                version="optuna_trial",
            ),
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise TrialPruned()

    # ------------------ Metric to Maximize ------------------
    # We'll use overall_kappa in this example
    logger.info(f"Trial completed with overall_kappa={overall_kappa}")
    return overall_kappa

# ----------------------- Main Function -----------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(seed=0)
    device = get_device()
    args = parse_arguments()

    # Retrieve config for the chosen dataset
    train_config = dataset_configurations.get(args.train_dataset, None)
    if train_config is None:
        logger.error(f"Configuration for dataset '{args.train_dataset}' not found.")
        raise ValueError(f"Configuration for dataset '{args.train_dataset}' not found.")

    # Setup cross-validation dataloaders
    # Adjust create_dataloaders_kfolds call as needed for your data
    dataloader_folds = create_dataloaders_kfolds(
        dir=train_config["directory"],
        dataset=args.train_dataset,
        num_folds=5,
        val_ratio=0.2,
        batch_size=16,
        num_workers=os.cpu_count() // 2,
        multiplier=train_config.get("multiplier", 1),
        downsampling_rate=train_config.get("downsampling_rate", 1),
    )
    logger.info("Dataloaders for cross-validation created.")

    # Initialize Optuna study
    study = optuna.create_study(direction="maximize")
    logger.info("Optuna study created. Starting WatchSleepNet optimization...")

    # Run only 5 trials for demonstration
    n_trials = 5

    # Optimize the objective function
    study.optimize(
        lambda trial: objective(
            trial=trial,
            train_config=train_config,
            dataloader_folds=dataloader_folds,
            device=device,
            model_name="watchsleepnet",
        ),
        n_trials=n_trials,
        timeout=None,  # or set a desired timeout in seconds
    )

    # Print best trial results
    logger.info("Optimization completed.")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value (Overall Kappa): {best_trial.value}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

if __name__ == "__main__":
    main()
