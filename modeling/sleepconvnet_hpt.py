import os
import argparse
import warnings
import logging
import random

import torch
import numpy as np
import optuna
from optuna.exceptions import TrialPruned

from data_setup import create_dataloaders_kfolds
from models.sleepconvnet import SleepConvNet
from config import SleepConvNetConfig, dataset_configurations
from engine import train_cross_validate_hpo, train_cross_validate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 0):
    """
    Set seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed} for reproducibility.")


def get_device() -> torch.device:
    """
    Determine the computation device (CPU or CUDA).

    Returns:
        torch.device: The device to use.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device set to: {device}")
    return device


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for SleepConvNet")

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


def initialize_model(model_params: dict, device: torch.device) -> SleepConvNet:
    """
    Initialize the SleepConvNet model with given parameters.

    Args:
        model_params (dict): Configuration parameters for SleepConvNet.
        device (torch.device): Computation device.

    Returns:
        SleepConvNet: Initialized SleepConvNet model.
    """
    model = SleepConvNet(
        input_size=model_params['input_size'],
        target_size=model_params['target_size'],
        num_segments=model_params['num_segments'],
        num_classes=model_params['num_classes'],
        dropout_rate=model_params['dropout_rate'],
        conv_layers_configs=model_params['conv_layers_configs'],
        dilation_layers_configs=model_params['dilation_layers_configs'],
    ).to(device)
    logger.info("Initialized SleepConvNet model.")
    return model

def objective(trial, config: dict, dataloader_folds: list, device: torch.device, train_config: dict, model_name: str):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        config (dict): Configuration parameters.
        dataloader_folds (list): List of dataloader tuples for cross-validation.
        device (torch.device): Computation device.
        train_config (dict): Training configuration.
        model_name (str): Name of the model architecture.

    Returns:
        float: The metric to maximize (e.g., Cohen's Kappa).
    """
    dropout_rate = 0.2  # Can be tuned as well

    # Choose conv_layers_configs
    conv_layers_configs = trial.suggest_categorical(
        "conv_layers_configs",
        [
            [(1, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)],  # final_in_channels=128
            [(1, 32, 3, 1), (32, 48, 3, 1), (48, 64, 3, 1)],  # final_in_channels=64
            [(1, 8, 3, 1), (8, 16, 3, 1), (16, 32, 3, 1)],  # final_in_channels=32
        ],
    )

    final_in_channels = conv_layers_configs[-1][1]

    # Choose dilation_layers_configs deterministically based on final_in_channels
    if final_in_channels == 128:
        dilation_layers_configs = [(128, 128, 7, d) for d in [2, 4, 8, 16, 32]]
    elif final_in_channels == 64:
        dilation_layers_configs = [(64, 64, 7, d) for d in [2, 4, 8]]
    elif final_in_channels == 32:
        dilation_layers_configs = [(32, 32, 7, d) for d in [2, 4]]

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    def model_init():
        return SleepConvNet(
            input_size=750,
            target_size=256,
            num_segments=1100,
            num_classes=3,
            dropout_rate=dropout_rate,
            conv_layers_configs=conv_layers_configs,
            dilation_layers_configs=dilation_layers_configs,
        ).to(device)

    try:
        # Run cross-validation with the suggested hyperparameters
        results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = (
            train_cross_validate_hpo(
                model_init=model_init,
                dataloader_folds=dataloader_folds,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_fn=config['LOSS_FN'],
                num_epochs=config['NUM_EPOCHS'],
                patience=config['PATIENCE'],
                device=device,
                checkpoint_path=train_config["get_model_save_path"](
                    model_name=model_name,
                    dataset_name=train_config["dataset"],
                    version="optuna_trial",
                ),
            )
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise TrialPruned()

    # Use overall_kappa as the metric to maximize
    logger.info(f"Trial completed with overall_kappa: {overall_kappa}")
    return overall_kappa

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(seed=0)
    device = get_device()
    args = parse_arguments()

    # Retrieve configuration based on the dataset argument
    train_config = dataset_configurations.get(args.train_dataset)
    if train_config is None:
        logger.error(f"Configuration for dataset '{args.train_dataset}' not found.")
        raise ValueError(f"Configuration for dataset '{args.train_dataset}' not found.")

    model_config = SleepConvNetConfig.to_dict()

    # Create folds for cross-validation
    dataloader_folds = create_dataloaders_kfolds(
        dir=train_config["directory"],
        dataset=args.train_dataset,
        num_folds=5,
        val_ratio=0.2,
        batch_size=model_config.get('BATCH_SIZE', 16),
        num_workers=os.cpu_count() // 2,
        multiplier=train_config.get("multiplier", 1),
        downsampling_rate=train_config.get("downsampling_rate", 1),
    )

    logger.info("Dataloaders for cross-validation created.")

    # Initialize Optuna study
    study = optuna.create_study(direction="maximize")
    logger.info("Optuna study created. Starting optimization...")

    # Optimize the objective function
    study.optimize(
        lambda trial: objective(
            trial=trial,
            config=model_config,
            dataloader_folds=dataloader_folds,
            device=device,
            train_config=train_config,
            model_name="sleepconvnet",
        ),
        n_trials=50,  # Adjust the number of trials as needed
        timeout=3600,  # Optional: Set a timeout in seconds
    )

    # Print best trial results
    logger.info("Optimization completed.")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (Overall Kappa): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

if __name__ == "__main__":
    main()
