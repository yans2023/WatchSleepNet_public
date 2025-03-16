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
from models.sleepppgnet import SleepPPGNet
from config import SleepPPGNetConfig, dataset_configurations
from engine import train_cross_validate_hpo, train_cross_validate

# ----------------------- Logging Configuration -----------------------

# Configure logging to output to console with a specific format
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

    Args:
        seed (int): The seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
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
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for SleepPPGNet")

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


def initialize_model(model_params: dict, device: torch.device) -> SleepPPGNet:
    """
    Initialize the SleepPPGNet model with given parameters.

    Args:
        model_params (dict): Configuration parameters for SleepPPGNet.
        device (torch.device): Computation device.

    Returns:
        SleepPPGNet: Initialized SleepPPGNet model.
    """
    model = SleepPPGNet(
        input_channels=model_params['input_channels'],
        num_classes=model_params['num_classes'],
        num_res_blocks=model_params['num_res_blocks'],
        tcn_layers=model_params['tcn_layers'],
        hidden_dim=model_params['hidden_dim'],
        dropout_rate=model_params['dropout_rate'],
    ).to(device)
    logger.info("Initialized SleepPPGNet model.")
    return model


# ----------------------- Objective Function -----------------------

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
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    num_res_blocks = trial.suggest_categorical("num_res_blocks", [2, 4, 6])
    tcn_layers = trial.suggest_categorical("tcn_layers", [1, 2])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    
    # Learning rate and weight decay
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # Memory debugging information
    logger.info(f"Trial config: batch_size={batch_size}, hidden_dim={hidden_dim}, "
                f"num_res_blocks={num_res_blocks}, tcn_layers={tcn_layers}")
    
    # Update dataloader batch size
    for i in range(len(dataloader_folds)):
        train_loader, val_loader, test_loader = dataloader_folds[i]
        # Recreate dataloaders with new batch size if different from original
        if train_loader.batch_size != batch_size:
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset
            
            new_train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=True
            )
            new_val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=val_loader.num_workers,
                pin_memory=True
            )
            new_test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=test_loader.num_workers,
                pin_memory=True
            )
            dataloader_folds[i] = (new_train_loader, new_val_loader, new_test_loader)

    def model_init():
        return SleepPPGNet(
            input_channels=1,
            num_classes=3,
            num_res_blocks=num_res_blocks,
            tcn_layers=tcn_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
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
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"CUDA OOM error with batch_size={batch_size}, hidden_dim={hidden_dim}, "
                          f"num_res_blocks={num_res_blocks}, tcn_layers={tcn_layers}")
            torch.cuda.empty_cache()
            raise TrialPruned()
        else:
            logger.error(f"An error occurred during training: {e}")
            raise TrialPruned()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise TrialPruned()

    # Use overall_kappa as the metric to maximize
    logger.info(f"Trial completed with overall_kappa: {overall_kappa}")
    return overall_kappa


# ----------------------- Main Function -----------------------

def main():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set seed for reproducibility
    set_seed(seed=0)

    # Determine computation device
    device = get_device()

    # Parse command-line arguments
    args = parse_arguments()

    # Retrieve configuration based on the dataset argument
    train_config = dataset_configurations.get(args.train_dataset)
    if train_config is None:
        logger.error(f"Configuration for dataset '{args.train_dataset}' not found.")
        raise ValueError(f"Configuration for dataset '{args.train_dataset}' not found.")

    # Add dataset name to train_config for use in objective function
    train_config["dataset"] = args.train_dataset

    # Convert model configuration to dictionary
    model_config = SleepPPGNetConfig.to_dict()

    # Create folds for cross-validation
    dataloader_folds = create_dataloaders_kfolds(
        dir=train_config["directory"],
        dataset=args.train_dataset,
        num_folds=5,
        val_ratio=0.2,
        batch_size=model_config.get('BATCH_SIZE', 1),  # Start with small batch size
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
            model_name="sleepppgnet",
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

    # Update SleepPPGNetConfig with best parameters
    logger.info("Best parameters for SleepPPGNet:")
    logger.info(f"batch_size = {trial.params.get('batch_size', 1)}")
    logger.info(f"hidden_dim = {trial.params.get('hidden_dim', 32)}")
    logger.info(f"num_res_blocks = {trial.params.get('num_res_blocks', 4)}")
    logger.info(f"tcn_layers = {trial.params.get('tcn_layers', 1)}")
    logger.info(f"dropout_rate = {trial.params.get('dropout_rate', 0.2)}")
    logger.info(f"learning_rate = {trial.params.get('learning_rate', 1e-4)}")
    logger.info(f"weight_decay = {trial.params.get('weight_decay', 1e-3)}")

    os.makedirs("optuna_studies", exist_ok=True)
    study.trials_dataframe().to_csv("optuna_studies/sleepppgnet_hpt_results.csv", index=False)
    logger.info("Optuna study results saved to 'optuna_studies/sleepppgnet_hpt_results.csv'.")


if __name__ == "__main__":
    main()