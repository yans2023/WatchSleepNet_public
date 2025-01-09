import os
import argparse
import warnings
import logging
import random

import torch
import numpy as np
import torch.nn as nn
import optuna
from optuna.exceptions import TrialPruned

# ---------------------
# Environment Variables and Warning Filters
# ---------------------
os.environ["TORCHDYNAMO_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------
# Your Custom Modules
# ---------------------
from data_setup import create_dataloaders_kfolds
from config import (
    InsightSleepNetConfig,
    dataset_configurations,
)
from engine import train_cross_validate_hpo  # Ensure this import references the correct function

# ---------------------
# Logging Configuration
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------
# Seed Settings
# ---------------------
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

# ---------------------
# Device Configuration
# ---------------------
def get_device() -> torch.device:
    """
    Determine the computation device (CPU or CUDA).

    Returns:
        torch.device: The device to use.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device set to: {device}")
    return device

# ---------------------
# Argument Parsing
# ---------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for InsightSleepNet")

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

    logger.info("Command-line arguments parsed successfully.")
    return args

# ---------------------
# Objective Function for Optuna
# ---------------------
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: The metric to maximize (e.g., Cohen's Kappa).
    """
    # 1) Sample hyperparameters
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
    dropout_rate = 0.2  # Fixed as per current setup; uncomment to tune
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)

    # 2) Define model_init to create an InsightSleepNet with the chosen hyperparameters
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

    try:
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
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise TrialPruned()

    # Return the metric you want to maximize; let's say overall_kappa
    return overall_kappa

# ---------------------
# Main Function
# ---------------------
def main():
    # Set seed for reproducibility
    set_seed(seed=0)

    # Determine computation device
    global DEVICE  # To ensure DEVICE is accessible inside objective
    DEVICE = get_device()

    # Parse command-line arguments
    global args  # To ensure args is accessible inside objective
    args = parse_arguments()

    # Retrieve configuration based on the dataset argument
    global train_config  # To ensure train_config is accessible inside objective
    train_config = dataset_configurations.get(args.train_dataset, None)
    if train_config is None:
        logger.error(f"No config found for dataset '{args.train_dataset}'.")
        raise ValueError(f"No config found for dataset '{args.train_dataset}'.")

    # Initialize model configuration
    global model_config  # To ensure model_config is accessible inside objective
    model_config = InsightSleepNetConfig

    # Create folds for cross-validation
    global dataloader_folds  # To ensure dataloader_folds is accessible inside objective
    dataloader_folds = create_dataloaders_kfolds(
        dir=train_config["directory"],
        dataset=args.train_dataset,
        num_folds=5,
        val_ratio=0.2,
        batch_size=model_config.BATCH_SIZE,
        num_workers=os.cpu_count() // 2,
        multiplier=train_config["multiplier"],
        downsampling_rate=train_config["downsampling_rate"],
    )

    logger.info("Dataloaders for cross-validation created.")

    # Retrieve loss function
    global loss_fn  # To ensure loss_fn is accessible inside objective
    loss_fn = model_config.LOSS_FN

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    logger.info("Optuna study created. Starting optimization...")

    study.optimize(objective, n_trials=10)

    # Log best trial
    logger.info("Optimization completed.")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (Kappa): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Optionally, save the study results to a file
    os.makedirs("optuna_studies", exist_ok=True)
    study.trials_dataframe().to_csv("optuna_studies/insightsleepnet_hpo_results.csv", index=False)
    logger.info("Optuna study results saved to 'optuna_studies/insightsleepnet_hpo_results.csv'.")

# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":
    main()
