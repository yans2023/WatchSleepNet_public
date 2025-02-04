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
from engine import train_cross_validate_hpo 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 0):

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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device set to: {device}")
    return device


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
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


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """

    num_blocks = trial.suggest_categorical("num_blocks", [4, 5, 6])

    last_layers_size = trial.suggest_categorical("last_layers_size", ["small", "big"])

    base_block = {
        "kernel_sizes": [9, 19, 39],
        "bottleneck_channels": 16,
        "use_residual": True,
    }

    block_configs = []
    in_ch = 32  
    for b_idx in range(num_blocks):
        # Decide how big "n_filters" is for the block
        if b_idx < num_blocks - 1:
            # Middle blocks
            n_filters = 16 if last_layers_size == "small" else 32
        else:
            # The last block
            n_filters = 32 if last_layers_size == "small" else 64

        # Create a copy
        block_dict = {
            "in_channels": in_ch,
            "n_filters": n_filters,
            "kernel_sizes": base_block["kernel_sizes"],
            "bottleneck_channels": base_block["bottleneck_channels"],
            "use_residual": base_block["use_residual"],
        }
        block_configs.append(block_dict)

        in_ch = 4 * n_filters

    def model_init():
        from models.insightsleepnet import InsightSleepNet
        model = InsightSleepNet(
            input_size=750,    # or  your dynamic "input_size" from config
            output_size=3,     # or 2 if "sleep_wake"
            block_configs=block_configs,
            # Possibly also set an "initial_conv_out=32" or so
            initial_conv_out=32,
            dropout_rate=0.2,  # e.g. fixed
            activation=nn.ReLU(),
        )
        return model

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

    return overall_kappa


def main():
    # 1) Set seed for reproducibility
    set_seed(seed=0)

    # 2) Determine computation device
    global DEVICE
    DEVICE = get_device()

    # 3) Parse command-line arguments
    global args
    args = parse_arguments()

    # 4) Retrieve configuration for dataset
    global train_config
    train_config = dataset_configurations.get(args.train_dataset, None)
    if train_config is None:
        logger.error(f"No config found for dataset '{args.train_dataset}'.")
        raise ValueError(f"No config found for dataset '{args.train_dataset}'.")

    # 5) Initialize model configuration
    global model_config
    model_config = InsightSleepNetConfig

    # 6) Create folds for cross-validation
    global dataloader_folds
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

    # 7) Retrieve loss function
    global loss_fn
    loss_fn = model_config.LOSS_FN

    # 8) Run Optuna study
    study = optuna.create_study(direction="maximize")
    logger.info("Optuna study created. Starting optimization...")

    study.optimize(objective, n_trials=5)  # e.g., small # of trials

    # 9) Log best trial
    logger.info("Optimization completed.")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (Kappa): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    os.makedirs("optuna_studies", exist_ok=True)
    study.trials_dataframe().to_csv("optuna_studies/insightsleepnet_hpo_results.csv", index=False)
    logger.info("Optuna study results saved to 'optuna_studies/insightsleepnet_hpo_results.csv'.")

if __name__ == "__main__":
    main()
