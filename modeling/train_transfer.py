import os
import random
import argparse
import warnings
import logging

import torch
import numpy as np

from data_setup import create_dataloaders, create_dataloaders_kfolds
from models.watchsleepnet import WatchSleepNet
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    dataset_configurations,
)
from engine import (
    train,
    train_and_evaluate,
    validate_step,
    train_ablate_evaluate,
    setup_model_and_optimizer,
    run_training_epochs,
    test_step,
    compute_metrics,
    compute_metrics_per_ahi_category,
)

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
    parser = argparse.ArgumentParser(description="Train and Finetune Sleep Classification Models")
    
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
        "--test_dataset",
        type=str,
        default="dreamt_pibi",
        choices=[
            "dreamt_pibi"
        ],
        help="Dataset to test on."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sleep_staging",
        choices=["sleep_staging", "sleep_wake"],
        help="Task to perform."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="watchsleepnet",
        choices=["watchsleepnet", "insightsleepnet", "sleepconvnet"],
        help="Model architecture to use."
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Flag to load and test a pre-trained model."
    )
    
    # New arguments for toggling TCN and Attention
    parser.add_argument(
        "--use_tcn",
        action="store_true",
        help="Enable TCN module."
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Enable Attention module."
    )
    
    # Set default values for ablation flags
    parser.set_defaults(use_tcn=False, use_attention=False)
    
    args = parser.parse_args()
    
    # Validate that train and test datasets are different
    if args.train_dataset == args.test_dataset:
        parser.error("Train and test datasets must be different for pretraining + finetuning setup.")
    
    logger.info("Command-line arguments parsed successfully.")
    return args


def initialize_model(model_name: str, config: dict, device: torch.device) -> torch.nn.Module:
    """
    Initialize the model based on the selected architecture.

    Args:
        model_name (str): Name of the model architecture.
        config (dict): Configuration parameters for the model.
        device (torch.device): Computation device.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if model_name == "watchsleepnet":
        model = WatchSleepNet(
            num_features=config['num_features'],
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            tcn_layers=config['tcn_layers'],
            num_classes=config['num_classes'],
        ).to(device)
        logger.info("Initialized WatchSleepNet model.")
    
    elif model_name == "insightsleepnet":
        model = InsightSleepNet(
            input_size=config['input_size'], 
            output_size=config['output_size']
        ).to(device)
        logger.info("Initialized InsightSleepNet model.")
    
    elif model_name == "sleepconvnet":
        model = SleepConvNet().to(device)
        logger.info("Initialized SleepConvNet model.")
    
    else:
        logger.error(f"Model '{model_name}' is not recognized.")
        raise ValueError(f"Model '{model_name}' is not recognized.")
    
    return model


def train_model(model: torch.nn.Module, config: dict, dataloaders: tuple, model_save_path: str, device: torch.device):
    """
    Train the model from scratch.

    Args:
        model (torch.nn.Module): The model to train.
        config (dict): Training configuration parameters.
        dataloaders (tuple): Tuple containing (train_dataloader, val_dataloader, test_dataloader).
        model_save_path (str): Path to save the trained model.
        device (torch.device): Computation device.
    """
    train_dl, val_dl, _ = dataloaders
    
    # Define loss function and optimizer
    loss_fn = config['LOSS_FN']
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['LEARNING_RATE'], 
        weight_decay=config['WEIGHT_DECAY']
    )
    logger.info("Defined loss function and optimizer.")
    
    # Train the model
    logger.info("Starting training from scratch.")
    training_logs = train(
        model_name=args.model,
        model_params=config,
        num_classes=config['num_classes'],
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        loss_fn=loss_fn,
        num_epochs=config['NUM_EPOCHS'],
        patience=config['PATIENCE'],
        device=device,
        model_save_path=model_save_path,
        saved_model_path=None,
        learning_rate=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY'],
        freeze_layers=False,
        use_tcn=config.get('use_tcn', False),
        use_attention=config.get('use_attention', False),
    )
    logger.info("Training completed.")
    
    # Load the best model and perform final validation
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        logger.info(f"Loaded the best model from {model_save_path} for final validation.")
    else:
        logger.warning(f"No checkpoint found at {model_save_path}. Using the current model state.")
    
    val_metrics = validate_step(model, val_dl, loss_fn, device)
    logger.info("\n" + "=" * 80)
    logger.info("Best Pretraining Validation Results")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {val_metrics[1]:.3f}")
    logger.info(f"Macro F1 Score: {val_metrics[2]:.3f}")
    logger.info(f"REM F1 Score: {val_metrics[4]:.3f}")
    logger.info(f"AUROC: {val_metrics[5]:.3f}")
    logger.info(f"Cohen's Kappa: {val_metrics[3]:.3f}")


def finetune_model(model: torch.nn.Module, config: dict, dataloader_folds: list, model_save_path: str, device: torch.device, use_tcn: bool, use_attention: bool):
    """
    Finetune the pre-trained model using transfer learning with cross-validation.

    Args:
        model (torch.nn.Module): The pre-trained model to finetune.
        config (dict): Finetuning configuration parameters.
        dataloader_folds (list): List of dataloader tuples for cross-validation.
        model_save_path (str): Path to save the finetuned model.
        device (torch.device): Computation device.
        use_tcn (bool): Flag to enable TCN module.
        use_attention (bool): Flag to enable Attention module.
    """
    loss_fn = config['LOSS_FN']
    
    if config.get('use_tcn') is None:
        config['use_tcn'] = use_tcn
    if config.get('use_attention') is None:
        config['use_attention'] = use_attention
    
    logger.info(f"Starting finetuning with use_tcn={use_tcn} and use_attention={use_attention}.")
    
    if args.model == "watchsleepnet":
        results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_ablate_evaluate(
            model_name=args.model,
            use_tcn=use_tcn,
            use_attention=use_attention,
            model_params=config,
            num_classes=config['num_classes'],
            dataloader_folds=dataloader_folds,
            saved_model_path=model_save_path,
            learning_rate=config['LEARNING_RATE'],
            weight_decay=config['WEIGHT_DECAY'],
            loss_fn=loss_fn,
            num_epochs=config['NUM_EPOCHS'],
            patience=config['PATIENCE'],
            device=device,
            checkpoint_path=model_save_path.replace(".pt", "_finetuned.pt"),
        )
        logger.info("Finetuning completed.")
        logger.info("Cross-Validation Results:")
        logger.info(f"Overall Accuracy: {overall_acc:.3f}")
        logger.info(f"Overall F1 Score: {overall_f1:.3f}")
        logger.info(f"Overall Kappa: {overall_kappa:.3f}")
        logger.info(f"REM F1 Score: {rem_f1:.3f}")
        logger.info(f"AUROC: {auroc:.3f}")
    else:
        logger.warning("Finetuning is not implemented for the selected model.")
        print("Not Implemented Yet.")


# ----------------------- Main Function -----------------------

def main():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set seed for reproducibility
    set_seed(seed=0)
    
    # Determine computation device
    device = get_device()
    
    # Parse command-line arguments
    global args  # Declare args as global to use in train_model
    args = parse_arguments()
    
    # Retrieve configuration based on the dataset argument
    train_config_class = dataset_configurations.get(args.train_dataset)
    test_config_class = dataset_configurations.get(args.test_dataset)
    
    if train_config_class is None or test_config_class is None:
        logger.error("Configuration for the provided datasets could not be found.")
        raise ValueError("Invalid dataset names provided.")
    
    # Initialize model configuration
    if args.model == "watchsleepnet":
        model_config_class = WatchSleepNetConfig
    elif args.model == "insightsleepnet":
        model_config_class = InsightSleepNetConfig
    elif args.model == "sleepconvnet":
        model_config_class = SleepConvNetConfig
    else:
        logger.error(f"Model '{args.model}' is not recognized.")
        raise ValueError(f"Model '{args.model}' is not recognized.")
    
    # Convert model configuration to dictionary
    model_config = model_config_class.to_dict()
    
    # Initialize the model
    model = initialize_model(args.model, model_config, device)
    
    # Generate dynamic save path for model and analysis
    model_save_path = train_config_class["get_model_save_path"](
        model_name=args.model, 
        dataset_name=args.train_dataset, 
        version="ablate_transfer"
    )
    
    logger.info(f"Model Save Path: {model_save_path}")
    
    # Load pre-trained model if --testing flag is set
    if args.testing:
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            logger.info(f"Loaded pre-trained model from: {model_save_path}")
        else:
            logger.error(f"Pre-trained model not found at: {model_save_path}")
            raise FileNotFoundError(f"Pre-trained model not found at: {model_save_path}")
    else:
        logger.info(f"Training model from scratch. Saving to: {model_save_path}")
        
        # Create dataloaders for training
        train_dataloader, val_dataloader, _ = create_dataloaders(
            dir=train_config_class["directory"],
            train_ratio=0.8,
            val_ratio=0.2,
            batch_size=model_config.get('BATCH_SIZE', 16),
            num_workers=os.cpu_count() // 2,
            dataset=args.train_dataset,
            multiplier=train_config_class.get("multiplier", 1),
            downsampling_rate=train_config_class.get("downsampling_rate", 1),
            task=args.task,
        )
        
        logger.info("Dataloaders for training and validation created.")
        
        # Train the model
        train_model(
            model=model,
            config=model_config,
            dataloaders=(train_dataloader, val_dataloader, None),
            model_save_path=model_save_path,
            device=device
        )
    
    # Perform finetuning only if not in testing mode
    if not args.testing:
        logger.info(f"Starting finetuning on dataset: {args.test_dataset}")
        
        # Create dataloaders with cross-validation for finetuning
        dataloader_folds = create_dataloaders_kfolds(
            dir=test_config_class["directory"],
            dataset=args.test_dataset,
            num_folds=5,
            val_ratio=0.2,
            batch_size=model_config.get('BATCH_SIZE', 16),
            num_workers=os.cpu_count() // 2,
            multiplier=test_config_class.get("multiplier", 1),
            downsampling_rate=test_config_class.get("downsampling_rate", 1),
        )
        
        logger.info("Dataloaders for cross-validation finetuning created.")
        
        # Define finetuning model save path
        finetune_save_path = test_config_class["get_model_save_path"](
            model_name=args.model, 
            dataset_name=args.test_dataset, 
            version="ablate_transfer"
        )
        
        logger.info(f"Finetune model save path: {finetune_save_path}")
        
        # Perform finetuning
        finetune_model(
            model=model,
            config=model_config,
            dataloader_folds=dataloader_folds,
            model_save_path=finetune_save_path,
            device=device,
            use_tcn=args.use_tcn,
            use_attention=args.use_attention
        )
    else:
        logger.info("Testing mode enabled. Finetuning skipped.")

# ----------------------- Entry Point -----------------------

if __name__ == "__main__":
    main()
