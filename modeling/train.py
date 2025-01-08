import os
import torch
import argparse
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Import your custom modules
from data_setup import create_dataloaders
from models.watchsleepnet import WatchSleepNet
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    dataset_configurations,
)
from engine import train, validate_step  # Updated import to 'engine_new'
from utils import print_model_info

# --- Reproducibility ---
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- System Settings ---
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Command-line Arguments ---
parser = argparse.ArgumentParser(description="Train and evaluate sleep classification models.")
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
    help="Which dataset to train on (single split, not cross-validation)."
)

parser.add_argument(
    "--task", 
    type=str, 
    default="sleep_staging", 
    choices=["sleep_staging", "sleep_wake"],
    help="Classification task: either sleep staging or sleep/wake."
)

parser.add_argument(
    "--model",
    type=str,
    default="watchsleepnet",
    choices=["watchsleepnet", "insightsleepnet", "sleepconvnet"],
    help="Which model architecture to train."
)

args = parser.parse_args()

# --- Retrieve dataset config based on --train_dataset ---
train_config = dataset_configurations.get(args.train_dataset, None)
if train_config is None:
    raise ValueError(f"No config found for dataset '{args.train_dataset}'.")

# --- Pick the appropriate model config based on --model ---
if args.model == "watchsleepnet":
    model_config = WatchSleepNetConfig
elif args.model == "insightsleepnet":
    model_config = InsightSleepNetConfig
elif args.model == "sleepconvnet":
    model_config = SleepConvNetConfig
else:
    raise ValueError("Model not recognized.")

# --- Convert model_config to a dictionary ---
# Assuming each Config class has a 'to_dict()' method.
# If not, you'll need to manually construct the dictionary.
# Example implementation of to_dict() is shown below.

def config_to_dict(config):
    """
    Converts a Config object to a dictionary.
    Assumes that all relevant attributes are defined and should be included.
    """
    # List all attributes that need to be included in model_params
    # Modify this list based on your actual Config classes
    params = {
        "num_features": getattr(config, "NUM_INPUT_CHANNELS", None),
        "input_size": getattr(config, "INPUT_SIZE", None),
        "output_size": getattr(config, "OUTPUT_SIZE", None),
        "num_channels": getattr(config, "NUM_CHANNELS", None),
        "kernel_size": getattr(config, "KERNEL_SIZE", None),
        "hidden_dim": getattr(config, "HIDDEN_DIM", None),
        "num_heads": getattr(config, "NUM_HEADS", None),
        "num_layers": getattr(config, "NUM_LAYERS", None),
        "tcn_layers": getattr(config, "TCN_LAYERS", None),
        "num_classes": getattr(config, "NUM_CLASSES", None),
        "n_filters": getattr(config, "N_FILTERS", None),
        "bottleneck_channels": getattr(config, "BOTTLENECK_CHANNELS", None),
        "kernel_sizes": getattr(config, "KERNEL_SIZES", None),
        "num_inception_blocks": getattr(config, "NUM_INCEPTION_BLOCKS", None),
        "use_residual": getattr(config, "USE_RESIDUAL", None),
        "dropout_rate": getattr(config, "DROPOUT_RATE", None),
        # Add other parameters as needed
    }
    # Remove keys with None values
    return {k: v for k, v in params.items() if v is not None}

model_params = config_to_dict(model_config)

# --- Construct a path to save the model checkpoint ---
model_save_path = train_config["get_model_save_path"](
    model_name=args.model, dataset_name=args.train_dataset, version="single_split"
)

print("Model checkpoint will be saved at:", model_save_path)

# --- Create a single train/val/test split (not cross-validation) ---
# For example: train_ratio = 0.7, val_ratio = 0.2, test_ratio = 0.1
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 1.0 - train_ratio - val_ratio
assert test_ratio > 0, "Train + Val ratio must be < 1.0"

print(f"Creating single-split dataloaders with ratios {train_ratio}/{val_ratio}/{test_ratio}...")

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    dir=train_config["directory"],
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=model_config.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    dataset=args.train_dataset,
    multiplier=train_config["multiplier"],
    downsampling_rate=train_config["downsampling_rate"],
    task=args.task,
)

# --- Define the loss function ---
loss_fn = model_config.LOSS_FN

# --- Train the model (uses your refactored 'train' function) ---
print(f"\nTraining {args.model} on {args.train_dataset}, task={args.task}.")

training_logs = train(
    model_name=args.model,
    model_params=model_params,  # Pass the dictionary
    # Removed num_classes, use_tcn, use_attention
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    num_epochs=model_config.NUM_EPOCHS,
    patience=model_config.PATIENCE,
    device=DEVICE,
    model_save_path=model_save_path,
    saved_model_path=None,  # Adjust if loading from a checkpoint is needed
    learning_rate=model_config.LEARNING_RATE,
    weight_decay=model_config.WEIGHT_DECAY,
    freeze_layers=False,
    # Removed use_tcn and use_attention
)

print("\nTraining complete. Attempting final test evaluation...")

# --- Re-instantiate the same model architecture & load the best checkpoint ---
final_model_path = model_save_path
if os.path.exists(final_model_path):
    print("Loading the best model from checkpoint for final evaluation.")
    # Re-instantiate the appropriate model based on args.model
    if args.model == "watchsleepnet":
        from models.watchsleepnet import WatchSleepNet
        final_model = WatchSleepNet(
            num_features=model_config.NUM_INPUT_CHANNELS,
            # feature_channels=model_config.FEATURE_CHANNELS,  # Uncomment if applicable
            num_channels=model_config.NUM_CHANNELS,
            kernel_size=model_config.KERNEL_SIZE,
            hidden_dim=model_config.HIDDEN_DIM,
            num_heads=model_config.NUM_HEADS,
            num_layers=model_config.NUM_LAYERS,
            tcn_layers=model_config.TCN_LAYERS,
            num_classes=model_config.NUM_CLASSES,
            # Include other parameters as needed
        ).to(DEVICE)

    elif args.model == "insightsleepnet":
        from models.insightsleepnet import InsightSleepNet
        final_model = InsightSleepNet(
            input_size=model_config.INPUT_SIZE,
            output_size=model_config.OUTPUT_SIZE,
            # Include other parameters as needed
        ).to(DEVICE)

    elif args.model == "sleepconvnet":
        from models.sleepconvnet import SleepConvNet
        final_model = SleepConvNet(
            # Initialize with necessary parameters
        ).to(DEVICE)

    else:
        raise ValueError("Unknown model type, can't re-instantiate.")

    # Load the best checkpoint
    final_model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    final_model.eval()

    # Now do final test evaluation
    test_loss, test_acc, test_f1, test_kappa, test_rem_f1, test_auroc = validate_step(
        model=final_model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=DEVICE,
        task=args.task,  # Pass the correct task for staging or wake
    )

    print(
        f"\n=== Final Test Metrics ===\n"
        f"Loss={test_loss:.4f}, Accuracy={test_acc:.3f}, F1 Score={test_f1:.3f}, "
        f"Kappa={test_kappa:.3f}, REM F1={test_rem_f1:.3f}, AUROC={test_auroc:.3f}"
    )
else:
    print("No best checkpoint found on disk—could not do final test evaluation.")
