import os
import torch
import argparse
from data_setup import create_dataloaders, create_dataloaders_kfolds
from config import WatchSleepNetConfig, dataset_configurations
from engine import train_cross_validate
import random
import numpy as np
from models.watchsleepnet import WatchSleepNet

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dataset",
    type=str,
    default="shhs_mesa_ibi",
    choices=[
        "shhs_ibi",
        "mesa_ppg",
        "mesa_pibi",
        "shhs_mesa_ibi",
    ],
)

parser.add_argument(
    "--task", type=str, default="sleep_staging", choices=["sleep_staging", "sleep_wake"]
)
parser.add_argument(
    "--model",
    type=str,
    default="watchsleepnet", 
    choices=["watchsleepnet"],
)

# New argument for toggling TCN and Attention
# Use store_true to enable features that are disabled by default
parser.add_argument(
    "--use_tcn", dest="use_tcn", action="store_true", help="Enable TCN module."
)
parser.add_argument(
    "--use_attention",
    dest="use_attention",
    action="store_true",
    help="Enable Attention module.",
)

parser.set_defaults(use_tcn=False, use_attention=False)

args = parser.parse_args()

# Retrieve configuration based on the dataset argument
train_config = dataset_configurations.get(args.train_dataset, None)

model_config = WatchSleepNetConfig.to_dict()
print("USE TCN: ", args.use_tcn)
print("USE ATTENTION: ", args.use_attention)

model = WatchSleepNet(
        num_features=model_config['num_features'],
        num_channels=model_config['num_channels'],
        kernel_size=model_config['kernel_size'],
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        tcn_layers=model_config['tcn_layers'],
        num_classes=model_config['num_classes'],
        use_tcn=args.use_tcn,
        use_attention=args.use_attention,
    ).to(DEVICE)

# Print the model architecture
print("Model Architecture:")
print(model)

# Function to count total parameters and total size
def count_parameters_and_size(model):
    total_params = 0
    total_size = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
    return total_params, total_size


# Calculate total parameters and size
total_params, total_size = count_parameters_and_size(model)
print(f"\nTotal trainable parameters: {total_params}")
print(f"Total model size: {total_size / (1024 ** 2):.2f} MB")  # Convert bytes to MB

# Print detailed information for each parameter
print("\nModel Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        param_size = param.numel() * param.element_size() / (1024**2)  # Size in MB
        print(f"{name}: {param.numel()} parameters, Size: {param_size:.4f} MB")

# Generate dynamic save path for model and analysis
model_save_path = dataset_configurations[args.train_dataset]["get_model_save_path"](
    model_name=args.model, dataset_name=args.train_dataset, version="cv_ablate_{}_tcn_{}_att".format(args.use_tcn, args.use_attention)
)

print(model_save_path)

print("Perform cross-validation on:", args.train_dataset)
dataloader_folds = create_dataloaders_kfolds(
    dir=train_config["directory"],
    dataset=args.train_dataset,
    num_folds=5,
    val_ratio=0.2,
    batch_size=model_config["BATCH_SIZE"],
    num_workers=NUM_WORKERS,
    multiplier=train_config["multiplier"],
    downsampling_rate=train_config["downsampling_rate"],
)

loss_fn = model_config['LOSS_FN']

model_config['use_attention'] = args.use_attention
model_config['use_tcn'] = args.use_tcn
print(args.use_attention, args.use_tcn)
print(model_config)

results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_cross_validate(
    model_name=args.model,
    model_params=model_config,  # Contains 'use_attention', 'use_tcn', 'num_classes', etc.
    dataloader_folds=dataloader_folds,
    saved_model_path=None,      # if training from scratch
    learning_rate=model_config.get('LEARNING_RATE', 1e-3),
    weight_decay=model_config.get('WEIGHT_DECAY', 1e-4),
    loss_fn=loss_fn,
    num_epochs=model_config.get('NUM_EPOCHS', 100),
    patience=model_config.get('PATIENCE', 10),
    device=DEVICE,
    checkpoint_path=train_config["get_model_save_path"](
        model_name=args.model,
        dataset_name=args.train_dataset, 
        version="cv_no_transfer_ablate_{}_tcn_{}_att".format(args.use_tcn, args.use_attention)
    ),
    freeze_layers=False, 
)

# Output the results
print(f"Results for use_tcn={args.use_tcn}, use_attention={args.use_attention}")
print(f"Overall Accuracy: {overall_acc}")
print(f"Overall F1 Score: {overall_f1}")
print(f"Overall Kappa: {overall_kappa}")
print(f"REM F1 Score: {rem_f1}")
print(f"AUROC: {auroc}")
