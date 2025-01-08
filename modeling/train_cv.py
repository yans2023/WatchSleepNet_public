import os
import torch
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from data_setup import create_dataloaders, create_dataloaders_kfolds
from models.watchsleepnet import WatchSleepNet  # Updated model name
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    dataset_configurations,
)
from engine import train, train_and_evaluate, validate_step, train_cross_validate
from utils import print_model_info
import random
import numpy as np

# Seed settings for reproducibility
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# System Settings
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parse arguments
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
parser.add_argument(
    "--model",
    type=str,
    default="watchsleepnet",  # Renamed from lstm to watchsleepnet
    choices=["watchsleepnet", "insightsleepnet", "sleepconvnet"],
)
args = parser.parse_args()


# Retrieve configuration based on the dataset argument
train_config = dataset_configurations.get(args.train_dataset, None)

# Model initialization based on user argument
if args.model == "watchsleepnet":
    model_config = WatchSleepNetConfig
    model = WatchSleepNet(
        num_features=model_config.NUM_INPUT_CHANNELS,
        feature_channels=model_config.FEATURE_CHANNELS,
        num_channels=model_config.NUM_CHANNELS,
        kernel_size=model_config.KERNEL_SIZE,
        hidden_dim=model_config.HIDDEN_DIM,
        num_heads=model_config.NUM_HEADS,
        num_layers=model_config.NUM_LAYERS,
        tcn_layers=model_config.TCN_LAYERS,
        num_classes=model_config.NUM_CLASSES,
    ).to(DEVICE)


elif args.model == "insightsleepnet":
    model_config = InsightSleepNetConfig
    model = InsightSleepNet(
        input_size=model_config.INPUT_SIZE, output_size=model_config.OUTPUT_SIZE
    ).to(DEVICE)


elif args.model == "sleepconvnet":
    model_config = SleepConvNetConfig
    model = SleepConvNet().to(DEVICE)

else:
    raise ValueError("Model not recognized.")

# Generate dynamic save path for model and analysis
model_save_path = dataset_configurations[args.train_dataset]["get_model_save_path"](
    model_name=args.model, dataset_name=args.train_dataset, version="vtrial"
)

print(model_save_path)


# Replace the finetuning step with cross-validation
print("Perform cross-validation on:", args.train_dataset)
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

print(
    "Model save path:",
    train_config["get_model_save_path"](
        model_name=args.model, dataset_name=args.train_dataset, version="vtrial"
    ),
)

# Define loss function
loss_fn = model_config.LOSS_FN

from WatchSleepNet_public.modeling.engine import train_cross_validate
# Perform cross-validation without transfer learning
results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_cross_validate(
    use_attention=True,
    use_tcn=True,
    model_name=args.model,
    model_params=model_config,
    # num_classes=model_config.NUM_CLASSES,
    num_classes = 3,
    dataloader_folds=dataloader_folds,
    learning_rate=model_config.LEARNING_RATE,
    weight_decay=model_config.WEIGHT_DECAY,
    loss_fn=loss_fn,
    num_epochs=model_config.NUM_EPOCHS,
    patience=model_config.PATIENCE,
    device=DEVICE,
    checkpoint_path=train_config["get_model_save_path"](
        model_name=args.model, dataset_name=args.train_dataset, 
        # version="vtrial"
        version="cv_no_transfer"
    ),
)
