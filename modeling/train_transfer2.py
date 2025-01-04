import os
import torch
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from data_setup import create_dataloaders, create_dataloaders_kfolds
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from config import WatchSleepNetConfig, InsightSleepNetConfig, SleepConvNetConfig, dataset_configurations
from engine import train, train_and_evaluate, validate_step, train_ablate_evaluate
from utils import print_model_info
import random
import numpy as np
from models.watchsleepnet2 import WatchSleepNet

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
    "--test_dataset",
    type=str,
    default="dreamt_pibi",
    choices=[
            "dreamt_pibi"
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
parser.add_argument("--testing", action="store_true")

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
# Set default values to False
parser.set_defaults(use_tcn=False, use_attention=False)
args = parser.parse_args()

assert (
    args.train_dataset != args.test_dataset
), "Train and test datasets must be different for pretraining + finetuning setup."

# Retrieve configuration based on the dataset argument
train_config = dataset_configurations.get(args.train_dataset, None)
test_config = dataset_configurations.get(args.test_dataset, None)

# Model initialization based on user argument
if args.model == "watchsleepnet":
    model_config = WatchSleepNetConfig
    model = WatchSleepNet(
        num_features=model_config.NUM_INPUT_CHANNELS,
        # feature_channels=model_config.FEATURE_CHANNELS,
        num_channels=model_config.NUM_CHANNELS,
        kernel_size=model_config.KERNEL_SIZE,
        hidden_dim=model_config.HIDDEN_DIM,
        num_heads=model_config.NUM_HEADS,
        num_layers=model_config.NUM_LAYERS,
        tcn_layers=model_config.TCN_LAYERS,
        num_classes=model_config.NUM_CLASSES,
        use_tcn=args.use_tcn,
        use_attention=args.use_attention,
    ).to(DEVICE)


elif args.model == "insightsleepnet":
    model_config = InsightSleepNetConfig
    model = InsightSleepNet(input_size=model_config.INPUT_SIZE, 
                            output_size=model_config.OUTPUT_SIZE).to(DEVICE)
    

elif args.model == "sleepconvnet":
    model_config = SleepConvNetConfig
    model = SleepConvNet().to(DEVICE)

else:
    raise ValueError("Model not recognized.")

# Generate dynamic save path for model and analysis
model_save_path = dataset_configurations[args.train_dataset]["get_model_save_path"](
    model_name=args.model, dataset_name=args.train_dataset, version="ablate_transfer"
)

print(model_save_path)

# Load pre-trained model if --testing is set
if args.testing:
    model.load_state_dict(torch.load(model_save_path))
    print("Loaded pre-trained model from:", model_save_path)
else:
    print("Pretrain model from scratch and save to:", model_save_path)

    # Train model
    train_dataloader, val_dataloader, _ = create_dataloaders(
        dir=train_config["directory"],
        train_ratio=0.8,
        val_ratio=0.2,
        batch_size=model_config.BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dataset=args.train_dataset,
        multiplier=train_config["multiplier"],
        downsampling_rate=train_config["downsampling_rate"],
        task=args.task,
    )

    # Define loss function (ignoring -1 as requested)
    loss_fn = model_config.LOSS_FN
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config.LEARNING_RATE, weight_decay=model_config.WEIGHT_DECAY
    )

    # Train the model
    results = train(
        model,
        args.model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        num_epochs=model_config.NUM_EPOCHS,
        device=DEVICE,
        patience=model_config.PATIENCE,
        model_save_path=model_save_path,
    )

    # Load the best model and perform final testing
    model.load_state_dict(torch.load(model_save_path))

    val_loss, val_acc, val_macro_f1, val_kappa, val_rem_f1, val_auroc = validate_step(
        model, val_dataloader, loss_fn, DEVICE
    )
    print("\n" + "=" * 80)
    print("Best Pretraining Validation Results")
    print("\n" + "=" * 80)
    print(f"Accuracy: {val_acc:.3f}")
    print(f"Macro F1 Score: {val_macro_f1:.3f}")
    print(f"REM F1 Score: {val_rem_f1:.3f}")
    print(f"AUROC: {val_auroc}")
    print(f"Cohen's Kappa: {val_kappa:.3f}")

# Now perform finetuning
print("Perform transfer learning on:", args.test_dataset)
dataloader_folds = create_dataloaders_kfolds(
    dir=test_config["directory"],
    dataset=args.test_dataset,
    num_folds=5,
    val_ratio=0.2,
    batch_size=model_config.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    multiplier=test_config["multiplier"],
    downsampling_rate=test_config["downsampling_rate"],
)

print(
    "Finetune model save path",
    test_config["get_model_save_path"](
        model_name=args.model, dataset_name=args.test_dataset, version="ablate_transfer"
    ),
)
# Define loss function (ignoring -1 as requested)
loss_fn = model_config.LOSS_FN

if args.model == "watchsleepnet":

    # Perform transfer learning with 5-fold cross-validation
    results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = (
        train_ablate_evaluate(
            model_name=args.model,
            use_tcn=args.use_tcn,
            use_attention=args.use_attention,
            model_params=model_config,
            num_classes=model_config.NUM_CLASSES,
            dataloader_folds=dataloader_folds,
            saved_model_path=model_save_path,
            learning_rate=model_config.LEARNING_RATE,
            weight_decay=model_config.WEIGHT_DECAY,
            loss_fn=loss_fn,
            num_epochs=model_config.NUM_EPOCHS,
            patience=model_config.PATIENCE,
            device=DEVICE,
            checkpoint_path=test_config["get_model_save_path"](
                model_name=args.model,
                dataset_name=args.test_dataset,
                version="ablate_transfer",
            ),
        )
    )
else:
    print("Not Implemented Yet.")
