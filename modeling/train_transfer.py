import os
import torch
import argparse
import warnings
import random
import numpy as np

from data_setup import create_dataloaders, create_dataloaders_kfolds
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
    dataset_configurations,
)
from engine import train, train_and_evaluate, validate_step

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------ Reproducibility ------------------
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------ System Settings ------------------
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------ Settings ------------------
TRAIN_DATASET = "shhs_mesa_ibi"
TEST_DATASET = "dreamt_pibi"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="watchsleepnet",  
    choices=["watchsleepnet", "insightsleepnet", "sleepconvnet", "sleepppgnet"],
    help="Which model architecture to use."
)
parser.add_argument(
    "--testing",
    action="store_true",
    help="If set, load an existing pretrained model and skip re-training."
)
args = parser.parse_args()


# --------------- Retrieve Dataset Configs ---------------
train_config = dataset_configurations.get(TRAIN_DATASET, None)
test_config = dataset_configurations.get(TEST_DATASET, None)
if train_config is None or test_config is None:
    raise ValueError(f"Invalid dataset configuration for {TRAIN_DATASET} or {TEST_DATASET}.")


# --------------- Pick the appropriate Model Config ---------------
if args.model == "watchsleepnet":
    model_config_class = WatchSleepNetConfig
elif args.model == "insightsleepnet":
    model_config_class = InsightSleepNetConfig
elif args.model == "sleepconvnet":
    model_config_class = SleepConvNetConfig
elif args.model == "sleepppgnet":
    model_config_class = SleepPPGNetConfig
else:
    raise ValueError(f"Unknown model: {args.model}")

model_config_dict = model_config_class.to_dict()  


# --------------- Build the save paths ---------------
model_save_path = train_config["get_model_save_path"](
    model_name=args.model, 
    dataset_name=TRAIN_DATASET, 
    version="vtrial"
)

finetune_save_path = test_config["get_model_save_path"](
    model_name=args.model, 
    dataset_name=TEST_DATASET, 
    version="vtrial"
)

print("Pretrain/Initial Model Save Path:", model_save_path)
print("Finetune Model Save Path:", finetune_save_path)


# --------------- Possibly load a pre-trained model ---------------
if args.testing:
    # We assume you've already pretrained => Just load & skip training
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"No existing model at: {model_save_path}")
    print("Loaded pre-trained model from:", model_save_path)
else:
    # --------------- Pretraining ---------------
    print(f"Pretrain model from scratch and save to: {model_save_path}")

    # Create a single train/val split for pretraining
    train_dataloader, val_dataloader, _ = create_dataloaders(
        dir=train_config["directory"],
        train_ratio=0.8,
        val_ratio=0.2,
        batch_size=model_config_class.BATCH_SIZE,  # or use .to_dict() approach
        num_workers=NUM_WORKERS,
        dataset=TRAIN_DATASET,
        multiplier=train_config["multiplier"],
        downsampling_rate=train_config["downsampling_rate"],
        task="sleep_staging",
    )

    # Build the loss function
    loss_fn = model_config_class.LOSS_FN
    train_logs = train(
        model_name=args.model,
        model_params=model_config_dict,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        num_epochs=model_config_class.NUM_EPOCHS,
        patience=model_config_class.PATIENCE,
        device=DEVICE,
        model_save_path=model_save_path,
        saved_model_path=None,
        learning_rate=model_config_class.LEARNING_RATE,
        weight_decay=model_config_class.WEIGHT_DECAY,
        freeze_layers=False,
    )

    # After training, the best checkpoint is already saved at model_save_path;
    # Optionally evaluate final validation metrics:
    from engine import validate_step
    if os.path.exists(model_save_path):
        # Load the best model is stored at model_save_path
        pretrained_state = torch.load(model_save_path, map_location=DEVICE)
        # We'll do a quick "validate_step" by creating a model again w/ same hyperparams
        # But for simplicity, just re-run the engine's approach or do a short check.
        print("\n[Pretraining Completed] Loading best checkpoint for final validation step.")
        # We can call setup_model_and_optimizer again with same hyperparams, then load_state_dict.
        from engine import setup_model_and_optimizer
        pretrained_model, _ = setup_model_and_optimizer(
            model_name=args.model,
            model_params=model_config_dict,
            device=DEVICE,
            saved_model_path=None,  # We'll manually load the state after creation
            learning_rate=model_config_class.LEARNING_RATE,
            weight_decay=model_config_class.WEIGHT_DECAY,
            freeze_layers=False,
        )
        pretrained_model.load_state_dict(pretrained_state)
        val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc = validate_step(
            pretrained_model, val_dataloader, loss_fn, DEVICE
        )
        print("\n" + "="*80)
        print("Best Pretraining Validation Results")
        print("="*80)
        print(f"Val Loss: {val_loss:.3f}")
        print(f"Val Acc: {val_acc:.3f}")
        print(f"Val F1: {val_f1:.3f}")
        print(f"Val Kappa: {val_kappa:.3f}")
        print(f"Val REM F1: {val_rem_f1:.3f}")
        print(f"Val AUROC: {val_auroc:.3f}")
    else:
        print("Warning: No checkpoint found after pretraining. Skipping final validation.")

# --------------- Transfer Learning / Finetuning ---------------
print(f"Perform transfer learning on: {TEST_DATASET}")
dataloader_folds = create_dataloaders_kfolds(
    dir=test_config["directory"],
    dataset=TEST_DATASET,
    num_folds=5,
    val_ratio=0.2,
    batch_size=model_config_class.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    multiplier=test_config["multiplier"],
    downsampling_rate=test_config["downsampling_rate"],
)
print("Dataloader folds for finetuning created.\n")

# We re-use the same `loss_fn`
loss_fn = model_config_class.LOSS_FN

# Now do cross-validation across the folds.
print("Starting cross-validation finetuning + evaluation...")
best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_and_evaluate(
    model_name=args.model,
    model_params=model_config_dict,  # The dictionary of hyperparams
    dataloader_folds=dataloader_folds,
    saved_model_path=model_save_path,  # This is the path to your pre-trained checkpoint
    loss_fn=loss_fn,
    num_epochs=model_config_class.NUM_EPOCHS,
    patience=model_config_class.PATIENCE,
    device=DEVICE,
    checkpoint_path=finetune_save_path,  # Base path for fold-specific finetuned checkpoints
    learning_rate=model_config_class.LEARNING_RATE,
    weight_decay=model_config_class.WEIGHT_DECAY,
    freeze_layers=False,
)

print("\n=== Transfer Learning Completed ===")
print(f"Overall ACC: {overall_acc:.3f}")
print(f"Overall F1:  {overall_f1:.3f}")
print(f"Overall Kappa: {overall_kappa:.3f}")
print(f"REM F1 Score: {rem_f1:.3f}")
print(f"AUROC: {auroc:.3f}")
