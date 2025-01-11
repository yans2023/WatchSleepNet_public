import os
import torch
import argparse
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from data_setup import create_dataloaders, create_dataloaders_kfolds
from config import (
    WatchSleepNetConfig, 
    InsightSleepNetConfig, 
    SleepConvNetConfig, 
    dataset_configurations,
)
from engine import (
    train,
    validate_step, 
    train_ablate_evaluate,
)
# We assume your ablation function is named "train_ablate_evaluate(...)"

# ----------------------- Reproducibility -----------------------
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------- System Settings -----------------------
NUM_WORKERS = os.cpu_count() // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- Arg Parsing -----------------------
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
    help="Name of the dataset used for pretraining."
)
parser.add_argument(
    "--test_dataset",
    type=str,
    default="dreamt_pibi",
    choices=["dreamt_pibi"],
    help="Name of the dataset used for transfer learning / fine-tuning."
)
parser.add_argument(
    "--task", 
    type=str, 
    default="sleep_staging", 
    choices=["sleep_staging", "sleep_wake"],
    help="Classification task."
)
parser.add_argument(
    "--model",
    type=str,
    default="watchsleepnet",
    choices=["watchsleepnet", "insightsleepnet", "sleepconvnet"],
    help="Which model architecture to use."
)
parser.add_argument(
    "--testing",
    action="store_true",
    help="If set, load pre-trained model and skip re-training."
)

# Ablation flags for watchsleepnet
parser.add_argument(
    "--use_tcn", 
    dest="use_tcn", 
    action="store_true", 
    help="Enable TCN module in WatchSleepNet."
)
parser.add_argument(
    "--use_attention",
    dest="use_attention",
    action="store_true",
    help="Enable Attention module in WatchSleepNet."
)
# Default them to False
parser.set_defaults(use_tcn=False, use_attention=False)

args = parser.parse_args()

assert (
    args.train_dataset != args.test_dataset
), "Train and test datasets must differ for pretraining + finetuning."

# ----------------------- Retrieve Dataset Configs -----------------------
train_config = dataset_configurations.get(args.train_dataset, None)
test_config = dataset_configurations.get(args.test_dataset, None)
if train_config is None or test_config is None:
    raise ValueError(f"No configuration found for {args.train_dataset} or {args.test_dataset}.")

# ----------------------- Model Config + Dictionary -----------------------
if args.model == "watchsleepnet":
    model_config_class = WatchSleepNetConfig
elif args.model == "insightsleepnet":
    model_config_class = InsightSleepNetConfig
elif args.model == "sleepconvnet":
    model_config_class = SleepConvNetConfig
else:
    raise ValueError(f"Model not recognized: {args.model}")

# Convert model config class to dict (assuming there's a .to_dict() method)
model_config_dict = model_config_class.to_dict()

# Now incorporate ablation flags if watchsleepnet
# (If not watchsleepnet, they won't matter.)
model_config_dict["use_tcn"] = args.use_tcn
model_config_dict["use_attention"] = args.use_attention

# Save paths
pretrain_save_path = train_config["get_model_save_path"](
    model_name=args.model, dataset_name=args.train_dataset, version="ablate_transfer"
)
finetune_save_path = test_config["get_model_save_path"](
    model_name=args.model, dataset_name=args.test_dataset, version="ablate_transfer"
)

print(f"Pretrain model checkpoint path: {pretrain_save_path}")
print(f"Finetune model checkpoint path: {finetune_save_path}")

# ----------------------- Possibly Load Pretrained -----------------------
if args.testing:
    # If testing flag is set, we assume pretrain is done, just load
    if os.path.exists(pretrain_save_path):
        print(f"Loaded pre-trained model from: {pretrain_save_path}")
    else:
        raise FileNotFoundError(f"Pre-trained model not found at: {pretrain_save_path}")
else:
    # ----------------------- Pretrain Phase -----------------------
    print(f"Pretraining from scratch on {args.train_dataset}. Saving to {pretrain_save_path} ...")

    train_dataloader, val_dataloader, _ = create_dataloaders(
        dir=train_config["directory"],
        train_ratio=0.8,
        val_ratio=0.2,
        batch_size=model_config_class.BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dataset=args.train_dataset,
        multiplier=train_config["multiplier"],
        downsampling_rate=train_config["downsampling_rate"],
        task=args.task,
    )

    # Build the loss function
    loss_fn = model_config_class.LOSS_FN

    # Call the new train(...) function
    train_logs = train(
        model_name=args.model,
        model_params=model_config_dict,  # includes ablation flags if watchsleepnet
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        num_epochs=model_config_class.NUM_EPOCHS,
        patience=model_config_class.PATIENCE,
        device=DEVICE,
        model_save_path=pretrain_save_path,
        saved_model_path=None,  # no pretrained
        learning_rate=model_config_class.LEARNING_RATE,
        weight_decay=model_config_class.WEIGHT_DECAY,
        freeze_layers=False,
    )

    # Optionally validate on val_dataloader
    if os.path.exists(pretrain_save_path):
        print("\nPretraining complete. Loading best checkpoint for final validation.")
        # Recreate the model from the dict, load state, do a quick val check:
        from engine import setup_model_and_optimizer, validate_step
        pretrained_model, _ = setup_model_and_optimizer(
            model_name=args.model,
            model_params=model_config_dict,
            device=DEVICE,
            saved_model_path=None,
            learning_rate=model_config_class.LEARNING_RATE,
            weight_decay=model_config_class.WEIGHT_DECAY,
            freeze_layers=False,
        )
        pretrained_model.load_state_dict(torch.load(pretrain_save_path, map_location=DEVICE))
        val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc = validate_step(
            pretrained_model, val_dataloader, loss_fn, DEVICE
        )
        print("\n=== Best Pretraining Validation Results ===")
        print(f"Val Loss:   {val_loss:.3f}")
        print(f"Val Acc:    {val_acc:.3f}")
        print(f"Val F1:     {val_f1:.3f}")
        print(f"Val Kappa:  {val_kappa:.3f}")
        print(f"Val REM F1: {val_rem_f1:.3f}")
        print(f"Val AUROC:  {val_auroc:.3f}")
    else:
        print("Warning: No best checkpoint found after pretraining. Skipping validation step.")

# ----------------------- Finetuning Cross-Validation -----------------------
print(f"\nNow performing finetuning on {args.test_dataset}.")
dataloader_folds = create_dataloaders_kfolds(
    dir=test_config["directory"],
    dataset=args.test_dataset,
    num_folds=5,
    val_ratio=0.2,
    batch_size=model_config_class.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    multiplier=test_config["multiplier"],
    downsampling_rate=test_config["downsampling_rate"],
)
print("Created folds for cross-validation finetuning.\n")

loss_fn = model_config_class.LOSS_FN

if args.model == "watchsleepnet":
    print("Fine-tuning WatchSleepNet with ablation flags:", 
          f"use_tcn={args.use_tcn}, use_attention={args.use_attention}")
    best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_ablate_evaluate(
        model_name=args.model,
        use_tcn=args.use_tcn,             # function param
        use_attention=args.use_attention, # function param
        model_params=model_config_dict,   # dictionary of watchsleepnet hyperparams
        num_classes=model_config_class.NUM_CLASSES,  # if your ablate fn requires
        dataloader_folds=dataloader_folds,
        saved_model_path=pretrain_save_path,   # load pretrained
        learning_rate=model_config_class.LEARNING_RATE,
        weight_decay=model_config_class.WEIGHT_DECAY,
        loss_fn=loss_fn,
        num_epochs=model_config_class.NUM_EPOCHS,
        patience=model_config_class.PATIENCE,
        device=DEVICE,
        checkpoint_path=finetune_save_path,  # base path for fold i
    )
    print("\n=== Finetuning Results ===")
    print(f"Overall Acc: {overall_acc:.3f}")
    print(f"Overall F1:  {overall_f1:.3f}")
    print(f"Overall Kappa: {overall_kappa:.3f}")
    print(f"REM F1 Score: {rem_f1:.3f}")
    print(f"AUROC:        {auroc:.3f}")
else:
    print("Not implemented for this model yet.")
