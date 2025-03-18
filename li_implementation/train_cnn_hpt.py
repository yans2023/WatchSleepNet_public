#!/usr/bin/env python3

"""
all_in_one_tunable_licnn_with_dynamic_batchsize_singleload.py

- Loads train/val/test each exactly ONCE at startup, storing them in memory.
- Runs hyperparameter search across multiple LiCNN conv configurations, 
  re-building DataLoaders from the same in-memory Datasets for each set 
  (but never re-reading from disk).
- Uses early stopping based on val_kappa.
- Dynamically chooses batch size based on whether the largest conv layer is "large".
- Finally, re-trains on (train+val) with best hyperparams, then test.

We remove MacroAUROC from per-epoch metrics for speed, only compute 
it at the end for final advanced metrics on test. We still watch 
for NaN/Inf, clip gradients, etc.
"""

import os
import glob
import gc
gc.collect()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# -------------------- EarlyStoppingAndCheckpoint --------------------
class EarlyStoppingAndCheckpoint:
    def __init__(
        self,
        patience=20,
        verbose=True,
        delta=0,
        checkpoint_path="model_checkpoint.pt",
        trace_func=print,
        monitor=("val_kappa",),
        monitor_modes=("max",),
    ):
        """
        This class watches one or more metrics (by default "val_kappa"),
        stops training if no improvement after 'patience' epochs,
        and reverts to the best checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.trace_func = trace_func
        self.monitor = monitor if isinstance(monitor, tuple) else (monitor,)
        self.monitor_modes = (
            monitor_modes if isinstance(monitor_modes, tuple) else (monitor_modes,)
        )
        self.best_scores = {metric: None for metric in self.monitor}
        self.best_metrics = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, metrics, model):
        # Extract relevant scores
        scores = {metric: metrics[metric] for metric in self.monitor}

        # If first time => save as best
        if None in self.best_scores.values():
            self.best_scores = scores
            self.best_metrics = metrics
            self.save_checkpoint(metrics, model)
        else:
            # Check improvement
            improvement_conditions = []
            for i, metric in enumerate(self.monitor):
                mode = self.monitor_modes[i]
                if mode == "min":
                    improved = (scores[metric] <= self.best_scores[metric] - self.delta)
                else:  # "max"
                    improved = (scores[metric] >= self.best_scores[metric] + self.delta)
                improvement_conditions.append(improved)

            if all(improvement_conditions):
                # update best
                self.save_checkpoint(metrics, model)
                for i, metric in enumerate(self.monitor):
                    mode = self.monitor_modes[i]
                    if mode == "min":
                        self.best_scores[metric] = min(scores[metric], self.best_scores[metric])
                    else:
                        self.best_scores[metric] = max(scores[metric], self.best_scores[metric])
                self.best_metrics = metrics
                self.counter = 0
            else:
                self.counter += 1
                self.trace_func(f"Early stopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, metrics, model):
        if self.verbose:
            improvement_msg = []
            for i, metric in enumerate(self.monitor):
                prev_best = self.best_scores[metric]
                new_val = metrics[metric]
                if prev_best is None:
                    improvement_msg.append(f"{metric} set to {new_val:.4f}")
                else:
                    improvement_msg.append(f"{metric} improved from {prev_best:.4f} to {new_val:.4f}")
            improvement_msg = " and ".join(improvement_msg)
            self.trace_func(f"[Checkpoint] {improvement_msg}")
        torch.save(model.state_dict(), self.checkpoint_path)

# -------------------- LiCNN with tunable hyperparams --------------------
class LiCNN(nn.Module):
    """
    Li-style CNN with 3 convolution layers, 2 max-pool layers, final linear,
    but out_channels and kernel_sizes are passed in as hyperparams.

    E.g.:
      conv1_out=16, conv1_ks=3
      conv2_out=32, conv2_ks=3
      conv3_out=64, conv3_ks=3
    """
    def __init__(
        self,
        num_classes=3,
        conv1_out=16, conv1_ks=3,
        conv2_out=32, conv2_ks=3,
        conv3_out=64, conv3_ks=3
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=conv1_ks, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=conv2_ks, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=conv3_ks, stride=1, padding=0)

        self.fc_in_features = None
        self.fc = None
        self.num_classes = num_classes

    def build_fc_if_needed(self, x):
        b, c, h, w = x.shape
        fc_in = c * h * w
        if self.fc_in_features is None:
            self.fc_in_features = fc_in
            self.fc = nn.Linear(fc_in, self.num_classes).to(x.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        if self.fc is None:
            self.build_fc_if_needed(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

# -------------- Data / metrics code --------------
class AllSubjectsDataset(Dataset):
    """
    A single pass that loads NPZ from disk, storing them in memory once.
    We do NOT want to do this multiple times for each hyperparam.
    Instead, we do it once for train, once for val, once for test.
    """
    def __init__(self, subject_files):
        X_list, y_list = [], []
        total_samples = 0
        print(f"Loading {len(subject_files)} subject files into memory...")
        for fpath in tqdm(subject_files, desc="Loading partition"):
            dataz = np.load(fpath)
            X = dataz["X"]
            y = dataz["y"]
            X_list.append(X)
            y_list.append(y)
            total_samples += X.shape[0]
        self.X_all = np.concatenate(X_list, axis=0)
        self.y_all = np.concatenate(y_list, axis=0)
        del X_list, y_list
        print(f"Partition dataset size = {total_samples} samples.")

    def __len__(self):
        return len(self.y_all)

    def __getitem__(self, idx):
        spect = np.expand_dims(self.X_all[idx], axis=0).astype(np.float32)
        lbl = int(self.y_all[idx])
        return spect, lbl

def gather_predictions(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Gather preds", leave=False):
            specs = specs.to(device)
            logits = model(specs)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.numpy())
    y_logits = np.concatenate(all_logits, axis=0)
    y_true = np.array(all_labels, dtype=int)
    return y_true, y_logits

def compute_rem_f1_binary(y_true, y_pred, rem_idx):
    from sklearn.metrics import f1_score
    pos_true = (y_true == rem_idx)
    pos_pred = (y_pred == rem_idx)
    return f1_score(pos_true, pos_pred, average="binary")

def compute_metrics_no_auroc(model, loader, device, num_classes=3, rem_class_idx=2):
    y_true, y_logits = gather_predictions(model, loader, device)
    y_pred = np.argmax(y_logits, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        rem_f1 = compute_rem_f1_binary(y_true, y_pred, rem_class_idx)
    except ValueError:
        rem_f1 = np.nan
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, macro_f1, rem_f1, kappa

def gather_predictions_proba(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Gather proba", leave=False):
            specs = specs.to(device)
            logits = model(specs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_labels, dtype=int)
    return y_true, y_probs

def final_advanced_metrics(model, loader, device, num_classes=3, rem_class_idx=2):
    y_true, y_probs = gather_predictions_proba(model, loader, device)
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        rem_f1 = compute_rem_f1_binary(y_true, y_pred, rem_class_idx)
    except ValueError:
        rem_f1 = np.nan
    unique_clz = np.unique(y_true)
    if len(unique_clz) == num_classes:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        macro_auroc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    else:
        macro_auroc = np.nan
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n=== Final Metrics (with MacroAUROC) ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"REM F1:        {rem_f1:.4f}")
    print(f"Macro AUROC:   {macro_auroc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}\n")

# -------------- Basic training loops --------------
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Train for 1 epoch with checks for NaN/Inf and gradient clipping.
    Return (avg_loss, avg_acc). If NaN -> return (np.nan, np.nan).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for specs, labels in tqdm(loader, desc="Train batch loop", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf in loss => stop early.")
            return float('nan'), float('nan')
        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # check grads
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print("NaN/Inf in grad => stop early.")
                return float('nan'), float('nan')

        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total

def basic_eval_loss_acc(model, loader, criterion, device):
    """
    Evaluate returning (loss, acc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Val batch loop", leave=False):
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * specs.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += specs.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total

# -------------- Helper: trains model with early stopping (BUT uses in-memory DS) --------------
def train_model_with_earlystop_inmemory(
    model_params: dict,
    ds_train: Dataset,
    ds_val: Dataset,
    device: torch.device,
    patience=10,
    max_epochs=200,
    lr=1e-1,
    max_grad_norm=1.0
):
    """
    We do NOT reload ds_train/ds_val from disk. 
    They are already in memory. We only build new DataLoader with an appropriate batch_size.
    If conv3_out >= 256 => batch_size=2**8, else => 2**10.
    Then we do training with early stopping.
    Returns: (best_val_kappa, final_state)
    """
    # Decide batch_size
    if model_params["conv3_out"] >= 256:
        batch_size = 2**10
    else:
        batch_size = 2**12

    # Build DataLoader from ds_train, ds_val
    print(f"[train_model_with_earlystop_inmemory] => param={model_params}, batch_size={batch_size}")
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = LiCNN(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stopper = EarlyStoppingAndCheckpoint(
        patience=patience,
        verbose=True,
        delta=0,
        checkpoint_path="temp_model_checkpoint.pt",
        trace_func=print,
        monitor=("val_kappa",),
        monitor_modes=("max",),
    )

    best_val_kappa = -999.0
    best_state = None

    for epoch in range(max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{max_epochs}, param={model_params}, batch_size={batch_size} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        if np.isnan(train_loss):
            print("Stopped early due to NaN => revert to best checkpoint so far.")
            break

        # Evaluate on val
        val_loss, val_acc = basic_eval_loss_acc(model, val_loader, criterion, device)
        va_acc, va_mf1, va_rf1, va_kappa = compute_metrics_no_auroc(model, val_loader, device, 3, 2)
        print(f"[Val metrics] Acc={va_acc:.4f}, Kappa={va_kappa:.4f}, MacroF1={va_mf1:.4f}, REMF1={va_rf1:.4f}")

        stopper({"val_kappa": va_kappa}, model)
        if va_kappa > best_val_kappa:
            best_val_kappa = va_kappa
            best_state = model.state_dict().copy()

        if stopper.early_stop:
            print("[EarlyStop] => stop training for these hyperparams.")
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_kappa, best_state


# -------------------- MAIN --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    COMBINED_DIR = ".../SHHS_ECG_preprocessed_combined"
    LR = 0.01
    MAX_EPOCHS = 100

    # 1) Gather .npz => load once
    subject_files = sorted(glob.glob(os.path.join(COMBINED_DIR, "*_combined.npz")))
    n_subj = len(subject_files)
    print(f"Found {n_subj} subject-level NPZ files in {COMBINED_DIR}.")

    # 2) subject-level split
    train_end = int(n_subj * 0.6)
    val_end   = int(n_subj * 0.8)

    train_files = subject_files[:train_end]
    val_files   = subject_files[train_end:val_end]
    test_files  = subject_files[val_end:]

    print(f"Train subjects: {len(train_files)}")
    print(f"Val subjects:   {len(val_files)}")
    print(f"Test subjects:  {len(test_files)}")

    # 3) Build ds_train, ds_val, ds_test => loaded only once
    print("\nLoading ds_train in memory =>")
    ds_train = AllSubjectsDataset(train_files)

    print("\nLoading ds_val in memory =>")
    ds_val   = AllSubjectsDataset(val_files)

    print("\nLoading ds_test in memory =>")
    ds_test  = AllSubjectsDataset(test_files)

    # Hyperparameter sets
    hparam_sets = [
        {
            "num_classes": 3,
            "conv1_out": 16, "conv1_ks": 3,
            "conv2_out": 32, "conv2_ks": 3,
            "conv3_out": 64, "conv3_ks": 3,
        },
        {
            "num_classes": 3,
            "conv1_out": 32, "conv1_ks": 5,
            "conv2_out": 64, "conv2_ks": 5,
            "conv3_out": 128, "conv3_ks": 3,
        },
        {
            "num_classes": 3,
            "conv1_out": 128, "conv1_ks": 7,
            "conv2_out": 256, "conv2_ks": 5,
            "conv3_out": 512, "conv3_ks": 3,
        }
    ]

    best_val_kappa = -999.0
    best_hparams = None
    best_state = None

    # 4) HP search => train each set on (ds_train, ds_val)
    for param_dict in hparam_sets:
        print(f"\n===== Searching with param_dict = {param_dict} =====")
        val_kappa, st_dict = train_model_with_earlystop_inmemory(
            model_params=param_dict,
            ds_train=ds_train,
            ds_val=ds_val,
            device=device,
            patience=10,
            max_epochs=MAX_EPOCHS,
            lr=LR,
            max_grad_norm=2.0
        )
        print(f" => result val_kappa={val_kappa:.4f} for {param_dict}")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_hparams = param_dict.copy()
            best_state = st_dict

    print(f"\n[HP Search] Best hyperparams => {best_hparams} with val_kappa={best_val_kappa:.4f}")

    # 5) clear memory
    gc.collect()
    del ds_train
    del ds_val
    torch.cuda.empty_cache()

    # 6) combine ds_train+ds_val => final training => then test
    # Reload combined dataset from train_files and val_files
    combined_files = train_files + val_files
    print("\nReloading combined dataset from train and val files...")
    ds_combined = AllSubjectsDataset(combined_files)
    print(f"ds_combined => total {len(ds_combined)} samples.")

    print("\nNow final training on ds_combined => earlystop against ds_test.")
    # final train
    final_val_kappa, final_state = train_model_with_earlystop_inmemory(
        model_params=best_hparams,
        ds_train=ds_combined,
        ds_val=ds_test,  # using test as "val" for early stop or you can skip early stopping here
        device=device,
        patience=10,
        max_epochs=MAX_EPOCHS,
        lr=LR,
        max_grad_norm=2.0
    )
    print(f"[Final combined training] best val_kappa={final_val_kappa:.4f}")

    final_model = LiCNN(**best_hparams).to(device)
    if final_state is not None:
        final_model.load_state_dict(final_state)

    torch.save(final_state, "final_best_model_after_hps.pt")
    print("Final model saved => final_best_model_after_hps.pt")
    print("\n=== Final advanced metrics on TEST set ===")
    if best_hparams["conv3_out"] >= 256:
        test_bs = 2**10
    else:
        test_bs = 2**12
    test_loader = DataLoader(ds_test, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    final_advanced_metrics(final_model, test_loader, device, num_classes=3, rem_class_idx=2)


if __name__ == "__main__":
    main()
