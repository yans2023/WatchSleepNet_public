#!/usr/bin/env python3

"""
finetune_licnn_svm_5fold.py

Example script:
1) Use a large LiCNN model (3-layer CNN) as in Li et al. 2021 for sleep staging.
2) Perform 5-fold training/evaluation by subject:
   - For each iteration: 
     a) Train/finetune LiCNN on 4 folds (call this "train set")
     b) Extract CNN representations (and any handcrafted features) on the 4 folds
     c) Train an SVM on those extracted features
     d) Evaluate the entire pipeline on the remaining 1 fold
3) Collect metrics: Accuracy, AUROC, Macro F1, Cohen’s Kappa, and REM-F1 (label=2).
4) Aggregate metrics across folds.

IMPORTANT:
- You must adapt data-loading, beat detection, and feature computation to match your environment.
- This script is just a *template* that follows the spirit of Li et al. 2021.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score
from tqdm import tqdm

# --------------------- 1) Define large LiCNN model ---------------------

class LiCNN_Large(nn.Module):
    """
    A 'large' version of the 3-layer LiCNN architecture, akin to 'Li et al. 2021' style:
      - conv1: out_channels=128, kernel=7
      - pool1: 2x2
      - conv2: out_channels=256, kernel=5
      - pool2: 2x2
      - conv3: out_channels=512, kernel=3
      - final linear => 4 or 5 outputs (depends on your scenario)
    We can adapt to n_classes. If you want a penultimate-layer embedding, 
    we store it in forward(...) for extraction.
    """
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        # after conv3 => shape ~ (B,512,H,W). We'll figure out H,W dynamically.

        self.penultimate = None  # will hold the flattened penultimate layer
        self.fc = None
        self.n_classes = n_classes

    def build_fc_if_needed(self, x):
        # x => (B, 512, h, w)
        b, c, h, w = x.shape
        fc_in = c * h * w
        if self.fc is None:
            self.fc = nn.Linear(fc_in, self.n_classes).to(x.device)

    def forward(self, x, return_penultimate=False):
        """
        If return_penultimate=True, we return (logits, penultimate_features).
        Otherwise, we return just logits.
        """
        # x: shape (B,1,64,64) or however your input is sized
        z = F.relu(self.conv1(x))
        z = self.pool1(z)

        z = F.relu(self.conv2(z))
        z = self.pool2(z)

        z = F.relu(self.conv3(z))
        # dynamic build fc
        self.build_fc_if_needed(z)
        # flatten
        penult = z.view(z.size(0), -1)  # penultimate layer
        logits = self.fc(penult)
        if return_penultimate:
            return logits, penult
        else:
            return logits

# --------------------- 2) Dataset placeholders ---------------------

class MySleepDataset(Dataset):
    """
    Placeholder for your 5-min window CRC spectrogram or other data.
    You might also store additional handcrafted features (like HRV) if you
    want them fed to the SVM. 
    For simplicity, let's store them in the __getitem__ as well.
    """
    def __init__(self, data_list, label_list, extra_features=None):
        """
        data_list: shape [N, 1, H, W] (like a spectrogram).
        label_list: shape [N], int labels 0..(n_classes-1).
        extra_features: shape [N, d] or None
        """
        self.data = data_list
        self.labels = label_list
        self.extra = extra_features
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        x = self.data[idx]   # shape (1,H,W)
        y = self.labels[idx]
        if self.extra is not None:
            e = self.extra[idx]
            return x, y, e
        else:
            return x, y, None

# --------------------- 3) Train/Finetune LiCNN on 4 folds ---------------------

def train_licnn(model, train_loader, val_loader, device, n_epochs=10, lr=1e-3):
    """
    Example finetuning. 
    We simply do a quick training loop on the CNN classification head for some epochs.
    Possibly freeze early layers if you want partial finetuning; or freeze nothing if you want full.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -999
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False):
            x_batch, y_batch, _ = batch
            x_batch = x_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)
        train_acc = total_correct / total_samples if total_samples>0 else 0

        # quick val pass
        val_acc = evaluate_licnn(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        # print something each epoch
        print(f"Epoch {epoch+1}/{n_epochs}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

def evaluate_licnn(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x_batch, y_batch, _ = batch
        x_batch = x_batch.to(device, dtype=torch.float)
        y_batch = y_batch.to(device, dtype=torch.long)
        with torch.no_grad():
            logits = model(x_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
    if total>0:
        return correct/total
    else:
        return 0

# --------------------- 4) Extract penultimate layer & combine features for SVM ---------------------

def extract_features_and_labels(model, loader, device):
    """
    Pass data through LiCNN, returning penultimate-layer embeddings + any extra features + labels.
    This yields the final training set for SVM.
    """
    model.eval()
    reps_list = []
    extra_list = []
    label_list = []
    for batch in loader:
        x_batch, y_batch, e_batch = batch
        x_batch = x_batch.to(device, dtype=torch.float)
        with torch.no_grad():
            logits, penult = model(x_batch, return_penultimate=True)
        # penult => shape (B, something)
        penult_np = penult.cpu().numpy()
        reps_list.append(penult_np)

        if e_batch is not None:
            e_np = e_batch.numpy()
        else:
            # no extra features
            e_np = np.zeros((x_batch.size(0), 0), dtype=np.float32)
        extra_list.append(e_np)
        label_list.append(y_batch.numpy())

    # combine
    reps_final = np.concatenate(reps_list, axis=0)
    extra_final = np.concatenate(extra_list, axis=0)
    lab_final = np.concatenate(label_list, axis=0)
    # total features => penult + extra
    feats_final = np.concatenate([reps_final, extra_final], axis=1) if extra_final.shape[1]>0 else reps_final
    return feats_final, lab_final

def train_and_eval_svm(feats_train, labs_train, feats_test, labs_test, n_classes=4):
    """
    Train an SVM on feats_train => labs_train. Then test on feats_test => labs_test.
    Return metrics dict with 'acc', 'auroc', 'macro_f1', 'rem_f1', 'kappa'
      (assuming label==2 => REM).
    """
    scaler = StandardScaler()
    feats_train_sc = scaler.fit_transform(feats_train)
    feats_test_sc  = scaler.transform(feats_test)

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(feats_train_sc, labs_train)
    preds = clf.predict(feats_test_sc)
    probas = clf.predict_proba(feats_test_sc)

    acc = accuracy_score(labs_test, preds)
    kappa = cohen_kappa_score(labs_test, preds)

    # Macro-F1:
    macro_f1 = f1_score(labs_test, preds, average='macro')
    # REM-F1 => label=2
    # but watch out if your label for REM is not '2'. Adjust accordingly
    rem_f1 = 0
    try:
        rem_f1 = f1_score(labs_test, preds, labels=[2], average='binary')
    except:
        pass

    # AUROC => for multi-class:
    # or if 2-class => do something simpler.
    # For multi-class, you can do e.g. 'ovr' approach
    auroc = 0
    if n_classes>2:
        # convert labs_test => one-hot
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        labs_bin = label_binarize(labs_test, classes=list(range(n_classes)))
        auroc = roc_auc_score(labs_bin, probas, average='macro', multi_class='ovr')
    else:
        # 2-class => auroc directly
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labs_test))==2:
            auroc = roc_auc_score(labs_test, probas[:,1])

    return {
        'acc': acc,
        'kappa': kappa,
        'macro_f1': macro_f1,
        'rem_f1': rem_f1,
        'auroc': auroc
    }

# --------------------- 5) Full 5-fold routine ---------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Suppose we have a dataset with 5 folds by subject. We define:
    # data_folds, label_folds, extra_folds are lists of length=5, each fold containing data, labels, extra_features
    # For example, data_folds[i] => shape (N_i, 1, 64,64) or your CRC shape
    # label_folds[i] => shape (N_i)
    # extra_folds[i] => shape (N_i, D) if you have handcrafted features

    # You must define these folds yourself from your data. Then:
    n_classes = 4  # or 5 if you consider Li et al. 2021 with 4 + wake, etc.

    # Let's say we have them loaded:
    data_folds = [...]
    label_folds = [...]
    extra_folds = [...]

    # We will store final metrics from each fold
    all_metrics = []

    for fold_idx in range(5):
        # 1) Combine all except fold_idx for training
        train_data = []
        train_labels = []
        train_extra = []
        test_data = data_folds[fold_idx]
        test_labels = label_folds[fold_idx]
        test_extra = extra_folds[fold_idx]

        for j in range(5):
            if j == fold_idx:
                continue
            train_data.append(data_folds[j])
            train_labels.append(label_folds[j])
            train_extra.append(extra_folds[j])

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_extra = np.concatenate(train_extra, axis=0)

        # 2) Build PyTorch Datasets
        ds_train = MySleepDataset(train_data, train_labels, train_extra)
        ds_test  = MySleepDataset(test_data, test_labels, test_extra)

        train_loader = DataLoader(ds_train, batch_size=256, shuffle=True, num_workers=4)
        val_loader   = DataLoader(ds_train, batch_size=256, shuffle=False, num_workers=4)
        # note: we do not have a separate "val" fold here, 
        # you could do a sub-split or try an internal approach. 
        # For simplicity let's just re-use train to check progress.

        test_loader = DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=4)

        # 3) Create LiCNN and train (finetune). 
        model = LiCNN_Large(n_classes=n_classes).to(device)
        train_licnn(model, train_loader, val_loader, device, n_epochs=10, lr=1e-2)

        # 4) Extract penultimate features from train set => train SVM
        feats_train, labs_train = extract_features_and_labels(model, train_loader, device)
        # 5) Extract penultimate features from test set => test SVM
        feats_test, labs_test = extract_features_and_labels(model, test_loader, device)

        # 6) Train & eval SVM
        results = train_and_eval_svm(feats_train, labs_train, feats_test, labs_test, n_classes=n_classes)
        print(f"Fold {fold_idx} results = {results}")
        all_metrics.append(results)

    # after 5 folds, aggregate
    # e.g. compute mean of each metric
    final_acc   = np.mean([m['acc']       for m in all_metrics])
    final_kappa = np.mean([m['kappa']     for m in all_metrics])
    final_mf1   = np.mean([m['macro_f1']  for m in all_metrics])
    final_rf1   = np.mean([m['rem_f1']    for m in all_metrics])
    final_auroc = np.mean([m['auroc']     for m in all_metrics])

    print("=== 5-FOLD AGGREGATED RESULTS ===")
    print(f"Accuracy:      {final_acc:.4f}")
    print(f"Macro F1:      {final_mf1:.4f}")
    print(f"REM F1:        {final_rf1:.4f}")
    print(f"AUROC:         {final_auroc:.4f}")
    print(f"Cohen’s Kappa: {final_kappa:.4f}")

if __name__ == "__main__":
    main()
