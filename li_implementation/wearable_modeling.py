#!/usr/bin/env python3
"""
THis script is incomplete because we could not verify the correct implementation of the CNN model,
and cannot continue on to perform transfer learning.

Example script:
- Use a large LiCNN model (3-layer CNN) with the best config.
- Perform 5-fold cross validation to train and evaluate the CNN+SVM pipeline.
- In addition to the CNN’s penultimate features, HRV features are computed from raw ECG signals.
- Spectrogram and ECG data come from separate directories (fill in the placeholders).
- The SVM combines CNN outputs with HRV features.
- Metrics are aggregated across folds.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.signal import find_peaks, welch

# LiCNN model using best config parameters.
class LiCNN_Large(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.fc = None
        self.n_classes = n_classes

    def build_fc_if_needed(self, x):
        b, c, h, w = x.shape
        fc_in = c * h * w
        if self.fc is None:
            self.fc = nn.Linear(fc_in, self.n_classes).to(x.device)

    def forward(self, x, return_penultimate=False):
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)
        z = F.relu(self.conv3(z))
        self.build_fc_if_needed(z)
        penult = z.view(z.size(0), -1)
        logits = self.fc(penult)
        if return_penultimate:
            return logits, penult
        else:
            return logits

# HRV Feature Engineering functions.
def detect_r_peaks(ecg_signal, fs):
    # Simple peak detection using a minimum distance of 0.3 sec.
    distance = int(0.3 * fs)
    peaks, _ = find_peaks(ecg_signal, distance=distance, height=np.mean(ecg_signal) + np.std(ecg_signal))
    return peaks

def sample_entropy(time_series, m, r):
    N = len(time_series)
    def _phi(m):
        x = np.array([time_series[i:N - m + i + 1] for i in range(m)]).T
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / ((N - m) * (N - m - 1) + 1e-10)
    return -np.log((_phi(m + 1) / _phi(m)) + 1e-10)

def extract_hrv_features(ecg_signal, fs=256):
    # Detect R-peaks and compute NN intervals.
    r_peaks = detect_r_peaks(ecg_signal, fs)
    if len(r_peaks) < 2:
        return np.zeros(8)
    nn_intervals = np.diff(r_peaks) / fs  # in seconds
    sdnn = np.std(nn_intervals)
    sampen = sample_entropy(nn_intervals, m=2, r=0.2 * np.std(nn_intervals))
    # Compute power spectral density (PSD) using Welch's method.
    fxx, pxx = welch(nn_intervals, fs=4, nperseg=min(256, len(nn_intervals)))
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    lf_mask = (fxx >= lf_band[0]) & (fxx < lf_band[1])
    hf_mask = (fxx >= hf_band[0]) & (fxx < hf_band[1])
    lf_power = np.trapz(pxx[lf_mask], fxx[lf_mask])
    hf_power = np.trapz(pxx[hf_mask], fxx[hf_mask])
    lfhf = lf_power / hf_power if hf_power > 0 else 0
    # Compute PRSA measures (deceleration capacity (DC) and acceleration capacity (AC)).
    try:
        import neurokit2 as nk
        prsa_results = nk.hrv_prsa(nn_intervals, sampling_rate=4)
        dc = prsa_results.get("DC", 0.0)
        ac = prsa_results.get("AC", 0.0)
    except Exception:
        dc = 0.0
        ac = 0.0
    # Compute novel indices: peak ratio and energy ratio using novel frequency bands.
    lf_novel_band = (0.01, 0.1)
    hf_novel_band = (0.1, 0.4)
    lf_novel_mask = (fxx >= lf_novel_band[0]) & (fxx < lf_novel_band[1])
    hf_novel_mask = (fxx >= hf_novel_band[0]) & (fxx < hf_novel_band[1])
    peaks_lf, _ = find_peaks(pxx[lf_novel_mask])
    peaks_hf, _ = find_peaks(pxx[hf_novel_mask])
    lf_peak_heights = pxx[lf_novel_mask][peaks_lf] if peaks_lf.size > 0 else np.array([])
    hf_peak_heights = pxx[hf_novel_mask][peaks_hf] if peaks_hf.size > 0 else np.array([])
    sum_lf_peaks = np.sum(np.sort(lf_peak_heights)[-2:]) if lf_peak_heights.size >= 2 else (np.sum(lf_peak_heights) if lf_peak_heights.size > 0 else 0.0)
    sum_hf_peaks = np.sum(np.sort(hf_peak_heights)[-2:]) if hf_peak_heights.size >= 2 else (np.sum(hf_peak_heights) if hf_peak_heights.size > 0 else 0.0)
    peak_ratio = sum_lf_peaks / sum_hf_peaks if sum_hf_peaks > 0 else 0.0
    lf_energy = np.trapz(pxx[lf_novel_mask], fxx[lf_novel_mask])
    hf_energy = np.trapz(pxx[hf_novel_mask], fxx[hf_novel_mask])
    energy_ratio = lf_energy / hf_energy if hf_energy > 0 else 0.0
    mean_val = np.mean(np.abs(ecg_signal))
    sqi = np.std(ecg_signal) / mean_val if mean_val != 0 else 0
    features = np.array([sdnn, sampen, lfhf, dc, ac, peak_ratio, energy_ratio, sqi])
    return features

# Custom dataset that loads spectrogram and ECG data from separate directories.
class CustomSleepDataset(Dataset):
    def __init__(self, spectrogram_file_list, ecg_file_list, labels, transform=None):
        # Expect lists of filenames (not full paths) for spectrogram and ECG data.
        self.spectrogram_file_list = spectrogram_file_list
        self.ecg_file_list = ecg_file_list
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Construct full paths using placeholder directories.
        spectrogram_dir = ""  # Fill in your spectrogram data directory.
        ecg_dir = ""          # Fill in your ECG data directory.
        spec_path = os.path.join(spectrogram_dir, self.spectrogram_file_list[idx])
        ecg_path = os.path.join(ecg_dir, self.ecg_file_list[idx])
        # Load spectrogram.
        spec_data = np.load(spec_path)
        x = spec_data['spectrogram']  # Expected shape: (1, H, W)
        # Load ECG and compute HRV features.
        ecg_data = np.load(ecg_path)
        if 'ecg' in ecg_data:
            ecg_signal = ecg_data['ecg'].flatten()
            hrv_feats = extract_hrv_features(ecg_signal, fs=250)
        else:
            hrv_feats = None
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, hrv_feats

# Function to train the LiCNN model.
def train_licnn(model, train_loader, val_loader, device, n_epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1e9
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
        train_acc = total_correct / total_samples if total_samples > 0 else 0
        val_acc = evaluate_licnn(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{n_epochs}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

# Function to evaluate the LiCNN model.
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
    return correct / total if total > 0 else 0

# Extract CNN penultimate features and HRV features.
def extract_features_and_labels(model, loader, device):
    model.eval()
    reps_list = []
    hrv_list = []
    label_list = []
    for batch in loader:
        x_batch, y_batch, hrv_batch = batch
        x_batch = x_batch.to(device, dtype=torch.float)
        with torch.no_grad():
            _, penult = model(x_batch, return_penultimate=True)
        reps_list.append(penult.cpu().numpy())
        if hrv_batch[0] is not None:
            hrv_feats = np.stack([np.array(feat) for feat in hrv_batch])
        else:
            hrv_feats = np.zeros((x_batch.size(0), 8), dtype=np.float32)
        hrv_list.append(hrv_feats)
        label_list.append(y_batch.numpy())
    reps_final = np.concatenate(reps_list, axis=0)
    hrv_final = np.concatenate(hrv_list, axis=0)
    lab_final = np.concatenate(label_list, axis=0)
    feats_final = np.concatenate([reps_final, hrv_final], axis=1)
    return feats_final, lab_final

# Train and evaluate an SVM on the extracted features.
def train_and_eval_svm(feats_train, labs_train, feats_test, labs_test, n_classes=3):
    scaler = StandardScaler()
    feats_train_sc = scaler.fit_transform(feats_train)
    feats_test_sc = scaler.transform(feats_test)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(feats_train_sc, labs_train)
    preds = clf.predict(feats_test_sc)
    probas = clf.predict_proba(feats_test_sc)
    acc = accuracy_score(labs_test, preds)
    kappa = cohen_kappa_score(labs_test, preds)
    macro_f1 = f1_score(labs_test, preds, average='macro')
    rem_f1 = 0
    try:
        rem_f1 = f1_score(labs_test, preds, labels=[2], average='binary')
    except Exception:
        pass
    auroc = 0
    if n_classes > 2:
        from sklearn.preprocessing import label_binarize
        labs_bin = label_binarize(labs_test, classes=list(range(n_classes)))
        auroc = roc_auc_score(labs_bin, probas, average='macro', multi_class='ovr')
    else:
        if len(np.unique(labs_test)) == 2:
            auroc = roc_auc_score(labs_test, probas[:,1])
    return {'acc': acc, 'kappa': kappa, 'macro_f1': macro_f1, 'rem_f1': rem_f1, 'auroc': auroc}

# Main function with 5-fold cross-validation.
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    n_classes = 3 # W vs NREM vs REM


    spectrogram_data_dir = ".../DREAMT_spec"  # Spectrogram data directory
    ecg_data_dir = ".../DREAMT_PPG"          # ECG data directory

    # Build file lists from the directories.
    spectrogram_file_list = sorted([f for f in os.listdir(spectrogram_data_dir) if f.endswith('.npz')])
    ecg_file_list = sorted([f for f in os.listdir(ecg_data_dir) if f.endswith('.npz')])

    # TODO: Dataloading

    # Perform 5-fold cross-validation.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    fold_idx = 0
    for train_idx, test_idx in kf.split(range(len(full_dataset))):
        fold_idx += 1
        print(f"\n===== Fold {fold_idx} =====")
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=4)
        val_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=4)
        model = LiCNN_Large(n_classes=n_classes).to(device)
        train_licnn(model, train_loader, val_loader, device, n_epochs=10, lr=1e-2)
        feats_train, labs_train = extract_features_and_labels(model, train_loader, device)
        feats_test, labs_test = extract_features_and_labels(model, test_loader, device)
        results = train_and_eval_svm(feats_train, labs_train, feats_test, labs_test, n_classes=n_classes)
        print(f"Fold {fold_idx} results: {results}")
        all_metrics.append(results)

    final_acc = np.mean([m['acc'] for m in all_metrics])
    final_kappa = np.mean([m['kappa'] for m in all_metrics])
    final_mf1 = np.mean([m['macro_f1'] for m in all_metrics])
    final_rf1 = np.mean([m['rem_f1'] for m in all_metrics])
    final_auroc = np.mean([m['auroc'] for m in all_metrics])
    print("\n=== 5-FOLD AGGREGATED RESULTS ===")
    print(f"Accuracy:      {final_acc:.4f}")
    print(f"Macro F1:      {final_mf1:.4f}")
    print(f"REM F1:        {final_rf1:.4f}")
    print(f"AUROC:         {final_auroc:.4f}")
    print(f"Cohen’s Kappa: {final_kappa:.4f}")

if __name__ == "__main__":
    main()
