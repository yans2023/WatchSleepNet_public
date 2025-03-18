import os
import time
import torch
import numpy as np

from config import (
    dataset_configurations,
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
)
from engine import setup_model_and_optimizer
from data_setup import SSDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# The dataset config for dreamt_pibi
dreamt_pibi_config = dataset_configurations["dreamt_pibi"]
DATASET_DIR = dreamt_pibi_config["directory"] 

# Example: pick one .npz from dreamt_pibi to measure
SINGLE_SUBJECT_FILE = ".../DREAMT_PIBI_SE/S002.npz"  # <-- Replace with an actual .npz file

# Paths to each model’s newest checkpoint:
MODEL_PATHS = {
    "insightsleepnet": "modeling/checkpoints/insightsleepnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt",
    "sleepconvnet":    "modeling/checkpoints/sleepconvnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt",
    "sleepppgnet":     "modeling/checkpoints/sleepppgnet/shhs_mesa_ibi/best_saved_model_cv_no_transfer_fold1.pt",
    "watchsleepnet":   "modeling/checkpoints/watchsleepnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt"
}

# Model hyperparameters used for instantiating each architecture:
MODEL_CONFIGS = {
    "insightsleepnet": InsightSleepNetConfig.to_dict(),
    "sleepconvnet":    SleepConvNetConfig.to_dict(),
    "sleepppgnet":     SleepPPGNetConfig.to_dict(),
    "watchsleepnet":   WatchSleepNetConfig.to_dict()
}

class SingleSubjectDataset(SSDataset):
    """
    Subclass that forces the file list to contain only one file
    (so we can easily load exactly one subject).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optionally, assert there is exactly one file:
        # assert len(self.files) == 1, "Expected exactly one file in file_list."

# Make sure SINGLE_SUBJECT_FILE actually exists in the dreamt_pibi directory:
dataset = SingleSubjectDataset(
    dir=DATASET_DIR,
    dataset="dreamt_pibi",  # matching your naming
    file_list=[SINGLE_SUBJECT_FILE],
    multiplier=dreamt_pibi_config["multiplier"],
    downsample_rate=dreamt_pibi_config["downsampling_rate"],
    task="sleep_staging"  # or "sleep_wake" if relevant
)

if len(dataset) == 0:
    raise FileNotFoundError(f"Could not find {SINGLE_SUBJECT_FILE} in {DATASET_DIR}.")

ibi, labels, num_segments, ahi = dataset[0]

X = ibi.unsqueeze(0).to(DEVICE)    # shape: [1, segments, samples_per_segment]
Y = labels.unsqueeze(0).to(DEVICE) # shape: [1, segments]

print("Single-subject IBI shape:", X.shape)
print("Single-subject labels shape:", Y.shape)
print(f"num_segments={num_segments}, AHI={ahi}")

def measure_inference_time(model, input_tensor, lengths=None, num_warmups=10, num_trials=50):
    """
    Measure average GPU inference time for a given model & input.
    If lengths is provided, the model is called as model(input_tensor, lengths).
    Returns the average time (seconds) per forward pass.
    """
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(num_warmups):
            if lengths is not None:
                _ = model(input_tensor, lengths)
            else:
                _ = model(input_tensor)
        # Synchronize before timing
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        # Timed forward passes
        for _ in range(num_trials):
            if lengths is not None:
                _ = model(input_tensor, lengths)
            else:
                _ = model(input_tensor)
        # Synchronize after
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

    return (end_time - start_time) / num_trials

for model_name, checkpoint_path in MODEL_PATHS.items():
    print("\n" + "="*80)
    print(f"Benchmarking: {model_name}")
    print("Checkpoint:", checkpoint_path)

    base_config = MODEL_CONFIGS[model_name]
    model, _ = setup_model_and_optimizer(
        model_name=model_name,
        model_params=base_config,
        device=DEVICE,
        saved_model_path=None,  # We'll manually load next
        learning_rate=base_config["LEARNING_RATE"],
        weight_decay=base_config["WEIGHT_DECAY"],
        freeze_layers=False
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)

    lengths_tensor = None
    lengths_tensor = torch.tensor([num_segments], device=DEVICE)

    avg_sec = measure_inference_time(model, X, lengths=lengths_tensor)
    avg_ms = avg_sec * 1000
    print(f"=> {model_name} average GPU inference time on 1 subject: {avg_ms:.3f} ms")

print("\nBenchmarking complete.")
