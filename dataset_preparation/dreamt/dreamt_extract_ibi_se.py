import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import time
import logging
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks
from pathlib import Path
from tqdm import tqdm
import warnings

"""
SE means segmented extraction
"""

def read_dreamt_data(
    file_path,
    select_chs=["BVP"],
    fixed_fs=100,
    epoch_duration=30,
    use_package="neurokit",
    downsample=False,
):
    """
    Load data from a DREAMT dataframe and extract sleep stages.
    """
    # Read data
    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}")

    # Extract BVP and sleep stages
    ecg_signal = data[select_chs[0]].values
    sleep_stages = data["Sleep_Stage"].values

    print(
        f"BVP signal length: {len(ecg_signal)}, Sleep stages length: {len(sleep_stages)}"
    )

    # Map sleep stages to integer values
    stage_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
    mapped_stages = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])

    # Remove samples labeled as "Missing"
    valid_indices = mapped_stages != -1
    ecg_signal = ecg_signal[valid_indices]
    mapped_stages = mapped_stages[valid_indices]

    print(
        f"Valid BVP signal length: {len(ecg_signal)}, Valid sleep stages length: {len(mapped_stages)}"
    )

    # Calculate the number of samples per epoch
    samples_per_epoch = fixed_fs * epoch_duration

    # Find the first and last non-wake epochs (N1, N2, N3, R)
    non_wake_epochs = np.where(np.isin(mapped_stages, [1, 2, 3, 4]))[0]
    if len(non_wake_epochs) == 0:
        raise ValueError("No valid sleep stages found in the data.")
    start_epoch = non_wake_epochs[0]
    start_idx = start_epoch // samples_per_epoch
    end_epoch = non_wake_epochs[-1]
    end_idx = end_epoch

    print(f"Start idx: {start_idx}, End idx: {end_idx}")

    # Ensure indices are within bounds
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(ecg_signal))

    # Perform trimming if necessary
    if start_idx < end_idx:
        ecg_signal = ecg_signal[start_idx:end_idx]
        mapped_stages = mapped_stages[start_idx : end_idx + 1]

    print(
        f"Trimmed BVP signal length: {len(ecg_signal)}, Trimmed sleep stages length: {len(mapped_stages)}"
    )

    valid_epochs = []
    valid_stages = []
    total_empty_segments = 0  # Initialize a counter for empty segments
    i = 0

    while i + samples_per_epoch <= len(ecg_signal):
        epoch_signal = ecg_signal[i : i + samples_per_epoch]
        epoch_stages = mapped_stages[i : i + samples_per_epoch]

        # If the epoch contains consistent stages, add it to the valid list
        if np.all(epoch_stages == epoch_stages[0]):
            epoch_signal, empty_segments = calculate_ibi_segment(
                epoch_signal, fixed_fs, use_package
            )
            total_empty_segments += empty_segments  # Accumulate empty segment counts
            valid_epochs.append(epoch_signal)
            valid_stages.extend([epoch_stages[0]] * samples_per_epoch)
            i += samples_per_epoch  # Move to the next epoch
        else:
            # Move one sample forward to correct for offset
            i += 1

    # Convert to numpy arrays
    valid_epochs = np.concatenate(valid_epochs).reshape(-1, 1)
    valid_stages = np.array(valid_stages).flatten()

    print(
        f"Valid epochs shape: {valid_epochs.shape}, Valid stages shape: {valid_stages.shape}"
    )
    print(f"Total empty segments detected: {total_empty_segments}")

    if downsample:
        valid_epochs = valid_epochs[::4]
        valid_stages = valid_stages[::4]
        fixed_fs = 25

    return valid_epochs.astype(np.float32), fixed_fs, valid_stages


def calculate_ibi_segment(ppg_signal, fs, use_package="neurokit"):
    empty_segment_count = 0  # Counter for empty segments

    if use_package == "neurokit":
        # Step 1: Preprocess and find peaks in the PPG signal
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", nk.misc.NeuroKitWarning)
            signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs)

            # Check if the specific warning was raised
            for warning in w:
                if "Too few peaks detected to compute the rate" in str(warning.message):
                    empty_segment_count += 1
                    return np.zeros_like(ppg_signal), empty_segment_count

            peaks = info["PPG_Peaks"]

            # Step 2: Calculate the Inter-Beat Intervals (IBI)
            ibi_values = np.diff(peaks) / fs  # IBI in seconds

    elif use_package == "scipy":
        from scipy.signal import find_peaks

        # Step 1: Preprocess the PPG signal if needed (e.g., filtering)
        ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=0.3*fs)

        # Step 2: Find the peaks using scipy's find_peaks
        peaks, _ = find_peaks(ppg_cleaned, distance=fs)  # Adjust distance for PPG
        if len(peaks) < 2:  # Handle the case where too few peaks are detected
            print("Warning: Too few peaks detected. Returning zeros.")
            empty_segment_count += 1
            return np.zeros_like(ppg_signal), empty_segment_count
        ibi_values = np.diff(peaks) / fs  # IBI in seconds
    else:
        raise ValueError("Package not implemented. Please use 'neurokit'.")

    # Create an array to hold the IBI values for each time point in the PPG signal
    ibi = np.zeros_like(ppg_signal)

    # Fill the IBI array with the corresponding IBI values between peaks
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]

    ibi[ibi >= 2.0] = 0.0  # Remove outliers
    return ibi, empty_segment_count


def extract_ibi_hard_code(root_path, filename):
    info_path = root_path / "participant_info.csv"
    info_df = pd.read_csv(info_path)
    # if recomputing, then we have to delete the code checking for file already existing
    try:
        in_dir = root_path / "data"
        out_dir = root_path / "DREAMT_PIBI_TEST"

        if not os.path.exists(os.path.join(out_dir, filename)):
            sid = filename.split("_")[0]
            print(sid)
            AHI = info_df.loc[info_df["SID"] == sid, "AHI"].values[0]
            print("AHI: ", AHI)

            ibi, fs, stages = read_dreamt_data(
                os.path.join(in_dir, filename),
                select_chs=["BVP"],
                fixed_fs=100,
                epoch_duration=30,
                use_package="neurokit",
                downsample=True
            )

            np.savez(
                os.path.join(out_dir, filename.split("_")[0] + ".npz"),
                data=ibi,
                fs=int(25),
                ahi = float(AHI), 
                stages=stages,
            )
    except Exception as e:
        logging.exception(f"Error processing file: {filename}")


root_path = "/mnt/linux_partition/DREAMT"
list_files = root_path / "data"
for file in tqdm(list_files.iterdir()):
    extract_ibi_hard_code(root_path, file)
