import pandas as pd
import numpy as np

def read_dreamt_data(file_path, select_chs=["BVP"], fixed_fs=100, epoch_duration=30):
    """
    Load data from a DREAMT dataframe and extract sleep stages.

    Parameters
    -----------
        file_path: path to the dataframe file
        select_chs: list of channels to extract
        fixed_fs: fixed sampling frequency
        epoch_duration: duration of each epoch in seconds

    Returns
    ----------
        a tuple containing a ndarray of ecg data, the sampling frequency,
        and an ndarray of sleep stages
    """
    # Read data
    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}")

    # Extract ECG and sleep stages
    ecg_signal = data[select_chs[0]].values
    sleep_stages = data["Sleep_Stage"].values

    print(f"BVP signal length: {len(ecg_signal)}, Sleep stages length: {len(sleep_stages)}")

    # Map sleep stages to integer values
    stage_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
    mapped_stages = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])

    # Remove samples labeled as "Missing"
    valid_indices = mapped_stages != -1
    ecg_signal = ecg_signal[valid_indices]
    mapped_stages = mapped_stages[valid_indices]

    print(f"Valid BVP signal length: {len(ecg_signal)}, Valid sleep stages length: {len(mapped_stages)}")

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
        mapped_stages = mapped_stages[start_idx:end_idx + 1]

    print(f"Trimmed BVP signal length: {len(ecg_signal)}, Trimmed sleep stages length: {len(mapped_stages)}")

    valid_epochs = []
    valid_stages = []
    i = 0

    while i + samples_per_epoch <= len(ecg_signal):
        epoch_signal = ecg_signal[i:i + samples_per_epoch]
        epoch_stages = mapped_stages[i:i + samples_per_epoch]

        # If the epoch contains consistent stages, add it to the valid list
        if np.all(epoch_stages == epoch_stages[0]):
            valid_epochs.append(epoch_signal)
            valid_stages.extend([epoch_stages[0]] * samples_per_epoch)
            i += samples_per_epoch  # Move to the next epoch
        else:
            # Move one sample forward to correct for offset
            i += 1

    # Convert to numpy arrays
    valid_epochs = np.concatenate(valid_epochs).reshape(-1, 1)
    valid_stages = np.array(valid_stages).reshape(-1, 1)

    print(f"Valid epochs shape: {valid_epochs.shape}, Valid stages shape: {valid_stages.shape}")

    return valid_epochs.astype(np.float32), fixed_fs, valid_stages