import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

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

    print(f"ECG signal length: {len(ecg_signal)}, Sleep stages length: {len(sleep_stages)}")

    # Map sleep stages to integer values
    stage_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
    mapped_stages = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])

    # Remove samples labeled as "Missing"
    valid_indices = mapped_stages != -1
    ecg_signal = ecg_signal[valid_indices]
    mapped_stages = mapped_stages[valid_indices]

    print(f"Valid ECG signal length: {len(ecg_signal)}, Valid sleep stages length: {len(mapped_stages)}")

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

    print(f"Trimmed ECG signal length: {len(ecg_signal)}, Trimmed sleep stages length: {len(mapped_stages)}")

    # Detect and correct offsets by ensuring each 30-second window is a valid epoch
    valid_epochs = []
    valid_stages = []
    i = 0
    while i + samples_per_epoch <= len(ecg_signal):
        epoch_signal = ecg_signal[i:i + samples_per_epoch]
        epoch_stages = mapped_stages[i:i + samples_per_epoch]

        # If the epoch contains consistent stages, add it to the valid list
        if np.all(epoch_stages == epoch_stages[0]):
            valid_epochs.append(epoch_signal)
            valid_stages.append(epoch_stages[0])
            i += samples_per_epoch  # Move to the next epoch
        else:
            # Move one sample forward to correct for offset
            i += 1

    valid_epochs = np.array(valid_epochs)
    valid_stages = np.array(valid_stages)

    print(f"Valid epochs shape: {valid_epochs.shape}, Valid stages shape: {valid_stages.shape}")

    return valid_epochs.astype(np.float32), fixed_fs, valid_stages

def dreamt_extraction(out_dir):
    data_path = Path("/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes")
    total_count = 0
    error_count = 0

    for file in tqdm(data_path.iterdir()):
        if file.suffix == '.csv':
            try:
                ecg_signal, fs, mapped_stages = read_dreamt_data(file)

                # Save file
                subject_id = file.stem.split("_")[0]
                save_path = Path(f"{out_dir}/{subject_id}.npz")
                np.savez(save_path, data=ecg_signal, stages=mapped_stages, fs=fs)
                total_count += 1
                print("ECG Extraction complete for:", subject_id)
            except Exception as e:
                print(f"ERROR: Exception occurred for {file.stem} with message: {str(e)}")
                error_count += 1

    print(f"Count of errors: {error_count}")
    print(f"Total count of recordings: {total_count}")

def subject_extraction(out_dir, subject_id):
    data_path = Path("/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes")
    file = data_path / f"{subject_id}_PSG_df.csv"
    if file.exists():
        try:
            ecg_signal, fs, mapped_stages = read_dreamt_data(file)
            print(np.unique(mapped_stages, return_counts=True))

            # Save file
            save_path = Path(f"{out_dir}/{subject_id}.npz")
            np.savez(save_path, data=ecg_signal, stages=mapped_stages, fs=fs)
            print(f"ECG Extraction complete for: {subject_id}")
        except Exception as e:
            print(f"ERROR: Exception occurred for {file.stem} with message: {str(e)}")
    else:
        print(f"File for subject {subject_id} not found.")

def test():
    subject_id = "S103"  # Specify the subject ID here
    subject_extraction("/media/nvme1/sleep/DREAMT_Version2/PPG_per_subject/", subject_id)

def main():
    dreamt_extraction("/media/nvme1/sleep/DREAMT_Version2/PPG_per_subject/")

if __name__ == "__main__":
    main()
