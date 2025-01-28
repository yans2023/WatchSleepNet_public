import mne
import numpy as np
from scipy.signal import resample
from xml.etree import ElementTree as ET

def read_edf_data(data_path, label_path, dataset, select_chs, target_fs=None):
    """
    Reads EDF data, extracts the specified channels, downsample data, and retrieves sleep stages.

    Parameters:
    - data_path: Path to the EDF file.
    - label_path: Path to the XML file with annotations.
    - dataset: String ("SHHS" or "MESA") to specify dataset-specific logic.
    - select_chs: List of channels to extract (e.g., ["ECG"] or ["Pleth"]).
    - target_fs: Desired sampling frequency for downsampling.

    Returns:
    - data: Extracted signal data.
    - fs: Sampling frequency of the output data.
    - stages: Sleep stage annotations aligned with the data.
    """

    # Dataset-specific channel exclusions
    if dataset == "SHHS":
        exclude_chs = ["SaO2", "H.R.", "SOUND", "AIRFLOW", "POSITION", "LIGHT"]
    elif dataset == "MESA":
        exclude_chs = ["EEG1", "EEG2", "Snore", "Thor", "Abdo", "Leg", "Therm", "Pos"]
    else:
        raise ValueError("Unsupported dataset. Use 'SHHS' or 'MESA'.")
    
    # Read EDF file
    raw_data = mne.io.read_raw_edf(data_path, verbose=0, exclude=exclude_chs, infer_types=False)

    raw_data.pick(picks=select_chs)
    original_fs = raw_data.info["sfreq"]
    data = raw_data.get_data().T

    # Downsample if needed
    if target_fs and original_fs > target_fs:
        data = resample(data, int(len(data) * target_fs / original_fs), axis=0)

    # Parse sleep stage annotations
    tree = ET.parse(label_path)
    root = tree.getroot()
    stages = np.array([stage.text for stage in root[4].findall("SleepStage")], dtype=np.int8)

    # Merge stages 3 and 4, adjust stage 5
    stages[stages == 4] = 3
    stages[stages == 5] = 4
    stages = np.repeat(stages, 30 * target_fs if target_fs else 30 * original_fs)

    # Ensure alignment of data and stages
    min_length = min(len(data), len(stages))
    data = data[:min_length]
    stages = stages[:min_length]

    return data, target_fs if target_fs else original_fs, stages

def save_to_npz(out_path, data, stages, fs):
    """
    Saves data, sleep stages, and sampling frequency to an NPZ file.
    """
    np.savez(out_path, data=data, stages=stages, fs=fs)
    print(f"Saved: {out_path}")
