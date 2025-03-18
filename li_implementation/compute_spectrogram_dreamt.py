import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_dreamt_psg_file(file_path, default_fs=100):
    """
    Reads a DREAMT_PSG CSV file and extracts the raw PPG signal and sleep stages.
    
    Expected CSV columns include:
      - "BVP": raw photoplethysmography signal.
      - "Sleep_Stage": sleep stage annotations.
    
    Sleep stages are mapped using:
      {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
      
    Returns:
      bvp: NumPy array of shape (N, 1) containing the raw PPG signal.
      fs: Sampling frequency (default_fs, if not provided in the file).
      stages: NumPy array of shape (N,) containing sleep stage annotations.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    if "BVP" not in df.columns:
        print(f"'BVP' column not found in {file_path}")
        return None
    
    # Extract raw PPG signal
    bvp = df["BVP"].values.astype(np.float32)
    
    # Map sleep stages from "Sleep_Stage" column
    stage_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
    if "Sleep_Stage" in df.columns:
        sleep_stages = df["Sleep_Stage"].values.astype(str)
        mapped_stages = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages], dtype=np.int8)
    else:
        mapped_stages = np.zeros(bvp.shape[0], dtype=np.int8)
    
    fs = default_fs  # Default sampling rate is set to 100 Hz
    bvp = bvp.reshape(-1, 1)
    
    return bvp, fs, mapped_stages

def process_all_dreamt_psg(input_folder, output_folder, default_fs=100):
    """
    Processes all CSV files in the input_folder, extracts the raw PPG signal (BVP)
    and sleep stage annotations, then saves the result in NPZ format in the output_folder.
    
    The saved NPZ files will contain:
      'data'   -> raw PPG signal (N x 1)
      'fs'     -> sampling frequency (scalar)
      'stages' -> sleep stage annotations (N,)
    """
    os.makedirs(output_folder, exist_ok=True)
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for fname in tqdm(file_list, desc="Processing DREAMT_PSG files"):
        file_path = os.path.join(input_folder, fname)
        result = process_dreamt_psg_file(file_path, default_fs=default_fs)
        if result is None:
            continue
        bvp, fs, stages = result
        base, _ = os.path.splitext(fname)
        out_fname = f"{base}.npz"
        out_path = os.path.join(output_folder, out_fname)
        np.savez(out_path, data=bvp, fs=fs, stages=stages)

if __name__ == "__main__":
    input_folder = ".../DREAMT_PSG" 
    output_folder = ".../DREAMT_PPG"  
    process_all_dreamt_psg(input_folder, output_folder, default_fs=100)
