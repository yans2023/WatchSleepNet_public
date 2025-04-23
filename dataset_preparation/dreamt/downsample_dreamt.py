import numpy as np
import os
import glob
import pandas as pd
from scipy.signal import resample, cheby2, filtfilt

# Define source (CSV files) and destination (output NPZ files) directories
source_folder = '/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes/'
dest_folder = '/mnt/nvme2/DREAMT_PPG_preprocesse/'
os.makedirs(dest_folder, exist_ok=True)

# Sampling parameters and epoch duration
orig_fs = 100.0         # assumed original sampling frequency in Hz
target_fs = 25.0        # target sampling frequency in Hz
epoch_duration = 30.0   # epoch duration in seconds

# Compute number of samples per epoch for original and target signals
orig_epoch_samples = int(orig_fs * epoch_duration)   # 256*30 = 7680 samples
new_epoch_samples = int(target_fs * epoch_duration)    # 25*30 = 750 samples

# Get list of all CSV files in the source folder
csv_files = glob.glob(os.path.join(source_folder, '*.csv'))

# Sleep stage mapping dictionary as provided
stage_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}

for file_path in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the raw PPG signal from the "BVP" column and the sleep stage strings
    data = df['BVP'].values.squeeze().astype(np.float64)
    raw_stages = df['Sleep_Stage'].values.squeeze()
    
    # Map raw sleep stage values to integer values using the provided mapping
    mapped_stages = np.array([stage_mapping.get(stage, -1) for stage in raw_stages])
    
    # --- Signal Processing Steps (following Nam et al. 2024) ---
    # 1. Filtering: Apply an 8th-order zero-phase low-pass Chebyshev Type II filter
    cutoff = 8      # cutoff frequency in Hz
    order = 8
    Rs = 40         # stop-band attenuation in dB
    nyq = 0.5 * orig_fs
    Wn = cutoff / nyq  # normalized cutoff frequency
    b, a = cheby2(order, Rs, Wn, btype='low', analog=False)
    
    # Ensure the data length is greater than the pad length required by filtfilt
    padlen_default = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen_default:
        pad_size = padlen_default + 1 - len(data)
        print(f"File {file_path}: data length {len(data)} is less than padlen {padlen_default}. Padding with {pad_size} samples.")
        data = np.pad(data, (0, pad_size), mode='edge')
    
    filtered_data = filtfilt(b, a, data)
    
    # 2. Detrending: Remove slow trends using a 10th-order polynomial fit
    x = np.arange(len(filtered_data))
    p = np.polyfit(x, filtered_data, 10)
    trend = np.polyval(p, x)
    detrended_data = filtered_data - trend
    
    # 3. Min–max normalization: Scale the detrended signal to the range [0, 1]
    min_val = detrended_data.min()
    max_val = detrended_data.max()
    normalized_data = (detrended_data - min_val) / (max_val - min_val)
    # --- End of Preprocessing ---
    
    # Trim the signal so that it only contains complete 30-second epochs
    total_samples = normalized_data.shape[0]
    num_epochs = total_samples // orig_epoch_samples
    trimmed_length = num_epochs * orig_epoch_samples
    normalized_data = normalized_data[:trimmed_length]
    mapped_stages = mapped_stages[:trimmed_length]
    
    # Reshape the continuous signals into epochs (each row is one 30-second epoch)
    data_epochs = normalized_data.reshape(num_epochs, orig_epoch_samples)
    stages_epochs = mapped_stages.reshape(num_epochs, orig_epoch_samples)
    
    # Downsample each epoch to 750 samples using Fourier resampling
    data_downsampled = resample(data_epochs, new_epoch_samples, axis=1)
    stages_downsampled = resample(stages_epochs, new_epoch_samples, axis=1)
    
    # Optionally flatten the epoch arrays back into 1D arrays
    data_ds = data_downsampled.reshape(-1)
    stages_ds = stages_downsampled.reshape(-1)
    
    # Construct the output filename: e.g., from "S002_PSG_df.csv" to "S002.npz"
    base_name = os.path.basename(file_path)
    subject_id = base_name.split('_')[0]
    out_file = os.path.join(dest_folder, f"{subject_id}.npz")
    
    # Save the processed data, mapped sleep stages, and updated sampling frequency (25 Hz)
    np.savez(out_file, data=data_ds, stages=stages_ds, fs=np.array(target_fs))
    
    print(f"Processed {base_name}: {num_epochs} epochs processed, downsampled to {new_epoch_samples} samples per epoch, saved as {subject_id}.npz")
