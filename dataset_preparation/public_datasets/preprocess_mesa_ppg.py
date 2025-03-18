import numpy as np
import os
import glob
from scipy.signal import resample, cheby2, filtfilt

# Define source and destination directories
source_folder = '.../MESA_PPG/'
dest_folder = '.../MESA_PPG/'

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Define sampling frequencies and epoch duration
orig_fs = 256.0         # original sampling frequency in Hz
target_fs = 25.0        # target sampling frequency in Hz
epoch_duration = 30.0   # epoch duration in seconds

# Calculate number of samples per epoch for original and target signals
orig_epoch_samples = int(orig_fs * epoch_duration)   # 256 * 30 = 7680 samples
new_epoch_samples = int(target_fs * epoch_duration)    # 25 * 30 = 750 samples

# Get list of all npz files in the source folder
file_list = glob.glob(os.path.join(source_folder, '*.npz'))

for file_path in file_list:
    # Load the npz file
    npzfile = np.load(file_path)
    data = npzfile['data']
    stages = npzfile['stages']
    
    # Ensure that data and stages are 1D arrays
    data = np.squeeze(data).flatten()
    stages = np.squeeze(stages).flatten()
    
    # --- Signal Processing Steps (following Nam et al. 2024) ---
    # 1. Filtering: Apply an 8th-order zero-phase low-pass Chebyshev Type II filter 
    cutoff = 8      # cutoff frequency in Hz
    order = 8
    Rs = 40         # stop-band attenuation in dB
    nyq = 0.5 * orig_fs
    Wn = cutoff / nyq  # normalized cutoff frequency
    b, a = cheby2(order, Rs, Wn, btype='low', analog=False)
    
    # Determine the default pad length used by filtfilt
    padlen_default = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen_default:
        pad_size = padlen_default + 1 - len(data)
        print(f"Data length {len(data)} is less than required padlen {padlen_default}. Padding with {pad_size} samples.")
        data = np.pad(data, (0, pad_size), mode='edge')
    
    filtered_data = filtfilt(b, a, data)
    
    # 2. Detrending: Remove slow trends using a 10th-order polynomial
    x = np.arange(len(filtered_data))
    p = np.polyfit(x, filtered_data, 10)
    trend = np.polyval(p, x)
    detrended_data = filtered_data - trend
    
    # 3. Min–max normalization: Scale the signal to the range [0, 1]
    min_val = detrended_data.min()
    max_val = detrended_data.max()
    normalized_data = (detrended_data - min_val) / (max_val - min_val)
    # --- End of Preprocessing ---
    
    # Ensure the total number of samples is a multiple of the epoch length
    total_samples = normalized_data.shape[0]
    num_epochs = total_samples // orig_epoch_samples
    trimmed_length = num_epochs * orig_epoch_samples
    normalized_data = normalized_data[:trimmed_length]
    stages = stages[:trimmed_length]
    
    # Reshape the continuous signals into epochs (each row is one 30-second epoch)
    data_epochs = normalized_data.reshape(num_epochs, orig_epoch_samples)
    stages_epochs = stages.reshape(num_epochs, orig_epoch_samples)
    
    # Downsample each epoch using Fourier resampling to maintain the 30-second alignment
    data_downsampled = resample(data_epochs, new_epoch_samples, axis=1)
    stages_downsampled = resample(stages_epochs, new_epoch_samples, axis=1)
    
    # Optionally flatten back to 1D arrays
    data_ds = data_downsampled.reshape(-1)
    stages_ds = stages_downsampled.reshape(-1)
    
    # Save the processed data and update the sampling frequency to target_fs (25 Hz)
    new_fs_arr = np.array(target_fs)
    file_name = os.path.basename(file_path)
    dest_file_path = os.path.join(dest_folder, file_name)
    np.savez(dest_file_path, data=data_ds, stages=stages_ds, fs=new_fs_arr)
    
    print(f"Processed {file_name}: {num_epochs} epochs processed, downsampled to {new_epoch_samples} samples per epoch (target {target_fs} Hz), and saved.")
