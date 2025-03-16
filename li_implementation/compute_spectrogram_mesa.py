import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter as maxfilt
from tqdm import tqdm

def compute_ppg_spectrogram(ppg_signal, fs=256, window_size=256, noverlap=128, nfft=256, f_min=0.1, f_max=4, f_sub=1, plot=False):
    """
    Compute a spectrogram from a PPG signal.
    
    Parameters:
      ppg_signal: 1D numpy array of PPG data.
      fs: Sampling rate in Hz.
      window_size: Number of samples per window.
      noverlap: Number of overlapping samples between windows.
      nfft: Number of FFT points.
      f_min: Lower frequency bound (Hz).
      f_max: Upper frequency bound (Hz).
      f_sub: Sub-sampling factor for frequency bins.
      plot: If True, plot the spectrogram.
    
    Returns:
      spec: 2D numpy array with shape (time, frequency) (log-scaled)
    """
    # Pad the signal to reduce edge effects.
    pad = window_size // 2
    padded_signal = np.zeros(ppg_signal.shape[0] + window_size)
    padded_signal[pad:pad + ppg_signal.shape[0]] = ppg_signal
    padded_signal += np.random.normal(scale=1e-10, size=padded_signal.shape)
    
    # Compute the spectrogram using a Blackman window.
    f, t, S = spectrogram(
        padded_signal,
        fs=fs,
        window=np.blackman(window_size),
        nperseg=window_size,
        noverlap=noverlap,
        nfft=nfft,
        mode='magnitude'
    )
    
    # Keep only the frequencies of interest.
    freq_mask = (f > f_min) & (f <= f_max)
    f_filtered = f[freq_mask]
    S = S[freq_mask, :]
    
    # Optionally apply a maximum filter for smoothing.
    S = maxfilt(np.abs(S), size=(f_sub, 1))
    
    # Optionally sub-sample the frequency axis.
    if f_sub > 1:
        S = S[::f_sub, :]
    
    # Transpose so that rows are time and columns are frequency.
    spec = np.swapaxes(S, 0, 1)
    
    # Apply logarithmic scaling.
    spec = np.log(spec + 1e-10)
    
    if plot:
        plt.figure(figsize=(10, 4))
        extent = [t[0], t[-1], f_filtered[0], f_filtered[-1]]
        plt.imshow(spec.T, aspect='auto', origin='lower', extent=extent, cmap='magma')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('PPG Spectrogram')
        plt.colorbar(label='Log Amplitude')
        plt.show()
    
    return spec

def process_all_subjects(input_folder, output_folder):
    # Create the output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # List all NPZ files in the input folder.
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
    
    for filename in tqdm(file_list, desc="Processing subjects"):
        file_path = os.path.join(input_folder, filename)
        
        try:
            # Load NPZ file.
            npz_file = np.load(file_path)
            # The "data" field contains the PPG signal.
            ppg_signal = np.squeeze(npz_file["data"])  # Flatten to 1D
            fs = float(npz_file["fs"])  # e.g., 256
            
            # Compute the spectrogram.
            spec = compute_ppg_spectrogram(ppg_signal, fs=fs,
                                           window_size=256,
                                           noverlap=128,
                                           nfft=256,
                                           f_min=0.1,
                                           f_max=4,
                                           f_sub=1,
                                           plot=False)
            
            # Construct output filename.
            # Example: "mesa-0001_spec.npy"
            base, ext = os.path.splitext(filename)
            out_filename = f"{base}_spec.npy"
            out_path = os.path.join(output_folder, out_filename)
            
            # Save the spectrogram.
            np.save(out_path, spec)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "/mnt/linux_partition/MESA_PPG"
    output_folder = "/mnt/linux_partition/MESA_PPG_spec"
    process_all_subjects(input_folder, output_folder)
