#!/usr/bin/env python3

"""
parallel_ecg_preprocessing_limited.py

- Parallel pipeline to process NPZ files with ECG data, but only load
  the first 1100 * 30 s = 33000 s of data (about 9.17 hours if fs=1,
  or 33000 * fs samples if fs is higher).
- Then run the single-subject pipeline and skip the rest of the data entirely.
- Uses n_jobs=10 for parallelization. On error, skip that file. On resume logic,
  we skip any subject that has _epoch_0000.npz already in output folder.
"""

import os
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import scipy.signal as ss
import neurokit2 as nk
from scipy.ndimage import zoom
from scipy.signal import csd


def process_single_subject_fast(
    file_path,
    output_folder,
    epoch_duration_sec=300,  # 5 min
    slide_sec=30,
    max_epochs=1100,
    sqi_threshold=0.7,
    fs_resample=4
):
    """
    Minimal single-subject pipeline:
      1) Load data from file_path, but only up to 1100 * 30 s = 33000 s => ~ 9.17 hours
      2) FIR bandpass (1.2–22 Hz).
      3) R-peak detection (NeuroKit2).
      4) Loop 5-min windows (sliding 30 s):
         - skip if stage mismatch
         - skip if SQI < threshold (placeholder)
         - local R-peaks => RR => outlier removal => resample => cross-spectral => 64x64
         - limit to 1100 accepted 5-min windows
      5) Save each epoch as <subject>_epoch_XXXX.npz (crc_spec + label).
      6) No prints, skip on error by letting the caller handle exceptions.

    We do not do debug prints in here. We'll rely on the parallel driver to skip if fail.
    """

    def fir_filter_ecg(x, fs, low_cut=1.2, high_cut=22.0, order=801):
        hp_coeff = ss.firwin(order, cutoff=low_cut, fs=fs, pass_zero=False)
        x_hp = ss.lfilter(hp_coeff, 1.0, x)
        lp_coeff = ss.firwin(order, cutoff=high_cut, fs=fs, pass_zero=True)
        return ss.lfilter(lp_coeff, 1.0, x_hp)
    
    def remove_outliers_in_rr(rr):
        L = len(rr)
        if L < 41:
            return rr
        rr_out = rr.copy()
        for i in range(20, L-20):
            seg = rr_out[i-20:i+21]
            w_clean = seg[~np.isnan(seg)]
            if len(w_clean) == 0:
                continue
            local_avg = w_clean.mean()
            if not np.isnan(rr_out[i]):
                if abs(rr_out[i]-local_avg) > 0.2*local_avg:
                    rr_out[i] = np.nan
        idx = np.arange(L)
        mask = ~np.isnan(rr_out)
        if mask.sum() < 2:
            return rr_out
        rr_out[~mask] = np.interp(idx[~mask], idx[mask], rr_out[mask])
        return rr_out
    
    def compute_crc_spectrogram(ibi_4hz, edr_4hz, n_fft=512, step=40, max_freq=0.4, fs_r=4):
        L = len(ibi_4hz)
        if L < n_fft:
            return None
        time_slices = (L - n_fft)//step + 1
        f0, P0 = csd(ibi_4hz[:n_fft], edr_4hz[:n_fft], fs=fs_r, nperseg=n_fft,
                     noverlap=0, window="hann")
        maskf = (f0 <= max_freq)
        freq_count = maskf.sum()
        out = np.zeros((freq_count, time_slices), dtype=np.float32)
        for col in range(time_slices):
            start = col*step
            seg_ibi = ibi_4hz[start:start+n_fft]
            seg_edr = edr_4hz[start:start+n_fft]
            f, Pxy = csd(seg_ibi, seg_edr, fs=fs_r, nperseg=n_fft,
                         noverlap=0, window="hann")
            out[:, col] = np.abs(Pxy)[maskf].astype(np.float32)
        return out

    # Placeholder sqi => always pass
    def compute_ecg_sqi_placeholder(_):
        return 1.0

    # ---------------------------------------------------------------
    # 1) Load data, but only up to 33000 seconds => 1100 * 30s
    # ---------------------------------------------------------------
    dataz = np.load(file_path)
    ecg = np.squeeze(dataz["data"])
    fs = float(dataz["fs"])
    stages = np.squeeze(dataz["stages"])

    # limit the samples
    # 1100 * 30 = 33000 s => #samples = 33000 * fs
    max_seconds = 1100 * 30
    max_samples = int(max_seconds * fs)
    if max_samples < len(ecg):
        ecg = ecg[:max_samples]
        stages = stages[:max_samples]

    if len(ecg) != len(stages):
        return  # mismatch => skip

    ecg_f = fir_filter_ecg(ecg, fs)

    signals, info = nk.ecg_process(ecg_f, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]
    if len(r_peaks) < 2:
        return

    total_samples = len(ecg_f)
    epoch_len = int(epoch_duration_sec * fs)
    slide_len = int(slide_sec * fs)
    base = os.path.splitext(os.path.basename(file_path))[0]

    epoch_count = 0
    i = 0
    os.makedirs(output_folder, exist_ok=True)

    while i <= total_samples - epoch_len:
        if epoch_count >= max_epochs:
            break
        start = i
        end = i + epoch_len
        stg = stages[start:end]
        # skip if stage mismatch
        if not np.all(stg == stg[0]):
            i += slide_len
            continue

        # sqi check => placeholder
        sqi_val = compute_ecg_sqi_placeholder(ecg_f[start:end])
        if sqi_val < sqi_threshold:
            i += slide_len
            continue

        # local R-peaks => local IBI
        mask_local = (r_peaks >= start) & (r_peaks < end)
        local_peaks = r_peaks[mask_local]
        if len(local_peaks) < 2:
            i += slide_len
            continue
        local_peaks_in_epoch = local_peaks - start
        rr = np.diff(local_peaks_in_epoch)/fs
        edr = ecg_f[local_peaks[:-1]]
        rr_clean = remove_outliers_in_rr(rr)
        if len(rr_clean) < 1:
            i += slide_len
            continue

        # resample at 4Hz
        rr_time = np.cumsum(rr_clean)
        rr_time -= rr_time[0]
        dur_sec = epoch_len/fs
        target_len = int(round(dur_sec * fs_resample))
        new_t = np.linspace(0, dur_sec, target_len)

        if len(rr_time)>1:
            ibi_4hz = np.interp(new_t, rr_time, rr_clean)
        else:
            ibi_4hz = np.zeros(target_len,dtype=float)

        edr_time = rr_time
        if len(edr_time)>1:
            edr_4hz = np.interp(new_t, edr_time, edr)
        else:
            edr_4hz = np.zeros(target_len,dtype=float)

        cxy = compute_crc_spectrogram(ibi_4hz, edr_4hz, n_fft=512, step=40, max_freq=0.4, fs_r=fs_resample)
        if cxy is None:
            i += slide_len
            continue
        freq_rows, time_cols = cxy.shape
        if freq_rows<2 or time_cols<2:
            i += slide_len
            continue

        from scipy.ndimage import zoom
        zf = (64/freq_rows, 64/time_cols)
        spec = zoom(cxy, zf, order=1).astype(np.float32)

        label = int(stg[0])
        out_fname = f"{base}_epoch_{epoch_count:04d}.npz"
        out_path = os.path.join(output_folder, out_fname)
        np.savez(out_path, crc_spec=spec, label=label)
        epoch_count += 1

        i += slide_len

def process_all_subjects_parallel(
    input_folder,
    output_folder,
    n_jobs=10,
    epoch_duration_sec=300,
    slide_sec=30,
    max_epochs=1100,
    sqi_threshold=0.7,
    fs_resample=4
):
    """
    Parallel approach: for each .npz in input_folder, do single-subject processing,
    skipping if an error occurs, and implementing 'resume' logic if we see
    epoch_0000.npz in output folder.

    We also only process the first 1100*30 s from each file to reduce memory usage.
    """
    import os
    import glob
    from joblib import Parallel, delayed
    from tqdm import tqdm

    os.makedirs(output_folder, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_folder, "*.npz")))

    def _job(fname):
        base = os.path.splitext(os.path.basename(fname))[0]
        # resume check => if base_epoch_0000.npz exists => skip
        test_out = os.path.join(output_folder, f"{base}_epoch_0000.npz")
        if os.path.exists(test_out):
            return  # skip, subject done

        try:
            process_single_subject_fast(
                file_path=fname,
                output_folder=output_folder,
                epoch_duration_sec=epoch_duration_sec,
                slide_sec=slide_sec,
                max_epochs=max_epochs,
                sqi_threshold=sqi_threshold,
                fs_resample=fs_resample
            )
        except:
            # skip on error
            pass

    Parallel(n_jobs=n_jobs)(
        delayed(_job)(f) for f in tqdm(file_list, desc=f"Processing with n_jobs={n_jobs}")
    )

    print(f"Done. Scanned {len(file_list)} subject files in {input_folder}")


if __name__=="__main__":
    input_folder = ".../SHHS_ECG"
    output_folder = ".../SHHS_ECG_preprocessed"

    process_all_subjects_parallel(
        input_folder=input_folder,
        output_folder=output_folder,
        n_jobs=8,              # only 10 workers => hopefully less memory usage
        epoch_duration_sec=300, # 5 min
        slide_sec=30,
        max_epochs=1100,
        sqi_threshold=0.7,
        fs_resample=4
    )
