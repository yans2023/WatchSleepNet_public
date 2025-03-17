import os
import numpy as np
import biosppy
import neurokit2 as nk
import logging
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks
import pandas as pd
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO)


def calculate_ibi_ecg(signal, fs):
    """
    Faithfully reproduces the old "biosppy" ECG approach:
      1) Run biosppy ECG pipeline.
      2) Compute IBI between consecutive R-peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    out = biosppy.signals.ecg.ecg(signal, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]  # indices of R-peaks
    ibi_values = np.diff(r_peaks) / fs  # IBI in seconds

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(r_peaks) - 1):
        ibi[r_peaks[i] : r_peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def calculate_ibi_ppg(signal, fs):
    """
    Faithfully reproduces the old "neurokit" PPG approach:
      1) Use neurokit2 PPG pipeline.
      2) Compute IBI between consecutive PPG peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    signals, info = nk.ppg_process(signal, sampling_rate=fs)
    peaks = info["PPG_Peaks"]  # indices of PPG peaks
    ibi_values = np.diff(peaks) / fs

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def process_file(
    filename,
    in_dir,
    out_dir,
    info_df,
    id_col,
    ahi_col,
    fs_col,
    method,
    dataset_name
):
    """
    Loads .npz file from 'in_dir', computes continuous IBI array using the
    EXACT old code approach:
      - 'biosppy' for ECG (SHHS)
      - 'neurokit' for PPG (MESA)
    Then it downsamples from fs to 25 Hz via a smarter approach:
      - integer slicing if original_fs is a multiple of 25
      - polyphase resampling (scipy.signal.resample_poly) otherwise.
    Saves the new NPZ file (IBI float64, fs=25, downsampled stages, AHI).
    """
    file_path = os.path.join(in_dir, filename)

    try:
        data = np.load(file_path)
        signal = data["data"].flatten()  # original waveform
        original_fs = int(data[fs_col].item())  # e.g., 125, 250, 256, etc.
        stages = data["stages"]

        # Compute IBI using the old code approach
        if method == "biosppy":  # ECG for SHHS
            ibi = calculate_ibi_ecg(signal, original_fs)
        elif method == "neurokit":  # PPG for MESA
            ibi = calculate_ibi_ppg(signal, original_fs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if ibi is None:
            return

        # Parse subject ID from filename
        if dataset_name == "SHHS":
            file_id = filename.split("-")[1].split(".npz")[0].lstrip("0")
        else:  # MESA or other
            if filename.startswith("mesa-"):
                # Remove the prefix "mesa-" and the suffix ".npz"
                file_id = filename[len("mesa-") : -len(".npz")]
            else:
                file_id = filename.split(".npz")[0]

        # Retrieve AHI from info_df
        matching_rows = info_df[info_df[id_col] == file_id.lstrip("0")]
        if matching_rows.empty:
            logging.warning(f"AHI not found for {filename} in dataset {dataset_name}.")
            return
        ahi = float(matching_rows[ahi_col].values[0])

        # Smarter downsampling to 25 Hz
        target_fs = 25
        if original_fs % target_fs == 0:
            # When original_fs is an integer multiple of 25, simple slicing works
            factor = int(original_fs // target_fs)
            ibi_ds = ibi[::factor]
            stages_ds = stages[::factor]
        else:
            # For non-integer resampling factors, use polyphase resampling
            ibi_ds = resample_poly(ibi, target_fs, original_fs)
            stages_ds = resample_poly(stages, target_fs, original_fs)

        # Construct the output filename
        if dataset_name == "MESA":
            out_filename = f"mesa-{file_id}.npz"
        else:
            out_filename = filename

        # Save the processed file
        out_path = os.path.join(out_dir, out_filename)
        np.savez(
            out_path,
            data=ibi_ds.astype(np.float64),
            fs=target_fs,
            stages=stages_ds,
            ahi=ahi
        )
        logging.info(f"{dataset_name} processed & saved => {out_filename}")

    except Exception as e:
        logging.exception(f"Error processing file: {filename} in {dataset_name}")


def process_dataset(
    dataset_name,
    in_dir,
    out_dir,
    info_path,
    id_col,
    ahi_col,
    fs_col,
    method
):
    """
    Main function:
      - Load dataset's CSV (for AHI, etc.).
      - Create out_dir.
      - Process each .npz file in parallel.
    """
    try:
        logging.info(f"=== Processing {dataset_name} dataset ===")

        # Load CSV info
        info_df = pd.read_csv(info_path, dtype={id_col: str})
        info_df[id_col] = info_df[id_col].str.lstrip("0")

        # Make sure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # List .npz files
        file_list = [f for f in os.listdir(in_dir) if f.endswith(".npz")]
        logging.info(f"{dataset_name}: Found {len(file_list)} .npz files in {in_dir}")

        # Decide how many CPU cores to use
        num_processes = max(1, cpu_count() // 2)
        logging.info(f"{dataset_name}: Using {num_processes} processes")

        # Prepare arguments
        args_list = [
            (
                f,
                in_dir,
                out_dir,
                info_df,
                id_col,
                ahi_col,
                fs_col,
                method,
                dataset_name
            )
            for f in file_list
        ]

        # Process in parallel
        with Pool(num_processes) as pool:
            pool.starmap(process_file, args_list)

        logging.info(f"=== Completed processing {dataset_name} ===\n")

    except Exception as e:
        logging.exception(f"Error processing dataset: {dataset_name}")


if __name__ == "__main__":
    """
    This script processes two datasets in one go:
      1) SHHS (ECG) via biosppy
      2) MESA (PPG) via neurokit

    Both resulting NPZ files (one for each record) will be placed in
    the SAME output directory, /mnt/linux_partition/SHHS_MESA_IBI/.
    """
    logging.info("Starting dataset processing...")
    output_dir = "/home/willkewang/Datasets/SHHS_MESA_IBI_new/" # TO fill in: path to the output directory, which needs to be made

    shhs_input_dir = "/mnt/linux_partition/SHHS_ECG/" # TO fill in: path to the SHHS extracted ECG dataset
    shhs_info_path = "/mnt/linux_partition/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv" # TO fill in: path to the SHHS info CSV file
    shhs_id_col = "nsrrid"
    shhs_ahi_col = "nsrr_ahi_hp3r_aasm15"
    shhs_fs_col = "fs"
    shhs_method = "biosppy"  # ECG from SHHS

    process_dataset(
        dataset_name="SHHS",
        in_dir=shhs_input_dir,
        out_dir=output_dir,
        info_path=shhs_info_path,
        id_col=shhs_id_col,
        ahi_col=shhs_ahi_col,
        fs_col=shhs_fs_col,
        method=shhs_method,
    )

    mesa_input_dir = "/mnt/linux_partition/MESA_PPG/" # TO fill in: path to the MESA extracted PPG dataset
    mesa_info_path = "/mnt/linux_partition/mesa/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv" # TO fill in: path to the MESA info CSV file
    mesa_id_col = "mesaid"
    mesa_ahi_col = "nsrr_ahi_hp3u"
    mesa_fs_col = "fs"
    mesa_method = "neurokit"  # PPG from MESA

    process_dataset(
        dataset_name="MESA",
        in_dir=mesa_input_dir,
        out_dir=output_dir,
        info_path=mesa_info_path,
        id_col=mesa_id_col,
        ahi_col=mesa_ahi_col,
        fs_col=mesa_fs_col,
        method=mesa_method,
    )

    logging.info("All done! SHHS + MESA IBI files are now in the same folder.")
