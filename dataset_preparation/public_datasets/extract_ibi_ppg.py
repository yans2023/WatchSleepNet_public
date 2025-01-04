import os
import numpy as np
import biosppy
import neurokit2 as nk
import time
import logging
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks

def calculate_ibi_from_ppg(
    filename, measure_time=False, downsample=False, use_package="neurokit"
):
    try:
        if measure_time:
            start_time = time.time()

        # Load the PPG data from the file
        ppg_data = np.load(filename)
        ppg_signal = ppg_data["data"].flatten()
        fs = int(ppg_data["fs"].item())

        if use_package == "neurokit":
            # Step 1: Preprocess and find peaks in the PPG signal
            signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs)
            peaks = info["PPG_Peaks"]

            # Step 2: Calculate the Inter-Beat Intervals (IBI)
            ibi_values = np.diff(peaks) / fs  # IBI in seconds
        elif use_package == "scipy":
            from scipy.signal import find_peaks

            # Step 1: Preprocess the PPG signal if needed (e.g., filtering)
            ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=fs)

            # Step 2: Find the peaks using scipy's find_peaks
            peaks, _ = find_peaks(
                ppg_cleaned, distance=fs * 0.6
            )  # Adjust distance for PPG
            ibi_values = np.diff(peaks) / fs  # IBI in seconds
        else:
            raise ValueError(
                "Package not implemented. Please use 'neurokit' or 'scipy'."
            )

        # Create an array to hold the IBI values for each time point in the PPG signal
        ibi = np.zeros_like(ppg_signal)

        # Fill the IBI array with the corresponding IBI values between peaks
        for i in range(len(peaks) - 1):
            ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]
        
        ibi[ibi >= 2.0] = 0.0 

        if downsample:
            # as a rule, downsample to 25 Hz
            ibi = ibi[:: int(fs / 25)]

        if measure_time:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed Time:", elapsed_time, "seconds")
        return ibi

    except Exception as e:
        logging.exception(f"Error processing file: {filename}")
        return None


def extract_ibi_hard_code(filename):
    # if recomputing, then we have to delete the code checking for file already existing
    try:
        in_dir = "/mnt/nvme2/MESA_PPG/"
        out_dir = "/mnt/nvme2/MESA_PIBI/"
        if not os.path.exists(os.path.join(out_dir, filename)):
            ibi = calculate_ibi_from_ppg(
                os.path.join(in_dir, filename),
                measure_time=False,
                downsample=True,
                use_package="neurokit",
            )
            if ibi is not None:
                ecg_data = np.load(os.path.join(in_dir, filename))
                np.savez(
                    os.path.join(out_dir, filename),
                    data=ibi,
                    fs=int(25),
                    stages=ecg_data["stages"][:: int(ecg_data["fs"] / 25)],
                )
    except Exception as e:
        logging.exception(f"Error processing file: {filename}")


def compute_all_ibis():  
    try:
        list_files = os.listdir("/mnt/nvme2/MESA_PPG/")
        num_processes = cpu_count()  # Get the number of CPU cores
        print(f"Number of CPU cores: {int(num_processes)}")

        # Create a pool of processes
        with Pool(int(num_processes / 2)) as pool:
            pool.map(extract_ibi_hard_code, list_files)
    except Exception as e:
        logging.exception("Error in parallel processing")


if __name__ == "__main__":
    logging.basicConfig(filename="extract_mesa_pibi.log", level=logging.ERROR)
    compute_all_ibis()
    # extract_ibi_hard_code("mesa-0393.npz")
