import os
import numpy as np
import biosppy
import neurokit2 as nk
import time
import logging
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks


def calculate_ibi_from_file(
    filename, measure_time=False, downsample=False, use_package="biosppy"
):
    try:
        if measure_time:
            start_time = time.time()

        ecg_data = np.load(filename)
        ecg_signal = ecg_data["data"].flatten()
        fs = ecg_data["fs"]
        if use_package == "biosppy":
            out = biosppy.signals.ecg.ecg(ecg_signal, sampling_rate=fs, show=False)
            r_peaks = out["rpeaks"]
            ibi_values = np.diff(r_peaks) / fs
        elif use_package == "neurokit":
            ibi_values = nk.ecg_intervalrelated(ecg_signal, sampling_rate=fs)
        else:
            raise ValueError(
                "Package not implemented. Please use 'biosppy' or 'neurokit'."
            )

        # Create an array to hold the IBI values for each time point in the ECG signal
        ibi = np.zeros_like(ecg_signal)

        # Fill the IBI array with the corresponding IBI values between R peaks
        for i in range(len(r_peaks) - 1):
            ibi[r_peaks[i]:r_peaks[i+1]] = ibi_values[i]
        
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
        in_dir = "/media/nvme1/sleep/ECG/SHHS/"
        out_dir = "/mnt/nvme2/SHHS_IBI/"
        if not os.path.exists(os.path.join(out_dir, filename)):
            ibi = calculate_ibi_from_file(
                os.path.join(in_dir, filename),
                measure_time=False,
                downsample=True,
                use_package="biosppy",
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
        # list_files = os.listdir("/mnt/nvme2/MESA_ECG/")
        list_files = os.listdir("/media/nvme1/sleep/ECG/SHHS/")
        num_processes = cpu_count()  # Get the number of CPU cores
        print(f"Number of CPU cores: {int(num_processes)}")

        # Create a pool of processes
        with Pool(int(num_processes / 2)) as pool:
            pool.map(extract_ibi_hard_code, list_files)
    except Exception as e:
        logging.exception("Error in parallel processing")


if __name__ == "__main__":
    logging.basicConfig(filename="extract_shhs_eibi.log", level=logging.ERROR)
    compute_all_ibis()
    # extract_ibi_hard_code("0393.npz")
