import os
import numpy as np
import pandas as pd

# Paths
npz_dir = "/mnt/nvme2/SHHS_MESA_IBI/"
mesa_info_dir = (
    "/media/nvme1/sleep/mesa/mesa/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv"
)

# Load the MESA harmonized dataset CSV
mesa_info_df = pd.read_csv(mesa_info_dir)

# Extract relevant columns: mesaid and nsrr_ahi_hp3u
mesa_info_df = mesa_info_df[["mesaid", "nsrr_ahi_hp3u"]]

# Convert mesaid to string to match with filenames
mesa_info_df["mesaid"] = mesa_info_df["mesaid"].astype(str).str.lstrip("0")

# Iterate through the npz files in the npz_dir for MESA files
for file_name in os.listdir(npz_dir):
    if file_name.endswith(".npz") and file_name.startswith("mesa-"):
        # Extract the mesaid from the filename
        mesaid = file_name.split("-")[1].split(".npz")[0].lstrip("0")

        # Get the corresponding AHI value from the CSV
        ahi_value = mesa_info_df.loc[mesa_info_df["mesaid"] == mesaid, "nsrr_ahi_hp3u"]

        if not ahi_value.empty:
            ahi = float(ahi_value.values[0])  # Convert AHI to float
        else:
            print(f"AHI not found for mesaid: {mesaid}")
            continue

        # Load the existing npz file
        npz_file_path = os.path.join(npz_dir, file_name)
        data = np.load(npz_file_path)

        # Load the arrays directly from the npz file
        data_array = data["data"]
        stages = data["stages"]
        fs = data["fs"]

        # Save the updated npz file with the AHI field
        np.savez(npz_file_path, data=data_array, stages=stages, fs=fs, ahi=ahi)

        print(f"Updated file: {file_name} with AHI: {ahi}")

print("All MESA files updated successfully.")
