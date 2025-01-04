import os
import numpy as np
import pandas as pd

# Paths
npz_dir = "/mnt/nvme2/SHHS_MESA_IBI/"
shhs_info_dir = (
    "/media/nvme1/sleep/shhs/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv"
)

# Load the SHHS harmonized dataset CSV
shhs_info_df = pd.read_csv(shhs_info_dir)

# Extract relevant columns: nsrrid, nsrr_ahi_hp3r_aasm15, and visitnumber
shhs_info_df = shhs_info_df[["nsrrid", "nsrr_ahi_hp3r_aasm15", "visitnumber"]]

# Convert nsrrid to string to match with filenames
shhs_info_df["nsrrid"] = shhs_info_df["nsrrid"].astype(str)

# Iterate through the npz files in the npz_dir
for file_name in os.listdir(npz_dir):
    if file_name.endswith(".npz") and (
        file_name.startswith("shhs1-") or file_name.startswith("shhs2-")
    ):
        # Extract the visit number and nsrrid from the filename
        nsrrid = file_name.split("-")[1].split(".npz")[0]
        visit = 1 if file_name.startswith("shhs1-") else 2

        # Get the corresponding AHI value from the CSV based on nsrrid and visitnumber
        ahi_value = shhs_info_df.loc[
            (shhs_info_df["nsrrid"] == nsrrid) & (shhs_info_df["visitnumber"] == visit),
            "nsrr_ahi_hp3r_aasm15",
        ]

        if not ahi_value.empty:
            ahi = float(ahi_value.values[0])  # Convert AHI to float
        else:
            print(f"AHI not found for nsrrid: {nsrrid}, visit: {visit}")
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

print("All files updated successfully.")
