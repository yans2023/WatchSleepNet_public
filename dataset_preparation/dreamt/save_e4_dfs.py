import os
import pandas as pd

# Define the input and output directories
input_dir = "/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes"
output_dir = "/media/nvme1/sleep/DREAMT_Version2/E4_dataframes"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the columns to keep
columns_to_keep = [
    "TIMESTAMP", "ECG", "BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI",
    "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"
]

# Iterate through the files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        try:
            # Read the CSV file
            df = pd.read_csv(input_file_path)

            # Filter the dataframe to keep only the specified columns
            filtered_df = df[columns_to_keep]

            # Save the filtered dataframe to the output directory
            filtered_df.to_csv(output_file_path, index=False)

            print(f"Successfully processed {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")