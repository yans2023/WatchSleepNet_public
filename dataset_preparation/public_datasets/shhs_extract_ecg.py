import os
from tqdm import tqdm
from utils import read_edf_data, save_to_npz

def process_shhs(out_dir):
    shhs_dirs = [
        "/mnt/linux_partition/shhs/polysomnography/edfs/shhs1/",
        "/mnt/linux_partition/shhs/polysomnography/edfs/shhs2/"
    ]
    for shhs_dir in shhs_dirs:
        files = [f for f in os.listdir(shhs_dir) if f.endswith(".edf")]
        for file in tqdm(files):
            sid = file.split("-")[1].split(".")[0]
            data_path = os.path.join(shhs_dir, file)
            print(data_path)
            label_path = f"/mnt/linux_partition/shhs/polysomnography/annotations-events-profusion/{os.path.basename(os.path.normpath(shhs_dir))}/{file.split('.')[0]}-profusion.xml"
            try:
                data, fs, stages = read_edf_data(data_path, label_path, dataset="SHHS", select_chs=["ECG"])
                save_to_npz(f"{out_dir}shhs-{sid}.npz", data, stages, fs)
            except Exception as e:
                print(f"Error processing {sid}: {e}")

if __name__ == "__main__":
    process_shhs("/mnt/linux_partition/SHHS_ECG/")
