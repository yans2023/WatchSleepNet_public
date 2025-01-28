import os
from tqdm import tqdm
from utils import read_edf_data, save_to_npz

def process_mesa(out_dir):
    mesa_dir = "/mnt/linux_partition/mesa/polysomnography/edfs/"
    files = [f for f in os.listdir(mesa_dir) if f.endswith(".edf")]
    for file in tqdm(files):
        sid = file.split("-")[-1].split(".")[0]
        data_path = os.path.join(mesa_dir, file)
        label_path = f"/mnt/linux_partition/mesa/polysomnography/annotations-events-profusion/{file.split('.')[0]}-profusion.xml"
        try:
            data, fs, stages = read_edf_data(data_path, label_path, dataset="MESA", select_chs=["Pleth"])
            save_to_npz(f"{out_dir}mesa-{sid}.npz", data, stages, fs)
        except Exception as e:
            print(f"Error processing {sid}: {e}")

if __name__ == "__main__":
    process_mesa("/mnt/linux_partition/MESA_PPG/")
