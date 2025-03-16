#!/usr/bin/env python3

"""
combine_small_npz_per_subject.py

This script scans a directory of NPZ epoch files, infers a subject ID from
each filename, merges all epoch files for that subject into one larger NPZ,
applies a label mapping, filters out epochs with invalid labels (i.e. label -1),
and prints the unique counts for each class (0, 1, and 2) per subject.
The resulting combined NPZ for each subject is stored in an output directory.

Usage:
  python3 combine_small_npz_per_subject.py \
      --input_dir /path/to/epoch_npz \
      --output_dir /path/to/combined_npz

It assumes each small .npz has at least:
  - 'crc_spec' (shape [64,64]) 
  - 'label' (integer)

Filename pattern is assumed to be like:
  subjectID_epoch_0000.npz
We parse the "subjectID" by splitting on "_epoch_".
"""

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

def parse_subject_id(fname):
    """
    Given a filename like 'shhs-200001_epoch_0082.npz',
    return 'shhs-200001'.
    Adjust if your naming pattern differs.
    """
    base = os.path.splitext(os.path.basename(fname))[0]  # e.g. shhs-200001_epoch_0082
    parts = base.split("_epoch_")
    if len(parts) == 2:
        return parts[0]  # 'shhs-200001'
    else:
        return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with small .npz epoch files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store merged subject-level .npz files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    print(f"Found {len(all_files)} .npz epoch files in {args.input_dir}")

    # Define label mapping
    label_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: -1, -1: -1}

    # Group files by subject ID
    subj_dict = {}
    for fpath in all_files:
        sid = parse_subject_id(fpath)
        if sid not in subj_dict:
            subj_dict[sid] = []
        subj_dict[sid].append(fpath)

    print(f"Identified {len(subj_dict)} unique subjects from filenames.")

    # For each subject, merge NPZ files and filter invalid epochs
    for sid, file_list in tqdm(subj_dict.items(), desc="Combining subjects"):
        X_list = []
        y_list = []
        for fpath in file_list:
            dataz = np.load(fpath)
            spect = dataz["crc_spec"]  # shape (64,64)
            raw_label = int(dataz["label"])
            # Map the raw label using label_map
            final_label = label_map.get(raw_label, -1)
            # Filter out epochs with invalid label (-1)
            if final_label == -1:
                continue
            X_list.append(spect)
            y_list.append(final_label)
        
        # If no valid epochs for this subject, skip saving
        if len(X_list) == 0:
            print(f"Subject {sid} has no valid epochs; skipping.")
            continue

        X_arr = np.stack(X_list, axis=0)  # shape (N,64,64)
        y_arr = np.array(y_list, dtype=np.int32)  # shape (N,)

        # Compute and print unique counts for classes 0, 1, 2
        unique, counts = np.unique(y_arr, return_counts=True)
        count_dict = dict(zip(unique, counts))
        # Ensure counts for each class are present
        count_0 = count_dict.get(0, 0)
        count_1 = count_dict.get(1, 0)
        count_2 = count_dict.get(2, 0)
        print(f"Subject {sid}: counts -> 0: {count_0}, 1: {count_1}, 2: {count_2}")

        out_fname = f"{sid}_combined.npz"
        out_path  = os.path.join(args.output_dir, out_fname)
        np.savez_compressed(out_path, X=X_arr, y=y_arr)
    
    print("Done. Combined all subjects into single .npz files each.")

if __name__ == "__main__":
    main()
