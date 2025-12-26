import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STAGE_TO_ID_5CLASS = {
    "WK": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def parse_date_from_hr_name(name: str):
    match = re.search(r"_(\d{8})", name)
    if not match:
        return None
    ymd = match.group(1)
    return datetime.strptime(ymd, "%Y%m%d").date()


def parse_date_from_label_name(name: str):
    match = re.search(r"_(\d{6})", name)
    if not match:
        return None
    yymmdd = match.group(1)
    year = 2000 + int(yymmdd[:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    return datetime(year, month, day).date()


def resolve_path(base_dir: Path, path_str: str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path_str


def load_hr_series(hr_path: Path):
    hr_map = {}
    hr_values = []
    for line in hr_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = obj.get("data", {})
        ts = data.get("deviceSendTime")
        hr = data.get("heartRate")
        if ts is None or hr is None:
            continue
        try:
            ts = int(ts)
            hr = float(hr)
        except (ValueError, TypeError):
            continue
        hr_map[ts] = hr
        if hr > 0:
            hr_values.append(hr)
    if not hr_values:
        raise ValueError(f"No valid heart rate values found in {hr_path}")
    return hr_map, float(np.median(hr_values))


def load_label_epochs(label_path: Path, start_date):
    epochs = []
    current_date = start_date
    prev_dt = None
    with label_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            time_str = row[1].strip()
            stage_str = row[2].strip().upper()
            if stage_str not in STAGE_TO_ID_5CLASS:
                continue
            dt = datetime.strptime(
                f"{current_date.isoformat()} {time_str}", "%Y-%m-%d %H:%M:%S"
            )
            if prev_dt and dt < prev_dt:
                current_date = current_date + timedelta(days=1)
                dt = datetime.strptime(
                    f"{current_date.isoformat()} {time_str}", "%Y-%m-%d %H:%M:%S"
                )
            prev_dt = dt
            epochs.append((dt, STAGE_TO_ID_5CLASS[stage_str]))
    if not epochs:
        raise ValueError(f"No label epochs parsed from {label_path}")
    return epochs


def build_epoch_ibi(hr_map, epoch_dt, fallback_hr):
    start_ts = int(epoch_dt.timestamp())
    hr_values = []
    missing = 0
    last_hr = None
    for i in range(30):
        hr = hr_map.get(start_ts + i)
        if hr is None or hr <= 0:
            missing += 1
            hr = last_hr if last_hr is not None else fallback_hr
        last_hr = hr
        hr_values.append(hr)
    hr_values = np.array(hr_values, dtype=np.float32)
    ibi_values = 60.0 / np.maximum(hr_values, 1e-6)
    ibi_25hz = np.repeat(ibi_values, 25)
    return ibi_25hz, missing


def convert_pair(hr_path: Path, label_path: Path, output_dir: Path):
    date = parse_date_from_hr_name(hr_path.name) or parse_date_from_label_name(
        label_path.name
    )
    if date is None:
        raise ValueError(f"Cannot determine start date for {hr_path} and {label_path}")

    hr_map, fallback_hr = load_hr_series(hr_path)
    epochs = load_label_epochs(label_path, date)

    ibi_segments = []
    stage_segments = []
    total_missing = 0

    for epoch_dt, stage_id in epochs:
        ibi_25hz, missing = build_epoch_ibi(hr_map, epoch_dt, fallback_hr)
        ibi_segments.append(ibi_25hz)
        stage_segments.append(np.full(ibi_25hz.shape[0], stage_id, dtype=np.int64))
        total_missing += missing

    ibi = np.concatenate(ibi_segments)
    stages = np.concatenate(stage_segments)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hr_path.stem}.npz"
    np.savez(
        output_path,
        data=ibi.astype(np.float32),
        fs=25,
        stages=stages.astype(np.int64),
        ahi=0.0,
    )

    print(
        f"Saved {output_path} | epochs={len(epochs)} | missing_seconds={total_missing}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert 1Hz HR JSON lines + PSG CSV labels into WatchSleepNet NPZ."
    )
    parser.add_argument("--input-dir", required=True, help="Base directory for input files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for NPZ files.")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("HR_FILE", "LABEL_FILE"),
        required=True,
        help="Pair of HR JSON lines file and PSG label CSV file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for hr_file, label_file in args.pair:
        hr_path = resolve_path(input_dir, hr_file)
        label_path = resolve_path(input_dir, label_file)
        convert_pair(hr_path, label_path, output_dir)


if __name__ == "__main__":
    main()
