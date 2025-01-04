import mne
import datetime
import pandas as pd
import numpy as np
from read_raw_e4 import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from read_raw_e4 import *
from utils import *
from tqdm import tqdm
from scipy import signal
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PSG_CHANNELS = [
    "C4-M1",
    "F4-M1",
    "O2-M1",
    "Fp1-O2",
    "T3 - CZ",
    "CZ - T4",
    "CHIN",
    "E1",
    "E2",
    "ECG",
    "LAT",
    "RAT",
    "SNORE",
    "PTAF",
    "FLOW",
    "THORAX",
    "ABDOMEN",
    "SAO2",
]

E4_CHANNELS = [
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "EDA",
    "HR",
    "IBI",
    "Sleep_Stage",
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]


def time_to_seconds(time_input):
    # If the input is a string, convert it to a datetime.time object
    if isinstance(time_input, str):
        time_obj = datetime.datetime.strptime(time_input, "%H:%M:%S.%f").time()
    elif isinstance(time_input, datetime.time):
        time_obj = time_input
    else:
        raise ValueError(
            "Unsupported type for time_input. Must be str or datetime.time."
        )

    seconds_since_start_of_day = (
        time_obj.hour * 3600
        + time_obj.minute * 60
        + time_obj.second
        + time_obj.microsecond / 1e6
    )
    # Add 24 hours (86400 seconds) if the time is past 12 AM
    if time_obj.hour < 12:
        seconds_since_start_of_day += 86400
    return seconds_since_start_of_day


class PSGFolder:
    """Open a folder containing the data from PSG recording
    during data collection
    """

    def __init__(self, folder_dir, sleep_dir, report_dir, psg_path, sid):
        """Initialize the E4folder object

        Parameters
        ----------
        folder_dir : str
            string of where the folder containing one E4 recording is located
        sleep_dir : str
            string of where the folder containing one hand-labeled sleep stages
            file is located
        report_dir : str
            string of the directory of a csv file documenting the events/labels
            of this E4 dataset
        psg_path : str
            string of the directory of a edf file of raw PSG data
        sid : str
            string of the participant number for matching its data and label

        Returns
        -------
        self : PSGFolder obj
            Object saving all relevant information regarding this PSG record

        Example
        -------
        folder_dir = './raw_data/E4_data/{}/Empatica/'.format(sid)
        sleep_dir = './raw_data/sleep_label/{}.csv'.format(sid)
        report_dir = './raw_data/APNEA/{} - report.csv'.format(sid)
        report_dir = './raw_data/PSG/{}.edf'.format(sid)
        PSG = PSGFolder(folder_dir, sleep_dir, report_dir, psg_path, sid)
        final_df = PSG.load()
        """

        self.folder_dir = folder_dir
        self.sleep_dir = sleep_dir
        self.report_dir = report_dir
        self.psg_folder = psg_path
        self.sid = sid
        self.PSG_df = None
        self.fs = None

    def read_E4_data(self):
        """Get the start and end time of E4 recording

        Returns
        -------
        E4_start_time : datetime
            Time information of the starting time of E4 recording in datetime type
        E4_end_time : datetime
            Time information of the ending time of E4 recording in datetime type
        """
        E4 = E4Folder(self.folder_dir, self.sleep_dir, self.report_dir, self.sid)
        final_E4_df = E4.get_final_df()
        E4_start_time = final_E4_df["TIMESTAMP"][0]
        E4_end_time = final_E4_df["TIMESTAMP"][len(final_E4_df) - 1]

        return E4_start_time, E4_end_time, final_E4_df

    def get_PSG(self):
        """Extract raw PSG data from its file and add corresponding time information
            to each data point

        Returns
        -------
        PSG_df : pandas dataframe
            Dataframe containing all 19 channels of PSG recording and its time
            columns: TIMESTAMP, C4-M1, F4-M1, O2-M1, Fp1-O2, T3 - CZ, CZ - T4,
                     CHIN, E1, E2, ECG, LAT, RAT, SNORE, PTAF, FLOW, THORAX,
                     ABDOMEN, SAO2
        """
        file_path = str(self.psg_folder + self.sid + ".edf")
        data = mne.io.read_raw_edf(file_path, verbose=False)
        raw_data = data.get_data()
        info = data.info
        channels = data.ch_names
        start_time = info["meas_date"]
        self.fs = info["sfreq"]

        PSG = {}
        for i in range(raw_data.shape[0]):
            # need to correct for channel names
            channel_name = channels[i]
            if channel_name == "CPAP FLOW":
                channel_name = "FLOW"
            elif channel_name == "FP1-O2":
                channel_name = "Fp1-O2"
            elif channel_name == "T3-CZ":
                channel_name = "T3 - CZ"
            PSG["{}".format(channel_name)] = raw_data[i]

        PSG_df = pd.DataFrame(PSG)
        datetime_array = np.array(
            [
                (start_time + datetime.timedelta(seconds=i / self.fs)).time()
                for i in range(raw_data.shape[1])
            ]
        )
        PSG_df["TIMESTAMP"] = datetime_array

        return PSG_df

    def extract_PSG(self, E4_start_time, E4_end_time):
        """Segment raw PSG data by the start ane end time of E4 data, and
            deidentify the time

        Parameters
        ----------
        E4_start_time : datetime
            Time information of the starting time of E4 recording in datetime type
        E4_end_time : datetime
            Time information of the ending time of E4 recording in datetime type

        Returns
        -------
        PSG_df : pandas dataframe
            Dataframe containing all 19 channels of PSG recording and its time
            columns: TIMESTAMP, C4-M1, F4-M1, O2-M1, Fp1-O2, T3 - CZ, CZ - T4,
                     CHIN, E1, E2, ECG, LAT, RAT, SNORE, PTAF, FLOW, THORAX,
                     ABDOMEN, SAO2
        """
        PSG_df = self.get_PSG()
        # sidenote: the timestamp instance does not have date, only the time of day component
        d1 = PSG_df[(PSG_df["TIMESTAMP"] >= E4_start_time)]
        d2 = PSG_df[(PSG_df["TIMESTAMP"] < E4_end_time)]
        PSG_df = pd.concat([d1, d2], ignore_index=True)

        return PSG_df

    def impute_missing_channels(self):
        for channel in PSG_CHANNELS:
            if channel not in self.PSG_df.columns:
                self.PSG_df[channel] = np.nan

    def join_e4_psg(
        self,
        E4_df,
        PSG_df,
        PSG_freq=200,
        downsample_freq=100,
        E4_signals=E4_CHANNELS,
        PSG_signals=PSG_CHANNELS,
    ):
        PSG_df["TIMESTAMP"] = PSG_df["TIMESTAMP"].apply(time_to_seconds)
        E4_df["TIMESTAMP"] = E4_df["TIMESTAMP"].apply(time_to_seconds)
        merged_df = pd.merge_asof(
            PSG_df.loc[:, ["TIMESTAMP"] + list(PSG_signals)],
            E4_df.loc[:, ["TIMESTAMP"] + list(E4_signals)],
            on="TIMESTAMP",
            direction="nearest",
        )

        if PSG_freq != downsample_freq:
            ratio = PSG_freq / downsample_freq
            indices = np.round(np.arange(0, len(merged_df), ratio)).astype(int)
            indices = indices[indices < len(merged_df)]
            merged_df = merged_df.iloc[indices].reset_index(drop=True)

        return merged_df

    def merge_e4_psg(self, E4_df, PSG_df):
        mask = E4_df["Sleep_Stage"].isin(["N1", "N2", "N3", "R", "W"])
        first_index = E4_df[mask].first_valid_index()
        if E4_df.loc[first_index, "Sleep_Stage"] != "W":
            E4_df.loc[first_index-1, "Sleep_Stage"] = "W"
            first_index -= 1
        E4_df = E4_df.loc[first_index:, :]
        E4_df = E4_df.reset_index(drop=True)
        # find valid last index
        stages_of_interest = ["N1", "N2", "N3", "W", "R"]
        last_index = E4_df[
            E4_df["Sleep_Stage"].isin(stages_of_interest)
        ].last_valid_index()

        if last_index is not None:
            E4_df = E4_df.loc[:last_index]
        else:
            # Handle the case where none of the stages are found, if necessary
            E4_df = E4_df  # or any other action you deem appropriate

        # Load PSG data

        joined_df = self.join_e4_psg(
            E4_df, PSG_df, PSG_freq=self.fs, downsample_freq=100
        )
        return joined_df

    def load(self):
        """Run all the PSG data preparing functions

        -------
        final_PSG_df : pandas dataframe
            Dataframe containing all PSG data
            time information is deidentified.
            columns: TIMESTAMP, "C4-M1", "F4-M1", "O2-M1", "Fp1-O2", "T3-CZ", "CZ-T4",
                     "CHIN", "E1", "E2", "ECG", "LAT", "RAT", "SNORE", "PTAF", "FLOW", "THORAX",
                     "ABDOMEN", "SAO2"
        """
        E4_start_time, E4_end_time, E4_df = self.read_E4_data()
        final_PSG_df = self.extract_PSG(E4_start_time, E4_end_time)
        self.PSG_df = final_PSG_df
        self.impute_missing_channels()
        self.PSG_df = self.merge_e4_psg(E4_df, self.PSG_df)
        self.PSG_df["TIMESTAMP"] = [0 + i * 1 / 100 for i in range(len(self.PSG_df))]
        return self.PSG_df


def extract_psg_all_subjects(start_index):
    # Adjust your path here
    folder_dir = "/media/nvme1/sleep/DREAMT_Version2/E4_data/"
    sleep_dir = "/media/nvme1/sleep/DREAMT_Version2/sleep_stage/"
    report_dir = "/media/nvme1/sleep/DREAMT_Version2/DREAMT_APNEA/"
    psg_path = "/media/nvme1/sleep/DREAMT_Version2/PSG/"

    info = pd.read_csv("/media/nvme1/sleep/DREAMT_Version2/participant_info.csv")
    list_sids = info.SID.tolist()
    list_sids.sort()
    aggre_path = '/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes/'

    for i in tqdm(range(start_index, len(list_sids))):
        sid = list_sids[i]
        PSG = PSGFolder(folder_dir, sleep_dir, report_dir, psg_path, sid)
        final_PSG_df = PSG.load()
        path = str(aggre_path + sid + "_PSG_df.csv")
        final_PSG_df.to_csv(path, index=False)
        print(sid)
        print(final_PSG_df.columns)


def test_psg_extraction():
    # Adjust your path here
    folder_dir = "/media/nvme1/sleep/DREAMT_Version2/E4_data/"
    sleep_dir = "/media/nvme1/sleep/DREAMT_Version2/sleep_stage/"
    report_dir = "/media/nvme1/sleep/DREAMT_Version2/DREAMT_APNEA/"
    psg_path = "/media/nvme1/sleep/DREAMT_Version2/PSG/"

    info = pd.read_csv("/media/nvme1/sleep/DREAMT_Version2/participant_info.csv")

    aggre_path = "/media/nvme1/sleep/DREAMT_Version2/PSG_dataframes/"
    sid = "S057"
    print(sid)
    
    PSG = PSGFolder(folder_dir, sleep_dir, report_dir, psg_path, sid)
    final_PSG_df = PSG.load()
    path = str(aggre_path + sid + "_PSG_df.csv")
    final_PSG_df.to_csv(path, index=False)
    print("PSG dataframe columns and shapes: ")
    print(final_PSG_df.columns)
    print(final_PSG_df.shape)


if __name__ == "__main__":
    test_psg_extraction()
    # extract_psg_all_subjects(start_index=0)
