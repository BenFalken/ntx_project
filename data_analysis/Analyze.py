# I'm writing an emotion classifier and I want you to write the pre processing step in a python script. The goal is to output 5 numpy arrays each of length N corresponding to N 3-second chunks of EEG data and their labels: "X" shall contain 8 columns and 750 rows for the EEG data (8 columns representing the number of electrodes and 750 rows representing the 3 seconds of 250 Hz data), "y1" and "y2" and "y3" which are each either a 0 or 1 for each data point, and "y4" which is one of the emotions ("neutral", "contentment", "amusement", "anger", "sadness", "disgust", and "fear"). The EEG data comes in the form of "PID_6_eeg_data_x.csv" where x is the trial number which shall be iterated over. The EEG data shall be put in the same folder as the script. There are also timestamps in a format like "video_timestamps_PID_6_Sandra_x" where x is the trial number which shall be iterated over. Within each trial, videos are played and I want you to ...

import pandas as pd
import os

# Participant IDs
pids = [1, 2, 3]  # Example participant IDs

# Path to the data directory relative to this script
data_dir = "../data_analysis/eeg-data/"
timestamps_dir = "../data_collection/Data/"

# Initialize an empty list to store the cropped EEG data arrays
cropped_eeg_data = []

# Loop through each participant ID
for pid in pids:
    # Build file paths
    eeg_file_path = os.path.join(data_dir, f"eeg_data_PID-{pid}.csv")
    video_timestamps_path = os.path.join(data_dir, f"video_timestamps_PID_{pid}.csv")

    # Read EEG data
    eeg_data = pd.read_csv(eeg_file_path)

    # Read video timestamps
    video_timestamps = pd.read_csv(
        video_timestamps_path, header=None, names=["start", "end"]
    )

    # Assuming the first row contains the relevant start and end timestamps
    start_timestamp, end_timestamp = (
        video_timestamps.loc[0, "start"],
        video_timestamps.loc[0, "end"],
    )

    # Filter EEG data based on the timestamps
    # Assuming column 'W' contains timestamps in eeg_data
    cropped_data = eeg_data[
        (eeg_data["W"] >= start_timestamp) & (eeg_data["W"] <= end_timestamp)
    ]

    # Append the cropped data to the list
    cropped_eeg_data.append(cropped_data)

# cropped_eeg_data now contains the EEG data cropped to video playback times for each participant