import os
import numpy as np
import pandas as pd

# Constants
N = 750  # Number of samples for 3 seconds at 250 Hz
sampling_rate = 250  # Hz
subject_id = 6
# Paths

eeg_data_folder = "eeg-data"
timestamp_data_folder = "timestamp-data"
emotion_data_folder = "emotion-data"
emotion_responses_filename = "Emotions! (Core) (Responses) - Form Responses 1.csv"

# Read the emotion and VAD data
emotion_df = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__), emotion_data_folder, emotion_responses_filename
    )
)

# Initialize arrays
X = []
y1, y2, y3, y4 = [], [], [], []

# Define a function to binarize VAD data
def binarize_vad(data):
    return 1 if data >= 5 else 0

# Process each subject and trial
trial_number = 3
while True:
    # EEG data file path
    eeg_file = f"PID_0{subject_id}_eeg_data_{trial_number}.txt"
    eeg_file_path = os.path.join(os.path.dirname(__file__), eeg_data_folder, eeg_file)

    if not os.path.isfile(eeg_file_path):
        break  # If the file doesn't exist, exit the loop

    # Read EEG data
    eeg_df = pd.read_csv(
        os.path.join(eeg_file_path),
        skiprows=5,
        header=None,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 22],
    )  # Indices for B-I and AE assuming A starts at index 0
    # Timestamp data file path
    timestamps_file = f"video_timestamps_PID_0{subject_id}_Sandra_{trial_number}.csv"
    timestamps_file_path = os.path.join(os.path.dirname(__file__), timestamp_data_folder, timestamps_file)
    
    # Read timestamp data
    timestamps_df = pd.read_csv(timestamps_file_path, header=None)
    # For each video, get the EEG data corresponding to the video duration
    for index, row in timestamps_df.iterrows():
        start_time, end_time = row
        chunk_mask = (eeg_df[22] >= start_time) & (eeg_df[22] <= end_time)
        chunk_data = eeg_df[chunk_mask].iloc[:N, :8].to_numpy()
        if chunk_data.shape[0] == N:  # Ensure we have a full 3-second chunk
            X.append(chunk_data)
            #y1.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 3]))  # valence
            #y2.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 2]))  # arousal
            #y3.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 1]))  # dominance
            #y4.append(emotion_df.iloc[subject_id, trial_number * 4])  # emotion label
    trial_number += 1

# Convert lists to numpy arrays
X = np.array(X)
print(X.shape)
np.save(os.path.join(os.path.dirname(__file__), "preprocessed_data.npy"), X)
#y1 = np.array(y1)
#y2 = np.array(y2)
#y3 = np.array(y3)
#y4 = np.array(y4)
#print(X)
#print(np.shape(X))