''' I'm writing an emotion classifier and I want you to write the pre processing step in a python script. The goal is to output 5 numpy arrays each of length N corresponding to N 3-second chunks of EEG data and their labels: "X" shall contain for each data point/3-second chunk 8 columns and 750 rows for the EEG data (8 columns representing the number of electrodes and 750 rows representing the 3 seconds of 250 Hz data), "y1" and "y2" and "y3" which are each either a 0 or 1 for each data point and represent valence, arousal, or dominance (VAD), and "y4" which is one of the emotions ("neutral", "contentment", "amusement", "anger", "sadness", "disgust", and "fear"). 

The EEG data comes in the form of "PID_6_eeg_data_x.csv" where x is the trial number which shall be iterated over. The EEG data shall be put in a folder called "eeg-data" in the same folder as the script. Electrodes 1-8 are in columns B-I with a time stamp for each data point in column AE, and the actual numbers for the data start in row 6. 

There are also timestamps in a format like "video_timestamps_PID_6_Sandra_x.csv" where x is the trial number which shall similarly be iterated over. The timestamp file will be placed in a folder called "timestamp-data" in the same folder as the script. Within each trial, videos are played and their beginning and end timestamps are recorded using time.time(). The data for this file looks like 2 columns, the first column indicating start time and second column indicating end time.

Participants dictate their emotions in response to the video in another file called "Emotions! (Core) (Responses) - Form Responses 1.csv" which will be placed in a folder called "emotion-data" in the same folder as the script. Here is code that binarizes the VAD data for this particular file:

import numpy as np
import pandas as pd 

number_videos = 24

df = pd.read_csv("/Users/ethan/Desktop/Emotions! (Core) (Responses) - Form Responses 1.csv")

print(df.iloc[0,0])

number_subjects = df.shape[0]

vad_labels = np.zeros((number_videos, number_subjects))

for i in range (0,number_subjects-1):
    for j in range (0,number_videos):
        data = df.iloc[11*j+6, i]
        if data < 5:
           vad_labels[j, i] = 0
        else:
            vad_labels[j, i] = 1

print(vad_labels)


Please write the code to generate the desired arrays
'''

import os
import numpy as np
import pandas as pd

# Constants
N = 750  # Number of samples for 3 seconds at 250 Hz
number_videos = 24
sampling_rate = 250  # Hz

# Paths
eeg_data_folder = "eeg-data"
timestamp_data_folder = "timestamp-data"
emotion_data_folder = "emotion-data"
emotion_responses_filename = "Emotions! (Core) (Responses) - Form Responses 1.csv"

# Read the emotion and VAD data
emotion_df = pd.read_csv(os.path.join(emotion_data_folder, emotion_responses_filename))
number_subjects = emotion_df.shape[0]

# Initialize arrays
X = []
y1, y2, y3, y4 = [], [], [], []

# Define a function to binarize VAD data
def binarize_vad(data):
    return 1 if data >= 5 else 0

# Process each subject and trial
for subject_id in range(1, number_subjects + 1):
    for trial_number in range(1, number_videos + 1):
        # EEG data file path
        eeg_file = f"PID_{subject_id}_eeg_data_{trial_number}.csv"
        eeg_file_path = os.path.join(eeg_data_folder, eeg_file)

        # Read EEG data
        eeg_df = pd.read_csv(eeg_file_path, skiprows=5, usecols=['B','C','D','E','F','G','H','I','AE'])
        
        # Timestamp data file path
        timestamps_file = f"video_timestamps_PID_{subject_id}_Sandra_{trial_number}.csv"
        timestamps_file_path = os.path.join(timestamp_data_folder, timestamps_file)
        
        # Read timestamp data
        timestamps_df = pd.read_csv(timestamps_file_path, header=None)

        # For each video, get the EEG data corresponding to the video duration
        for index, row in timestamps_df.iterrows():
            start_time, end_time = row
            chunk_mask = (eeg_df['AE'] >= start_time) & (eeg_df['AE'] <= end_time)
            chunk_data = eeg_df[chunk_mask].iloc[:N, :8].to_numpy()

            if chunk_data.shape[0] == N:  # Ensure we have a full 3-second chunk
                X.append(chunk_data)
                y1.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 3]))  # valence
                y2.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 2]))  # arousal
                y3.append(binarize_vad(emotion_df.iloc[subject_id, trial_number * 4 - 1]))  # dominance
                y4.append(emotion_df.iloc[subject_id, trial_number * 4])  # emotion label

# Convert lists to numpy arrays
X = np.array(X)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)

# At this point, X is an array of EEG chunks, and y1, y2, y3, and y4 are the corresponding labels
