import os
import numpy as np
import pandas as pd

# Constants
N = 750  # Number of samples for 3 seconds at 250 Hz
sampling_rate = 250  # Hz
total_videos = 23  # Total number of videos
cut_initial = 30 * sampling_rate  # Cut the first 30 seconds
cut_final = 5 * sampling_rate  # Cut the last 5 seconds

# Paths
base_path = os.path.dirname(__file__)
eeg_data_folder = os.path.join(base_path, "eeg-data")

# Initialize a list for all video data
video_data = []

# Process each video
for video_index in range(1, total_videos + 1):
    session_folder = f"OpenBCISession_0420_vid{video_index}"
    session_path = os.path.join(eeg_data_folder, session_folder)

    # Find the EEG file in the session folder
    for file in os.listdir(session_path):
        if file.startswith("OpenBCI-RAW"):
            eeg_file_path = os.path.join(session_path, file)
            break
    else:
        continue  # Skip to next video if no file is found

    # Read EEG data, assuming column indices for electrodes are from 1 to 8
    eeg_df = pd.read_csv(
        eeg_file_path, skiprows=5, header=None, usecols=list(range(1, 9))
    )

    # Handle initial and final cuts
    eeg_df = eeg_df.iloc[cut_initial:-cut_final]

    # Break data into 3-second chunks
    chunks = [
        eeg_df.iloc[i : i + N, :].values
        for i in range(0, len(eeg_df), N)
        if i + N <= len(eeg_df)
    ]
    video_data.append(chunks)

# Determine the maximum number of chunks to resize the array correctly
max_chunks = max(len(chunks) for chunks in video_data)

# Initialize a numpy array with proper dimensions now known
X = np.zeros((total_videos, 8, max_chunks, N))

# Populate the numpy array
for i, chunks in enumerate(video_data):
    for j, chunk in enumerate(chunks):
        if j < max_chunks:
            X[i, :, j, :] = chunk.T  # Transpose chunk to match the shape (8, 750)

# Save the array
print(X.shape)
print(X)
np.save(os.path.join(base_path, "preprocessed_data.npy"), X)
