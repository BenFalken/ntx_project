from StaticVariables import data_file_name, sampling_frequency, n_channels, daisy_used

import csv
import numpy as np

# import scipy.signal as sig
import time
from pylsl import StreamInlet, resolve_byprop

# grab data stream from CytonDataPackager
print("looking for an EEG stream")
stream = resolve_byprop("source_id", "CytonDataPackager")
inlet = StreamInlet(stream[0])
print("Got stream!")

start_signal_received = False
saved_eeg_data = np.zeros(
    (1000000, n_channels + 3)
)  # Added +3 for Trigger, Shutdown, and Timestamp

if daisy_used:
    header = [
        "channel1",
        "channel2",
        "channel3",
        "channel4",
        "channel5",
        "channel6",
        "channel7",
        "channel8",
        "channel9",
        "channel10",
        "channel11",
        "channel12",
        "channel13",
        "channel14",
        "channel15",
        "channel16",
        "Trigger",
        "Shutdown",
        "Timestamp",
    ]
else:
    header = [
        "channel1",
        "channel2",
        "channel3",
        "channel4",
        "channel5",
        "channel6",
        "channel7",
        "channel8",
        "Trigger",
        "Shutdown",
        "Timestamp",
    ]
# sos = sig.butter(4, [2, 40], 'bandpass', fs=sampling_frequency, output='sos')
try:
    # init data sample index
    sample_index = 0

    print("Waiting for start signal...")
    while True:
        # get new sample until start signal detected
        if not start_signal_received:
            sample, _ = inlet.pull_sample()

            if sample[n_channels] == 1:
                start_signal_received = True
                print("Start signal received.")

        # pull in new chunk
        if start_signal_received:
            chunk, timestamps = inlet.pull_chunk()
            if len(chunk) > 0:
                for i, sample in enumerate(chunk):
                    # Convert sample to list if it's a numpy array
                    sample_list = (
                        sample.tolist() if isinstance(sample, np.ndarray) else sample
                    )

                    # Append LSL timestamp and computer's current time to the sample list
                    timestamped_sample = sample_list + [timestamps[i], time.time()]

                    # Assign the timestamped sample to the saved_eeg_data array
                    saved_eeg_data[
                        sample_index, : len(timestamped_sample)
                    ] = timestamped_sample
                    sample_index += 1

                # Check if a shutdown signal has been detected
                if saved_eeg_data[sample_index - 1, n_channels + 1] == 1:
                    print("Shutdown signal received")
                    break

except KeyboardInterrupt:
    print("Interrupted by user")

# print data to csv file
with open(str(data_file_name), "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(saved_eeg_data[:sample_index])
    print("Data collected and exported!")
