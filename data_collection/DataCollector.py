from StaticVariables import data_file_name

import csv
import numpy as np
from pylsl import StreamInlet, resolve_byprop

# grab data stream from CytonDataPackager
print("looking for an EEG stream")
stream = resolve_byprop('source_id', 'CytonDataPackager')
inlet = StreamInlet(stream[0])
print("Got stream!")

start_signal_received = False
saved_eeg_data = np.zeros((1000000, 10))
header = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8","Trigger","Shutdown"]

try:
    # init data sample index
    sample_index = 0

    print("Waiting for start signal...")
    while True:
        # get new sample until start signal detected
        if not start_signal_received:
            sample,_ = inlet.pull_sample()

            if sample[8] == 1:
                start_signal_received = True
                print("Start signal received.")

        # pull in new chunk
        if start_signal_received:
            chunk,_ = inlet.pull_chunk()

            if len(chunk) > 0:
                saved_eeg_data[sample_index:sample_index+np.shape(chunk)[0]] = np.array(chunk)
                sample_index += np.shape(chunk)[0]

                # check if a shutdown signal has been detected
                if saved_eeg_data[sample_index-1, 9] == 1:
                    print("Shutdown signal received")
                    break
except KeyboardInterrupt:
    # pull in one extra chunk
    chunk,_ = inlet.pull_chunk()

    if len(chunk) > 0:
        saved_eeg_data[sample_index:sample_index+np.shape(chunk)[0]] = np.array(chunk)
        sample_index += np.shape(chunk)[0]

# print data to csv file
with open(str(data_file_name), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(saved_eeg_data[:sample_index])
    print('Data collected and exported!')