import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from pylsl import StreamInlet, resolve_stream
import threading
from queue import Queue
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================
# Neural Network stuff
# ==========================

class CNN_LSTM3(nn.Module):
    def __init__(self):
        super(CNN_LSTM3, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=8, out_channels=24, kernel_size=20, stride=4)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(24)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lstm1 = nn.LSTM(input_size=45, hidden_size=30)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm2 = nn.BatchNorm1d(24)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(720, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.elu(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x, _ = self.lstm1(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

os.chdir("C:/ntx_project/data_analysis")
cwd = os.getcwd()
print(cwd)

vnet = CNN_LSTM3()
vnet.load_state_dict(torch.load(cwd + '/valence_reg_net'))

anet = CNN_LSTM3()
anet.load_state_dict(torch.load(cwd + '/arousal_reg_net'))

dnet = CNN_LSTM3()
dnet.load_state_dict(torch.load(cwd + '/dominance_reg_net'))

# =======================
# EEG Pull LSL
# =======================

clf_sample_len = 750

valence_arr = []
arousal_arr = []
dominance_arr = []

def update_point(scatter, coords):
    # Update the offsets for the scatter plot
    scatter._offsets3d = (np.array([coords[0]]), np.array([coords[1]]), np.array([coords[2]]))
    plt.draw()

def pull_eeg_data(queue):
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    eeg_data = []
    header = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"]

    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            eeg_data.append([*sample])

            if len(eeg_data) % clf_sample_len == 0:
                eeg_chunk = eeg_data[-750:]
                eeg_chunk = np.array(eeg_chunk)
                eeg_chunk = eeg_chunk.reshape(1, 8, 750)
                eeg_chunk_norm = stats.zscore(eeg_chunk, axis=2)
                to_clf = torch.Tensor(eeg_chunk_norm)

                valence_val = vnet(Variable(to_clf))
                arousal_val = anet(Variable(to_clf))
                dominance_val = dnet(Variable(to_clf))

                valence_arr.append(valence_val.item())
                arousal_arr.append(arousal_val.item())
                dominance_arr.append(dominance_val.item())
                print(valence_val.item(), arousal_val.item(), dominance_val.item())

                new_coords = [valence_val.item(), arousal_val.item(), dominance_val.item()]
                queue.put(new_coords)

    except KeyboardInterrupt:
        with open('eeg_data_session1.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial coordinates
    initial_coords = [0, 0, 0]
    scatter = ax.scatter(initial_coords[0], initial_coords[1], initial_coords[2], color='b')

    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])

    plt.ion()  # Turn on interactive mode
    plt.show()

    coords_queue = Queue()
    eeg_thread = threading.Thread(target=pull_eeg_data, args=(coords_queue,))
    eeg_thread.daemon = True
    eeg_thread.start()

    try:
        while True:
            while not coords_queue.empty():
                new_coords = coords_queue.get()
                update_point(scatter, new_coords)
            plt.pause(0.1)  # Keep the plot open and responsive
    except KeyboardInterrupt:
        print("Plotting stopped.")

if __name__ == "__main__":
    main()
