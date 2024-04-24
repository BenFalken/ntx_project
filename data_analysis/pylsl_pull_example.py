import csv
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.jit import script, trace
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

from pylsl import StreamInlet, resolve_stream

# ==========================
# Neural Network stuff
# ==========================

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=8, out_channels=40, kernel_size=20, stride=4)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(40)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lstm1 = nn.LSTM(input_size=45, hidden_size=30)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm2 = nn.BatchNorm1d(40)
        self.lstm2 = nn.LSTM(input_size=30, hidden_size=10)
        self.dropout3 = nn.Dropout(p=0.5)
        self.batchnorm3 = nn.BatchNorm1d(40)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(400,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x, _ = self.lstm1(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        x, _ = self.lstm2(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)

        return x

cwd = os.getcwd()

vnet = CNN_LSTM()
vnet.load_state_dict(torch.load(cwd + '/vnet'))

anet = CNN_LSTM()
anet.load_state_dict(torch.load(cwd + '/anet'))

dnet = CNN_LSTM()
dnet.load_state_dict(torch.load(cwd + '/dnet'))

# =======================
# EEG Pull LSL
# =======================

clf_sample_len = 750

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

eeg_data = []
header = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"]

try:
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        eeg_data.append([*sample])
        #print(timestamp, sample)
        if len(eeg_data) % clf_sample_len == 0:
            eeg_chunk = eeg_data[-750:]
            eeg_chunk = np.array(eeg_chunk)
            eeg_chunk = eeg_chunk.reshape(1, 8, 750)
            to_clf = torch.Tensor(eeg_chunk)
            # classify here, something like net.forward(Variable(to_clf))
            valence_clf = torch.argmax(vnet.forward(Variable(to_clf)), dim=-1)
            arousal_clf = torch.argmax(anet.forward(Variable(to_clf)), dim=-1)
            dominance_clf = torch.argmax(dnet.forward(Variable(to_clf)), dim=-1)
            print(valence_clf, arousal_clf, dominance_clf)
except KeyboardInterrupt:
    with open('eeg_data_session1.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        # write the data
        writer.writerow(header)
        writer.writerows(eeg_data)
    print('Data collected and exported!')
    


