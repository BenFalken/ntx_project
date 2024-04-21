# Here is my current code that trains a model given some data. I want you to write code that will train the model and then output it somehow so that later, I can use it in a live program to feed in EEG data live through a data stream.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.jit import script, trace
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.transforms import v2

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

x_train = None
y_train = None

x_valid = None
y_valid = None

x_test = None
y_test = None
train_subset = torch.utils.data.TensorDataset(
    torch.Tensor(x_train), torch.Tensor(y_train)
)
val_subset = torch.utils.data.TensorDataset(
    torch.Tensor(x_valid), torch.Tensor(y_valid)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(x_test), torch.Tensor(y_test)
)

bsz = 20
train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=bsz)
val_loader = torch.utils.data.DataLoader(val_subset, shuffle=False, batch_size=bsz)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=bsz)
samp_time = 750
n_channels = 8


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=8, out_channels=40, kernel_size=20, stride=4
        )
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
        self.dense = nn.Linear(400, 2)
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


input_size = (n_channels, samp_time)

# Random input tensor with the specified dimensions
input_tensor = torch.randn(1, *input_size)
print(input_tensor.shape)

# Forward pass through the model
# output_tensor = model(input_tensor)

net = CNN_LSTM()
print(net.forward(Variable(input_tensor)))
criterion = nn.CrossEntropyLoss()  # add to device here
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-7)
train_hist = []
val_hist = []

num_epochs = 100

for epoch_idx in tqdm(range(num_epochs)):
    net.train()
    train_count = 0
    train_correct_count = 0
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.float()
        train_y = train_y.long()
        optimizer.zero_grad()
        logits = net(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat = torch.argmax(logits, dim=-1)
            train_correct_count += torch.sum(y_hat == train_y, axis=-1)
            train_count += train_x.size(0)

    train_acc = train_correct_count / train_count
    train_hist.append(train_acc)

    net.eval()
    val_count = 0
    val_correct_count = 0
    with torch.no_grad():
        for idx, (val_x, val_y) in enumerate(val_loader):
            val_x = val_x.float()
            val_y = val_y.long()
            logits = net(val_x).detach()
            y_hat = torch.argmax(logits, dim=-1)
            val_correct_count += torch.sum(y_hat == val_y, axis=-1)
            val_count += val_x.size(0)
    val_acc = val_correct_count / val_count
    val_hist.append(val_acc)

    print("Train acc: {:.3f}, Val acc: {:.3f}".format(train_acc, val_acc))
### Data
X = np.load("preprocessed_data.npy")
X = np.swapaxes(X, 1, 2)
X.shape
x_train = X
y_train = "valence"
### Labels
emocsv = pd.read_csv("emotion-data/Emotions! (Core) (Responses) - Form Responses 1.csv")
emocsv2 = emocsv.iloc[:, 4:]
new_cols = [
    "How feel",
    "Pos",
    "Energ",
    "Dom",
    "Content",
    "Amused",
    "Angry",
    "Sad",
    "Disgust",
    "Afraid",
    "Emo",
]
emo_labels = pd.DataFrame()

for row in range(3):
    sel_row = emocsv2.iloc[row]
    for i in np.arange(0, emocsv2.shape[1], 11):
        obs = sel_row[i : i + 11].to_frame().T
        obs = obs.rename(columns={obs.columns[i]: new_cols[i] for i in range(11)})
        emo_labels = pd.concat([emo_labels, obs])
emo_labels["Pos2"] = (emo_labels["Pos"] > 4) + 0
emo_labels["Energ2"] = (emo_labels["Energ"] > 4) + 0
emo_labels["Dom2"] = (emo_labels["Dom"] > 4) + 0
valence = emo_labels["Pos2"].to_numpy()
valence
emo_labels
formal_emotions = emo_labels["Emo"]
formal_emotions = formal_emotions.to_numpy()
formal_emotions = formal_emotions.reshape((formal_emotions.size, 1))
from sklearn.preprocessing import OneHotEncoder

cat = OneHotEncoder()
emotions_onehot = cat.fit_transform(formal_emotions).toarray()
emotions_onehot

net.forward(Variable(torch.Tensor(x_train)))
