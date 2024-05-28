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
import tkinter as tk
import random
import time
import threading
from scipy import stats
from scipy.stats import mode

ranThrough = False

# ==========================
# Neural Network stuff
# ==========================

class CNN_LSTM2(nn.Module):
    def __init__(self):
        super(CNN_LSTM2, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=8, out_channels=24, kernel_size=20, stride=4)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(24)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lstm1 = nn.LSTM(input_size=45, hidden_size=30)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm2 = nn.BatchNorm1d(24)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(720,2)
        self.softmax = nn.Softmax(dim=1)
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
        x = self.softmax(x)
        return x

os.chdir("C:/ntx_project/data_analysis")
cwd = os.getcwd()
print(cwd)

vnet = CNN_LSTM2()
vnet.load_state_dict(torch.load(cwd + '/valence_net'))

anet = CNN_LSTM2()
anet.load_state_dict(torch.load(cwd + '/arousal_net'))

dnet = CNN_LSTM2()
dnet.load_state_dict(torch.load(cwd + '/dominance_net'))

# =======================
# GUI Setup
# =======================


RED = "red"
GREEN = "green"
WIDTH, HEIGHT = 1000, 500

# Function to draw the dot
def draw_dot(canvas, color, x, y):
    canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=color, outline="", tags="dot")


def update_dot(canvas, valence_clf, arousal_clf, dominance_clf):
    canvas.delete("dot")  # Clear previous dot
    if valence_clf == 1:
        draw_dot(canvas, "white", 125, HEIGHT // 4)  # Top half of the screen
    if valence_clf == 0:
        draw_dot(canvas, "white", 125, 3* HEIGHT // 4)  # Bottom half of the screen
    if arousal_clf == 1:
        draw_dot(canvas, "white", 325, HEIGHT // 4)  # Top half of the screen
    if arousal_clf == 0:
        draw_dot(canvas, "white", 325, 3* HEIGHT // 4)  # Bottom half of the screen
    if dominance_clf == 1:
        draw_dot(canvas, "white", 525, HEIGHT // 4)  # Top half of the screen
    if dominance_clf == 0:
        draw_dot(canvas, "white", 525, 3* HEIGHT // 4)  # Bottom half of the screen


# Create the main window
root = tk.Tk()
root.title("EE-JAMS Real Time Emotional Classifier")

# Create a canvas to draw on
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Draw the rectangles
canvas.create_rectangle(50, 50, 200, 250, fill="#9fc5e8", outline="")
canvas.create_rectangle(50, 250, 200, 450, fill="#0b5394", outline="")  

canvas.create_rectangle(250, 50, 400, 250, fill="#d3d3d3", outline="")
canvas.create_rectangle(250, 250, 400, 450, fill="#999999", outline="")

canvas.create_rectangle(450, 50, 600, 250, fill="#ecb3cc", outline="")  
canvas.create_rectangle(450, 250, 600, 450, fill="#a64d79", outline="")

valence = tk.Label(root, text="Valence", fg="black", bg="white", font=("Consolas", 20))
valence.place(x=75, y=10)

arousal = tk.Label(root, text="Arousal", fg="black", bg="white", font=("Consolas", 20))
arousal.place(x=275, y=10)

dominance = tk.Label(root, text="Dominance", fg="black", bg="white", font=("Consolas", 20))
dominance.place(x=455, y=10)

high = tk.Label(root, text="HIGH", fg="black", bg="white", font=("Consolas", 10))
high.place(x=10, y=150)

low = tk.Label(root, text="LOW", fg="black", bg="white", font=("Consolas", 10))
low.place(x=10, y=350)

description = tk.Label(root, text="Valence:  Positivity/desirability of emotion\n\nArousal:  Intensity of emotion\n\nDominance:  Sense of control over emotion", fg="black", bg="white", anchor="w", justify="left", font=("Consolas", 11))
description.place(x=620, y=200)

# =======================
# EEG Pull LSL
# =======================

clf_sample_len = 750

valence_arr=[]
arousal_arr=[]
dominance_arr=[]

# first resolve an EEG stream on the lab network

def pull_eeg_data():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
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
                eeg_chunk_norm = stats.zscore(eeg_chunk, axis=2)
                to_clf = torch.Tensor(eeg_chunk_norm)
                # classify here, something like net.forward(Variable(to_clf))
                valence_clf = torch.argmax(vnet.forward(Variable(to_clf)), dim=-1)
                arousal_clf = torch.argmax(anet.forward(Variable(to_clf)), dim=-1)
                dominance_clf = torch.argmax(dnet.forward(Variable(to_clf)), dim=-1)
                valence_value = valence_clf.numpy()[0]
                arousal_value = valence_clf.numpy()[0]
                dominance_value = valence_clf.numpy()[0]
                valence_arr.append(valence_value)
                arousal_arr.append(arousal_value)
                dominance_arr.append(dominance_value)
                print(valence_clf, arousal_clf, dominance_clf)
                update_dot(canvas, valence_clf, arousal_clf, dominance_clf)

    except KeyboardInterrupt:
        with open('eeg_data_session1.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
        
        # write the data
            writer.writerow(header)
            writer.writerows(eeg_data)
        print('Data collected and exported!')
    

eeg_thread = threading.Thread(target=pull_eeg_data)
eeg_thread.daemon = True
eeg_thread.start()

root.mainloop()



valence_mode=mode(valence_arr)
arousal_mode=mode(arousal_arr)
dominance_mode=mode(dominance_arr)

valence_pred=valence_mode[0]
arousal_pred=arousal_mode[0]
dominance_pred=dominance_mode [0]

print(valence_arr)
print(valence_pred)

def predict_valence(valence_pred):
    if valence_pred == 1:
        print("Valence Prediction: HIGH")
    if valence_pred == 0:
        print("Valence Prediction: LOW")

def predict_arousal(arousal_pred):
    if arousal_pred == 1:
        print("Arousal Prediction: HIGH")
    if arousal_pred == 0:
        print("Arousal Prediction: LOW")

def predict_dominance(dominance_pred):
    if dominance_pred == 1:
        print("Dominance Prediction: HIGH")
    if dominance_pred == 0:
        print("Dominance Prediction: LOW")

predict_valence(valence_pred)
predict_arousal(arousal_pred)
predict_dominance(dominance_pred)


# =======================
# Prediction GUI
# =======================

# RED = "red"
# GREEN = "green"
# WIDTH, HEIGHT = 1000, 500

# def predict_dot(canvas, valence_pred, arousal_pred, dominance_pred):
#     canvas.delete("dot")  # Clear previous dot
#     if valence_pred == 1:
#         draw_dot(canvas, "white", 125, HEIGHT // 4)  # Top half of the screen
#     if valence_pred == 0:
#         draw_dot(canvas, "white", 125, 3* HEIGHT // 4)  # Bottom half of the screen
#     if arousal_pred == 1:
#         draw_dot(canvas, "white", 325, HEIGHT // 4)  # Top half of the screen
#     if arousal_pred == 0:
#         draw_dot(canvas, "white", 325, 3* HEIGHT // 4)  # Bottom half of the screen
#     if dominance_pred == 1:
#         draw_dot(canvas, "white", 525, HEIGHT // 4)  # Top half of the screen
#     if dominance_pred == 0:
#         draw_dot(canvas, "white", 525, 3* HEIGHT // 4)  # Bottom half of the screen

# # if not ranThrough:
# #     valence_pred=mode(valence_arr)
# #     arousal_pred=mode(arousal_arr)
# #     dominance_pred=mode(dominance_arr)
# #     ranThrough = True

# valence_pred=mode(valence_arr)
# arousal_pred=mode(arousal_arr)
# dominance_pred=mode(dominance_arr)

# # Create the main window
# root2 = tk.Tk()
# root2.title("EE-JAMS Emotional Classification")

# # Create a canvas to draw on
# canvas = tk.Canvas(root2, width=WIDTH, height=HEIGHT, bg="white")
# canvas.pack()

# # Draw the rectangles
# canvas.create_rectangle(50, 50, 200, 250, fill="#9fc5e8", outline="")
# canvas.create_rectangle(50, 250, 200, 450, fill="#0b5394", outline="")  

# canvas.create_rectangle(250, 50, 400, 250, fill="#d3d3d3", outline="")
# canvas.create_rectangle(250, 250, 400, 450, fill="#999999", outline="")

# canvas.create_rectangle(450, 50, 600, 250, fill="#ecb3cc", outline="")  
# canvas.create_rectangle(450, 250, 600, 450, fill="#a64d79", outline="")

# valence = tk.Label(root2, text="Valence", fg="black", bg="white", font=("Consolas", 20))
# valence.place(x=75, y=10)

# arousal = tk.Label(root2, text="Arousal", fg="black", bg="white", font=("Consolas", 20))
# arousal.place(x=275, y=10)

# dominance = tk.Label(root2, text="Dominance", fg="black", bg="white", font=("Consolas", 20))
# dominance.place(x=455, y=10)

# high = tk.Label(root2, text="HIGH", fg="black", bg="white", font=("Consolas", 10))
# high.place(x=10, y=150)

# low = tk.Label(root2, text="LOW", fg="black", bg="white", font=("Consolas", 10))
# low.place(x=10, y=350)

# description = tk.Label(root2, text="Valence:  Positivity/desirability of emotion\n\nArousal:  Intensity of emotion\n\nDominance:  Sense of control over emotion", fg="black", bg="white", anchor="w", justify="left", font=("Consolas", 11))
# description.place(x=620, y=200)

# predict_dot(canvas, valence_pred, arousal_pred, dominance_pred)
# root2.mainloop()