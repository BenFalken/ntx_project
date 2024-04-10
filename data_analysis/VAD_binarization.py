import numpy as np
import pandas as pd 

number_videos = 24

df = pd.read_csv("C:/ntx_project/data_analysis/Emotions! (Core) (Responses) - Form Responses 1.csv")

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
            



# column 6 is valence, 7 is arousal, 8 is dominance

#17, 18, 19

#28, 29, 30

# if value between 1-5 --> 0 (low VAD)
# if value between 6-10 --> 2 (high VAD)
