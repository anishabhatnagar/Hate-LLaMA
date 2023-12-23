import pandas as pd
import os
import csv
import pandas as pd
import glob
import moviepy.editor as mp
import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import librosa.display
import matplotlib.pyplot as plt
import tarfile
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle

# Helper function to generate mfccs
def extract_mfcc(path):
    #print("p",path)
    audio, sr=librosa.load(path)
    #print("p",sr)
    mfccs=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)
	
FOLDER_NAME ='./'
audioPath = "Embeddings/"  

allAudioFeatures = {}
failedList = []

with open('Data/train_list.pkl','rb') as fp:
    train_list = pickle.load(fp)
    
with open('Data/val_list.pkl','rb') as fp:
    val_list = pickle.load(fp)

with open('Data/test_list.pkl','rb') as fp:
    test_list = pickle.load(fp)


for i in tqdm(train_list):
    try:
        #print(audioPath+"train_audios_new/"+i[:-4]+".wav")
        aud = extract_mfcc(audioPath+"train_audios_new/"+ i[:-4]+".wav")
        allAudioFeatures[i]=aud
    except:
        print("Error", i)
        failedList.append(i)


for i in tqdm(val_list):
    try:
        aud = extract_mfcc(audioPath+"val_audios_new/"+i[:-4]+".wav")
        allAudioFeatures[i]=aud
    except:
        print("Error", i)
        failedList.append(i)
        
for i in tqdm(test_list):
    try:
        aud = extract_mfcc(audioPath+"test_audios_new/"+i[:-4]+".wav")
        allAudioFeatures[i]=aud
    except:
        print("Error", i)
        failedList.append(i)


for i in failedList:
    allAudioFeatures[i] = np.zeros(40)


import pickle
with open(FOLDER_NAME+'MFCCFeatures.p', 'wb') as fp:
    pickle.dump(allAudioFeatures,fp)


#--------------------Audio Spectrum Generation For VGG19---------------------

import os
from os import walk, listdir
import cv2
import shutil
import pandas as pd
from scipy.io.wavfile import read
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


for selected_folder in tqdm(train_list):
    try:
        path1 = os.path.join("Embeddings/train_audios_new", selected_folder[:-4]+'.wav')

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(111)
        ax.plot(read(path1)[1])

        fig.savefig("Embeddings/" + "Audio_plots/" + selected_folder[:-4] + '.png')
        fig.clf()
        plt.close(fig)

    except Exception as e:
        print(e)
        continue
        
for selected_folder in tqdm(val_list):
    try:
        path1 = os.path.join("Embeddings/val_audios_new", selected_folder[:-4]+'.wav')

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(111)
        ax.plot(read(path1)[1])

        fig.savefig("Embeddings/" + "Audio_plots/" + selected_folder[:-4] + '.png')
        fig.clf()
        plt.close(fig)

    except Exception as e:
        print(e)
        continue

for selected_folder in tqdm(test_list):
    try:
        path1 = os.path.join("Embeddings/test_audios_new", selected_folder[:-4]+'.wav')

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(111)
        ax.plot(read(path1)[1])

        fig.savefig("Embeddings/" + "Audio_plots/" + selected_folder[:-4] + '.png')
        fig.clf()
        plt.close(fig)

    except Exception as e:
        print(e)
        continue       
