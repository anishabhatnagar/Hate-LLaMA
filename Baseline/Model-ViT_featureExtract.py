
FOLDER_NAME = './'
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from tqdm import tqdm
from sklearn.metrics import *
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

print(torch.__version__)


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


num_video_features = 1024
num_audio_features = 128
num_features = 256

k = 2
epochs = 1
batch_size = 1
learning_rate = 1e-4
log_interval = 1
minFrames = 100
img_x1, img_y1 = 299, 299
img_x2, img_y2 = 224, 224

begin_frame, end_frame, skip_frame = 0, minFrames, 0


# In[7]:


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#device = torch.device("cpu") 
device


# In[8]:


import pickle
# with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
#     allDataAnnotation = pickle.load(fp)

# # train, test split
# train_list, train_label= allDataAnnotation['train']
# val_list, val_label  =  allDataAnnotation['val']
# test_list, test_label  =  allDataAnnotation['test']


# # In[9]:

allVidList = []
with open('Data/train_list.pkl','rb') as fp:
    train_list = pickle.load(fp)


with open('Data/val_list.pkl','rb') as fp:
    val_list = pickle.load(fp)

with open('Data/test_list.pkl','rb') as fp:
    test_list = pickle.load(fp)
    
allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list)    

with open('Data/train_label.pkl','rb') as fp:
    train_label = pickle.load(fp)

with open('Data/val_label.pkl','rb') as fp:
    val_label = pickle.load(fp)

with open('Data/test_label.pkl','rb') as fp:
    test_label = pickle.load(fp)


allVidLab = []



allVidLab.extend(train_label)
allVidLab.extend(val_label)
allVidLab.extend(test_label)

with open('Data/train_list.pkl','rb') as fp:
    train_list = pickle.load(fp)


with open('Data/val_list.pkl','rb') as fp:
    val_list = pickle.load(fp)

with open('Data/test_list.pkl','rb') as fp:
    test_list = pickle.load(fp)
    
allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list) 




def read_images(path, selected_folder):
    X = []
    currFrameCount = 0
    videoFrameCount = len([name for name in os.listdir(os.path.join(path, selected_folder))])
    if videoFrameCount <= minFrames:
        for i in range(videoFrameCount):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))

            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    else:
        step = int(videoFrameCount/minFrames)
        for i in range(0,videoFrameCount,step):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))
            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    return X



# set path
data_image_path = "Dataset_Images/" 



for folder, label in tqdm(list(zip(allVidList, allVidLab))):
    if os.path.exists("Embeddings/VITF/"+folder[:-4]+"_vit.p")==True:
        continue
    try:
        video = read_images(data_image_path, folder[:-4])
        inputs = feature_extractor(images=video, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        video_features =[(last_hidden_states[i][0].detach().numpy()) for i in range(0,100)]
        print("here")
        with open("Embeddings/VITF/"+folder[:-4]+"_vit.p", 'wb') as fp:
            pickle.dump(video_features,fp)
        del video
        del inputs
        del last_hidden_states
    except:
        print(folder)
        pass

