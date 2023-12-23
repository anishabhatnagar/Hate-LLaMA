FOLDER_NAME = './'

import os
from os import walk, listdir
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
import pickle

video_labels = []
target_folder = FOLDER_NAME +'Dataset_Images/'

with open('Data/train_list.pkl','rb') as fp:
    train_list = pickle.load(fp)
    
with open('Data/val_list.pkl','rb') as fp:
    val_list = pickle.load(fp)
    
with open('Data/test_list.pkl','rb') as fp:
    test_list = pickle.load(fp)
    
folder1 = train_list + val_list + test_list

for subDir in ["val_videos_new" ,"test_videos_new"]:
    print(subDir)
    for f in tqdm(listdir(FOLDER_NAME + 'Dataset_New/' +subDir)):
        print(f)
        if(f.split('.')[-1] == 'mp4'):
          # Extracting label of video
          #print(f, FOLDER_NAME + subDir + '/' + f)
          #break
              success, _ = cv2.VideoCapture(FOLDER_NAME + 'Dataset_New/' +subDir + '/' + f).read()
              if not success:
                print(f)
                continue  
              # Extracting frames from video
              try:
                os.mkdir(os.path.join(target_folder +  f.split('.')[0]))
              except FileExistsError:
                pass
              if os.listdir(os.path.join(target_folder + '/' +  f.split('.')[0])):
                continue
              vidcap = cv2.VideoCapture(FOLDER_NAME + 'Dataset_New/' +subDir + '/' + f)
              success = True
              count = 0
              while success:
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
                success,img = vidcap.read()
                if not success:
                  break
                print("her")
                cv2.imwrite( target_folder + '/' + f.split('.')[0] + '/' + "frame_{}".format(count) + '.jpg', img)     # save frame as JPEG file
                count = count + 1
