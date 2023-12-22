#To remove the extra videos from selct_videos that we manually deleted form our final_benchmark.csv while labelling

import os
import pandas as pd
import shutil

# Define the file and folder paths
csv_file = 'final_benchmark.csv'  # path to the CSV file
videos_folder = 'select_videos'  # path to the videos folder
removed_videos_folder = 'removed_videos'  # actual path to the removed videos folder

# Ensure the removed videos folder exists
if not os.path.exists(removed_videos_folder):
    os.makedirs(removed_videos_folder)

# Load the CSV file to get the list of videos to keep
df = pd.read_csv(csv_file)
videos_to_keep = set(df['Filename'].tolist()) 

# Iterate through the videos in the folder
for video in os.listdir(videos_folder):
    # Check if the video is not in the list of videos to keep
    if video not in videos_to_keep:
        # Move the video to the removed videos folder
        shutil.move(os.path.join(videos_folder, video), removed_videos_folder)

print("Video removal and transfer completed.")

