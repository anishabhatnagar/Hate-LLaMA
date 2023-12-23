import os
import pandas as pd
from moviepy.editor import VideoFileClip

source_folder = 'downloaded_videos'  
output_csv = 'video_lengths_list.csv'  

video_data = []

for file in os.listdir(source_folder):
    if file.endswith(('.mp4')):
        full_path = os.path.join(source_folder, file)
        try:
            with VideoFileClip(full_path) as video:
                video_data.append({'Filename': file, 'Duration': video.duration})
        except Exception as e:
            print(f"Error processing file {file}: {e}")

df = pd.DataFrame(video_data)

df_sorted = df.sort_values(by='Duration')

df_sorted.to_csv(output_csv, index=False)

# Count videos less than 4 minutes
count_less_than_4_min = df_sorted[df_sorted['Duration'] < 240].shape[0]
print(f"Number of videos less than 4 minutes: {count_less_than_4_min}")


# ADD LENGTHS TO CSV
# video_lengths_list.csv has 'Filename' and 'Duration'
# videos.csv has 'Keyword', 'Page URL', 'Video Link', and 'Length'

df_durations = pd.read_csv('video_lengths_list.csv')  # The CSV with filenames and durations
df_videos = pd.read_csv('videos.csv')  # The CSV with video links

# Extract the filename from the 'Video Link' column in the second CSV
df_videos['Filename'] = df_videos['Video Link'].apply(lambda x: x.split('/')[-1])

# Merge the DataFrames on the 'Filename' column
merged_df = df_videos.merge(df_durations, on='Filename', how='left')

# The 'Duration' column from df_durations now corresponds to the 'Length' in df_videos
# replacing the 'Length' column in df_videos with the 'Duration' from df_durations:
merged_df['Length'] = merged_df['Duration']

# dropping the 'Duration' column as it's no longer needed
merged_df.drop('Duration', axis=1, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('all_videos_info.csv', index=False)