# REMOVING VIDEOS FROM CSV FILE THAT ARE MORE THAN 4 MINUTES
import pandas as pd
import os
import shutil

csv_file_path = 'all_videos_info.csv'

df_4 = pd.read_csv(csv_file_path)

# Remove rows with empty 'Length' column
df_4 = df_4.dropna(subset=['Length'])

# Remove rows with 'Length' greater than 240 seconds
df_4 = df_4[df_4['Length'] <= 240]

# Save the cleaned DataFrame to a new CSV file
cleaned_csv_path = 'benchmark_dataset.csv'
df_4.to_csv(cleaned_csv_path, index=False)

print(f"The cleaned CSV file has been saved to: {cleaned_csv_path}")




# then we created a new folder called select_videos which has all the video files that are less than 4 minutes 
# moved from downloaded_videos folder with the help of a csv file above "benchmark_dataset.csv"

df_select = pd.read_csv('cleaned_file.csv')

downloaded_videos_path = 'downloaded_videos'
selected_videos_path = 'selected_videos'

os.makedirs(selected_videos_path, exist_ok=True)

for filename in df_select['Filename']:
    # Construct the full file path
    source_path = os.path.join(downloaded_videos_path, filename)
    destination_path = os.path.join(selected_videos_path, filename)
    
    # Check if the file exists in the downloaded_videos folder
    if os.path.exists(source_path):
        # Move the file
        shutil.move(source_path, destination_path)
    else:
        print(f"File {filename} does not exist in the downloaded videos folder.")




# RENAMING VIDEO FILES IN THE select_videos FOLDER (500 videos)

folder_path = "select_videos"
csv_file_path = "benchmark_dataset.csv"#benchmark_copy

df_rename = pd.read_csv(csv_file_path)

# Create a new column for the updated filenames
df_rename['New_Filename'] = ''

# Iterate through the DataFrame and rename the files
for index, row in df_rename.iterrows():
    old_filename = row['Filename']
    new_filename = f"benchmark{index + 1}.mp4" 
    
    # Rename the file in the folder
    old_file_path = os.path.join(folder_path, old_filename)
    new_file_path = os.path.join(folder_path, new_filename)
    
    try:
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_filename} to {new_filename}")
        
        # Update the DataFrame with the new filename
        df_rename.at[index, 'New_Filename'] = new_filename
    except FileNotFoundError:
        print(f"File not found: {old_filename}")
    except FileExistsError:
        print(f"File already exists: {new_filename}")

# Update the original "Filename" column with the new filenames
df_rename['Filename'] = df_rename['New_Filename']

# Drop the temporary "New_Filename" column
df_rename.drop(columns=['New_Filename'], inplace=True)

# Save the updated DataFrame to the CSV file
df_rename.to_csv(csv_file_path, index=False)

print("All files renamed and CSV updated.")




