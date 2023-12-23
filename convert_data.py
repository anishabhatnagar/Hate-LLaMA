import json
import pandas as pd

df = pd.read_csv("HateMM_dataset/HateMM_annotation.csv")

df.columns

prompt = "Is this hateful? Answer (Yes/No)"    

df['video_file_name']

hatevids_list = []
nonhatevids_list = []

for index, row in df.iterrows():
    video_name = row['video_file_name']
    if video_name.lstrip('hate_video'):
        hatevids_list.append(video_name)
    if video_name.lstrip('non_hate_video'):
        nonhatevids_list.append(video_name)

print(len(hatevids_list))
print(len(nonhatevids_list))

json_data = []

for index, row in df.iterrows():
    video_name = row['video_file_name']
    label = row['label']
    
    answer = 'Yes' if label == 'Hate' else 'No'
    
    video_qa = {
        "video": video_name,
        "QA": [
            {"q": prompt, "a": answer}
        ]
    }
    
    json_data.append(video_qa)

json_string = json.dumps(json_data, indent=2)
with open('output.json', 'w') as json_file:
    json_file.write(json_string)


print(json_string)