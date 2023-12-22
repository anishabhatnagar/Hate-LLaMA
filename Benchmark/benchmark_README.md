# Hate-LLaMA

Step 1: python3 video_scraping.py

This script is designed to automate the process of searching for videos on BitChute based on specific keywords, downloading these videos, and saving their details into a CSV file using Selenium WebDriver and ChromeDriver.


Step 2: python3 video_lengths_list.py

This script processes MP4 videos in a specified folder to extract their durations, which are then saved into a CSV file. It then merges this duration data with another CSV file containing video details like keywords and URLs, extracting filenames from video links for accurate merging.


Step 3: python3 clean_data.py

This script filters videos longer than 4 minutes from a CSV file, moves the remaining short videos to a new folder, and then renames them in a standardized format for easier management, updating the CSV file accordingly with the new filenames.


Step 4: python3 final_filter.py

The script is designed to clean up a video folder by removing videos that are not listed in a specified CSV file. 


