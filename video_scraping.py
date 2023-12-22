import csv
import requests
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# Function to search BitChute and get video page URLs
def search_bitchute(keyword):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(f'https://www.bitchute.com/search/?query={keyword}&kind=video&sort=new')
    time.sleep(5)

    video_page_urls = []
    containers = driver.find_elements(By.CLASS_NAME, "video-result-image-container")
    for container in containers:
        links = container.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute('href')
            if href:
                video_page_urls.append(href)

    driver.quit()
    return video_page_urls


# Function to get the direct video link from a BitChute video page
def get_video_link(page_url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(page_url)
    time.sleep(3)

    try:
        video_source = driver.find_element(By.TAG_NAME, "source")
        video_link = video_source.get_attribute('src')
    except Exception as e:
        print(f"Error getting video link from {page_url}: {e}")
        video_link = None

    driver.quit()
    return video_link


# Function to download the video
def download_video(video_url, file_name):
    try:
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Downloaded {file_name}")
        else:
            print(f"Failed to download {file_name}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")


# Read keywords from a text file
def read_keywords(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def export_to_csv(data, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Keyword', 'Page URL', 'Video Link'])
        for item in data:
            writer.writerow(item)


keywords = read_keywords('keywords.txt')
download_folder = 'downloaded_videos'
csv_data = []

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

for keyword in keywords:
    print(f"Searching for {keyword}...")
    page_urls = search_bitchute(keyword)

    for page_url in page_urls:
        print(f"Processing {page_url}...")
        video_link = get_video_link(page_url)

        if video_link:
            csv_data.append([keyword, page_url, video_link])
            file_name = os.path.join(download_folder, video_link.split('/')[-1])
            print(f"Downloading {video_link}...")
            download_video(video_link, file_name)

export_to_csv(csv_data, 'videos.csv')
print("Video Scraping completed.")

