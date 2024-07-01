import requests
import os
import zipfile
from tqdm import tqdm
import concurrent.futures

START_DATE = 202300 # dataset start date in format of yyyymm
END_DATE = 202311 # dataset end date in format of yyyymm

DATA_DIR = '../data/kg_raw'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

MAX_THREADS = 200

def download_file(line):
    if not (".export.CSV" in line and line.endswith(".zip")):
        return
    file_url = line.split(' ')[2]

    # filter by date
    file_name = os.path.basename(file_url)
    date_str = file_name[:6]
    date = int(date_str)
    if date < START_DATE or date > END_DATE:
        return

    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        full_file_path = os.path.join(DATA_DIR, file_name)
        with open(full_file_path, 'wb') as f:
            f.write(file_response.content)

        with zipfile.ZipFile(full_file_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        os.remove(full_file_path)
    else:
        print(f"Unable to download file: {file_url}")


if __name__ == "__main__":

    file_list_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    response = requests.get(file_list_url)

    if response.status_code == 200:
        file_list_lines = response.text.split('\n')
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            list(tqdm(executor.map(download_file, file_list_lines), total=len(file_list_lines)))
    else:
        print("Unable to fetch the file list. HTTP request failed.")

