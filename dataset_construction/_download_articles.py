import argparse
import requests
from newspaper import Article
from newsplease import NewsPlease
import hashlib
import logging
from tqdm import tqdm 
from user_agent import generate_user_agent
import sqlite3
from datetime import datetime
import json

logging.basicConfig(level=logging.ERROR, format='[%(asctime)s] - %(levelname)s - %(message)s')

def download_site(url: str):
    headers = {
        "User-Agent": generate_user_agent()
    }
    try:
        response = requests.get(url, timeout=(5, 10), headers=headers)
        article = Article(url, headers=headers)
        article.download()
        article.parse()
        main_text = NewsPlease.from_html(response.content).maintext
        return {
            'URLID': url, 
            'MD5': hashlib.md5(url.encode()).hexdigest(),
            'Title': article.title, 
            'Top Image': article.top_image,
            'Text': main_text, 
            'Html': article.html, 
            'Images': ', '.join(article.images)
        }, None
    except Exception as e:
        return None, {
            'URL': url,
            'Error': str(e),
            'Timestamp': str(datetime.now()),
            'Headers': str(headers)
        }

def save_to_database(results, output_path, table_name):
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    if table_name == 'sites_data':
        for row in results:
            cursor.execute('''
                INSERT INTO sites_data (URLID, MD5, Title, Top_Image, Text, Html, Images) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['URLID'], row['MD5'], row['Title'], row['Top Image'], row['Text'], row['Html'], row['Images']))
    elif table_name == 'failed_sites_data':
        for row in results:
            cursor.execute('''
                INSERT INTO failed_sites_data (URL, Error, Timestamp, Headers) 
                VALUES (?, ?, ?, ?)
            ''', (row['URL'], row['Error'], row['Timestamp'], row['Headers']))
    conn.commit()
    conn.close()

def process_urls_from_txt(file_path: str, output_path: str, start_idx: int, end_idx: int):
    urls = json.load(open(file_path))[start_idx:end_idx]

    successful, failed = 0, 0
    success_data, fail_data = [], []

    progress_bar = tqdm(urls, desc="Processing URLs")
    
    for url in progress_bar:
        s_result, f_result = download_site(url.strip())
        if s_result:
            successful += 1
            success_data.append(s_result)
        if f_result:
            failed += 1
            fail_data.append(f_result)
        if len(success_data) >= 2000:
            save_to_database(success_data, output_path, 'sites_data')
            success_data.clear()
        if len(fail_data) >= 2000:
            save_to_database(fail_data, output_path, 'failed_sites_data')
            fail_data.clear()

        # Update progress bar description with successful and failed counts
        progress_bar.set_postfix(successful=successful, failed=failed)

    # Save remaining data
    if success_data:
        save_to_database(success_data, output_path, 'sites_data')
    if fail_data:
        save_to_database(fail_data, output_path, 'failed_sites_data')

    return successful, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process URLs in parallel.")
    parser.add_argument("--total_processes", type=int, required=True, help="Total number of processes")
    parser.add_argument("--current_process", type=int, required=True, help="Current process number (1-indexed)")
    parser.add_argument("--output_db", type=str, required=True, help="Output SQLite database file name")
    parser.add_argument("--input_file", type=str, default="data/text_tmp/unique_urls.json", help="Input file with URLs")
    args = parser.parse_args()

    input_file_path = args.input_file
    output_db_path = f"data/text_tmp/{args.output_db}"

    # Create tables
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sites_data (
            URLID TEXT PRIMARY KEY, 
            MD5 TEXT, 
            Title TEXT, 
            Top_Image TEXT, 
            Text TEXT, 
            Html TEXT, 
            Images TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS failed_sites_data (
            URL TEXT PRIMARY KEY, 
            Error TEXT, 
            Timestamp TEXT, 
            Headers TEXT
        )
    ''')
    conn.commit()
    conn.close()

    # Get user inputs
    total_urls = len(json.load(open(input_file_path)))

    print(f"Total number of URLs: {total_urls}")
    total_processes = args.total_processes
    current_process = args.current_process

    # Calculate start and end index for the current process
    urls_per_process = total_urls // total_processes
    start_idx = (current_process - 1) * urls_per_process
    if current_process == total_processes:
        end_idx = total_urls
    else:
        end_idx = current_process * urls_per_process

    successful, failed = process_urls_from_txt(input_file_path, output_db_path, start_idx, end_idx)
    logging.info(f"Process {current_process} completed. {successful} URLs processed successfully, {failed} failed.")
