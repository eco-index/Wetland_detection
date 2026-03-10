import asyncio
import gc
from planet import Session, reporting
from planet.clients import OrdersClient
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
from httpx import RemoteProtocolError

# Define constants
#DOWNLOAD_DIR = Path("/data/BHCEP/Wetlands/Taranaki_Wetlands/Taranaki_Planet_data/downloads/Taranaki_2020_Sep_downloads")
#DOWNLOAD_DIR = Path("/data/BHCEP/Wetlands/NZ_Wetlands/NZ_Planet_data/NZ-Planet_2020_Sep")
DOWNLOAD_DIR = Path("/data/BHCEP/Wetlands/Taranaki_Wetlands/Taranaki_Planet_data/Taranaki_Planet_data_2023_cogs")
FILTER_DATE = datetime(2024, 11, 10, tzinfo=timezone.utc)
MAX_RETRIES = 3  # Max retry attempts for failed downloads
MAX_CONCURRENT_DOWNLOADS = 3  # Max concurrent downloads

# Function to calculate MD5 checksum of a file
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Pre-scan download directory to find existing files and their checksums
def scan_existing_files(directory):
    existing_files = {}
    file_count = 0
    print("Scanning existing files in download directory...")

    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_count += 1
            if file_count % 100 == 0:
                print(f"{file_count} files scanned... currently checking: {file_path}")
            existing_files[file_path.name] = calculate_md5(file_path)
    
    print(f"Finished scanning. Total files scanned: {file_count}")
    return existing_files

async def list_recent_orders(client, after_date):
    recent_order_ids = []
    async for order in client.list_orders():
        created_on_str = order['created_on'].replace("Z", "00:00")
        created_date = datetime.fromisoformat(created_on_str[:26]).astimezone(timezone.utc)
        if created_date > after_date:
            recent_order_ids.append(order['id'])
    return recent_order_ids

async def download_with_retry(client, download_link, local_file_path, retries=MAX_RETRIES):
    attempt = 0
    while attempt < retries:
        try:
            with reporting.StateBar(state="downloading") as reporter:
                reporter.update(state='downloading')
                await client.download_asset(download_link, local_file_path, progress_bar=True)
                reporter.update(state="downloaded")
            print(f"Download complete for {local_file_path}")
            return True
        except RemoteProtocolError as e:
            attempt += 1
            print(f"Attempt {attempt} failed for {local_file_path} with error: {e}")
            if attempt == retries:
                print(f"Max retries reached. Failed to download {local_file_path}")
                return False
            else:
                print("Retrying download...")

async def download_completed_orders(client, order_id, directory, existing_files, semaphore):
    async with semaphore:  # Control concurrency with semaphore
        download_path = directory / order_id
        download_path.mkdir(parents=True, exist_ok=True)

        # Retrieve order details to get the expected checksums
        order_info = await client.get_order(order_id)
        assets = order_info.get('_links', {}).get('results', [])

        # Check each asset against the existing file list and checksum
        for asset in assets:
            file_name = asset['name']
            download_link = asset['location']
            remote_checksum = asset.get('md5_checksum')
            local_file_path = download_path / file_name

            # Check if file exists and verify checksum
            if local_file_path.exists() and existing_files.get(file_name) == remote_checksum:
                print(f"File {local_file_path} already exists and matches checksum. Skipping download.")
                continue  # Skip downloading this file

            # Download the file with retry logic if it does not exist or checksum does not match
            success = await download_with_retry(client, download_link, local_file_path)
            
            # If the download was successful, calculate and store checksum
            if success:
                existing_files[file_name] = calculate_md5(local_file_path)
                print(f"Downloaded and verified {local_file_path}")

            # Trigger garbage collection after each file download to manage memory
            gc.collect()

async def main():
    async with Session() as sess:
        client = sess.client("orders")
        
        # Pre-scan existing files to check for pre-downloaded files and their checksums
        existing_files = scan_existing_files(DOWNLOAD_DIR)

        # Get orders created after the specified date
        recent_order_ids = await list_recent_orders(client, FILTER_DATE)
        print("Recent Order IDs after", FILTER_DATE, ":", recent_order_ids)

        # Set up a semaphore to limit the number of concurrent downloads
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        # Download each completed order
        tasks = [download_completed_orders(client, order_id, DOWNLOAD_DIR, existing_files, semaphore) for order_id in recent_order_ids]
        await asyncio.gather(*tasks)

# Run the async main function
asyncio.run(main())
