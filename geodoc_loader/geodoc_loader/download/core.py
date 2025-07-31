import requests
from tqdm.auto import tqdm
import zipfile
import os
import shutil

def download_file_with_progress(url, target_path, timeout=120):
    """
    Downloads a file from a given URL and saves it to the target path, displaying a progress bar in MB.

    Args:
        url (str): The URL of the file to download.
        target_path (str): The local file path where the downloaded file will be saved.
    Returns:
        tuple: A tuple containing a boolean indicating success or failure, and an error message if applicable
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MiB

        with open(target_path, 'wb') as file, tqdm(
            desc=f"Downloading {target_path}",
            total=total_size // block_size,
            unit='MB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data) // block_size)
        return True, None
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download {url}. Reason: {e}")
        return False, str(e)
    except IOError as e:
        print(f"Error: Failed to write to {target_path}. Reason: {e}")
        return False, str(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, str(e)

def unzip_file(zip_file_path, target_directory, delete_file=True):
    """
    Unzips a ZIP file into the specified target directory.

    Args:
        zip_file_path (str): The path to the ZIP file to be extracted.
        target_directory (str): The directory where the contents will be extracted.
        delete_file (bool): If True, deletes the ZIP file after extraction.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(target_directory, exist_ok=True)
        
        # Open the ZIP file and extract its contents
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
            print(f"Successfully extracted {zip_file_path} to {target_directory}")

        # Optionally delete the ZIP file after extraction
        if delete_file:
            os.remove(zip_file_path)
            print(f"Deleted ZIP file: {zip_file_path}")
        return True, None
    except zipfile.BadZipFile as e:
        print(f"Error: {zip_file_path} is not a valid ZIP file. Reason: {e}")
        return False, str(e)
    except FileNotFoundError as e:
        print(f"Error: {zip_file_path} does not exist. Reason: {e}")
        return False, str(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, str(e)

def delete_local_temp_files(temp_dir):
    """Deletes local temporary files."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Deleted local temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error deleting local temporary directory {temp_dir}: {e}")

def download_data_from_sources(sources, target_path, timeout=120):
    """
    Downloads files from the specified sources, unzips them, and saves them to the target path.
    Args:
        sources (list): List of dictionaries containing 'name' and 'url' for each source
        target_path (str): Path where the downloaded files will be saved
        timeout (int): Timeout for the download operation in seconds
    Returns:
        list: A list of dictionaries with the status of each download operation
    """
    os.makedirs(target_path, exist_ok=True)
    files_to_download = len(sources)
    results = []
    for i, source in enumerate(sources):
        name = source['name']
        print(f"Downloading {i + 1}/{files_to_download}: {name}")
        url = source['url']
        target_file = f"{target_path}/{name}.zip"

        result, err = download_file_with_progress(url, target_file, timeout=timeout)
        if not result:
            print(f"Error downloading {url}: {err}")
            results.append({'name': name, 'status': 'failed', 'error': str(err)})
            continue

        result, err = unzip_file(target_file, f'{target_path}/{name}', delete_file=True)
        if not result:
            print(f"Error unzipping {target_file}: {err}")
            results.append({'name': name, 'status': 'failed', 'error': str(err)})
            continue

        results.append({'name': name, 'status': 'success', 'error': None})
        print(f"Successfully downloaded and unzipped {name}")

    return results