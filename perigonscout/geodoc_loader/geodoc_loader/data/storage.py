from google.cloud import storage
import os
import random  # Import random for shuffling

def download_files_from_gcp(project_id, bucket_name, destination_folder, num_files, shuffle_files=False, mime_type="application/pdf"):
    """
    Downloads a specific number of files with a given MIME type from a GCP bucket to a local folder.
    Ensures files are saved with the correct extensions.

    Args:
        project_id (str): The GCP project ID.
        bucket_name (str): The name of the GCP bucket.
        destination_folder (str): The local folder to save the downloaded files.
        num_files (int): The number of files to download.
        shuffle_files (bool): Whether to shuffle the files before downloading.
        mime_type (str): The MIME type to filter files (e.g., "application/pdf" for PDFs).

    Returns:
        list: A list of downloaded file paths.
    """
    # Initialize the GCP storage client
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(f"{project_id}-{bucket_name}")

    # List all blobs (files) in the bucket and filter by MIME type
    blobs = [blob for blob in bucket.list_blobs() if blob.content_type == mime_type]
    total_files = len(blobs)

    if total_files == 0:
        print(f"No files with MIME type '{mime_type}' found in the bucket.")
        return []

    # Shuffle the files if the flag is set to True
    if shuffle_files:
        random.shuffle(blobs)

    # Determine the number of files to download
    files_to_download = min(num_files, total_files)
    print(f"Downloading {files_to_download} file(s) with MIME type '{mime_type}' from the bucket...")

    downloaded_files = []

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Map MIME types to file extensions
    mime_to_extension = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "text/plain": ".txt",
        # Add more MIME types and extensions as needed
    }
    file_extension = mime_to_extension.get(mime_type, "")

    for i, blob in enumerate(blobs[:files_to_download]):
        # Append the correct file extension if it exists
        file_name = blob.name
        if file_extension and not file_name.endswith(file_extension):
            file_name += file_extension

        destination_path = os.path.join(destination_folder, file_name)
        blob.download_to_filename(destination_path)
        downloaded_files.append(destination_path)
        print(f"Downloaded {i + 1}/{files_to_download}: {file_name}")

    return downloaded_files