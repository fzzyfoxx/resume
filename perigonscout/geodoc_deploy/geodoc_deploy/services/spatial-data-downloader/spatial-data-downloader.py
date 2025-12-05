import warnings
warnings.filterwarnings("ignore")

import os
from geodoc_loader.download.process import download_spatial_data_from_queue

if __name__ == "__main__":
    service = str(os.environ.get("SERVICE", None))
    queue_limit = int(os.environ.get("QUEUE_LIMIT", 1))

    if not service:
        raise ValueError("SERVICE environment variable is not set. Please set it to the name of the service you want to run.")
    
    if not queue_limit:
        raise ValueError("QUEUE_LIMIT environment variable is not set. Please set it to the number of items you want to process from the queue.")
    
    print(f"Starting spatial data downloader for service '{service}' with queue limit {queue_limit}...")
    download_spatial_data_from_queue(queue_limit=queue_limit, service=service)