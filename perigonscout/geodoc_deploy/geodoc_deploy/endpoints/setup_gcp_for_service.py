import argparse
from geodoc_config import get_service_config
import importlib

def main():
    """
    Creates a GCS bucket, a BigQuery dataset, and tables for provided service.
    """
    parser = argparse.ArgumentParser(description="Setup service in GCP")
    parser.add_argument('--service', type=str, required=True, help='Name of the service to set up')
    args = parser.parse_args()

    config = get_service_config(args.service, "setup")

    setup_module = config['setup_module']
    setup_function = config['setup_function']
    handler = getattr(importlib.import_module(setup_module), setup_function)
    
    handler(config)

if __name__ == "__main__":
    main()