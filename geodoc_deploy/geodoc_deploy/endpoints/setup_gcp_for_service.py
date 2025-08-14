from geodoc_loader.download.gcp import (
    create_gcs_bucket,
    create_bigquery_dataset,
    create_bigquery_table
)
from google.cloud import bigquery, storage
import argparse
from geodoc_config import get_service_config

def main():
    """
    It creates a GCS bucket, a BigQuery dataset, and tables for provided service.
    """
    parser = argparse.ArgumentParser(description="Setup service in GCP")
    parser.add_argument('--service', type=str, required=True, help='Name of the service to set up')
    args = parser.parse_args()

    config = get_service_config(args.service, "setup")

    bq_client = bigquery.Client()
    project_id = bq_client.project

    bucket_name = f"{project_id}-{config['bucket_name']}"

    create_gcs_bucket(project_id=project_id, bucket_name=bucket_name)
    create_bigquery_dataset(project_id=project_id, dataset_name=config['dataset_id'], location=config['location'])

    for table in config['tables']:
        create_bigquery_table(
            table_name=table['table_name'],
            collection_name=config['dataset_id'],
            project_id=project_id,
            columns_spec=table['columns'],
            additional_columns=[]
        )

if __name__ == "__main__":
    main()