from google.cloud import bigquery, storage
from geodoc_loader.download.gcp import (
    create_gcs_bucket,
    create_bigquery_dataset,
    create_bigquery_table
)

def single_table_setup(config):
    """
    Sets up a GCS bucket and a BigQuery dataset with specified tables.

    Args:
        config (dict): Configuration dictionary containing:
            - bucket_name (str): Name of the GCS bucket to create.
            - dataset_id (str): ID of the BigQuery dataset to create.
            - location (str): Location for the BigQuery dataset.
            - tables (list): List of tables to create with their specifications.
    Returns:
        None
    """
    bq_client = bigquery.Client()
    project_id = bq_client.project

    if 'bucket_name' in config:
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