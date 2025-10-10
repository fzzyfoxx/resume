from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from geodoc_loader.download.gcp import create_gcs_bucket, create_bigquery_dataset, create_bigquery_table
import json

def bdot_setup(config):

    client = bigquery.Client()
    project_id = client.project

    dataset_name = config['dataset_id']
    location = config['location']

    # prepare GCS bucket for BDOT data
    bucket_name = f"{project_id}-{config['bucket_name']}"
    bucket = create_gcs_bucket(project_id, bucket_name)

    # create BigQuery dataset for BDOT data
    dataset_id = create_bigquery_dataset(project_id, dataset_name, location=location)

    # create tables for BDOT data
    tables_info = config['data_tables']
    columns_spec = config['default_columns']

    for table_spec in tables_info:
        table_name = table_spec['table_name']
        additional_columns = table_spec['columns']
        table_id = create_bigquery_table(
            table_name=table_name,
            collection_name=dataset_name,
            project_id=project_id,
            columns_spec=columns_spec,
            additional_columns=additional_columns
        )

    # create BigQuery tables to store logs for BDOT data loading
    for table in config['system_tables']:
        create_bigquery_table(
            table_name=table['table_name'],
            collection_name=dataset_name,
            project_id=project_id,
            columns_spec=table['columns'],
            additional_columns=[]
        )