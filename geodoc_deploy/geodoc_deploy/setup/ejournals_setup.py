from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from geodoc_loader.download.gcp import create_gcs_bucket, create_bigquery_dataset, create_bigquery_table
from geodoc_config import load_config_by_path, load_config
from geodoc_loader.download.gcp import upload_dicts_to_bigquery_table
import datetime

def calc_priority(year, month, starting_year, starting_month):
    return (year - starting_year) * 12 + (month - starting_month) + 1

def gen_ejournals_queue_input(provinces, starting_year, starting_month):
    queue_input = []
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    for province in provinces:
        for year in range(starting_year, current_year + 1):
            if year == current_year:
                end_month = current_month
            else:
                end_month = 12
            for month in range(1, end_month + 1):
                priority = calc_priority(year, month, starting_year, starting_month)
                queue_input.append({
                    'id': '_'.join([province, str(year), str(month)]),
                    'province_id': province,
                    'year': year,
                    'month': month,
                    'completed': False,
                    'last_call': None,
                    'acts_found': None,
                    'acts_downloaded': None,
                    'priority': priority
                })
    # Sort by priority (lower number means higher priority)
    queue_input = sorted(queue_input, key=lambda x: x['priority'])
    return queue_input

def prepare_ejournals_status_table(project_id):

    config = load_config('ejournals_config')
    
    provinces = list(load_config_by_path(config['ejournals_urls_path'], config['ejournals_urls_filename']).keys())

    queue_input = gen_ejournals_queue_input(provinces, config['starting_year'], config['starting_month'])

    upload_result = upload_dicts_to_bigquery_table(
        project_id=project_id,
        dataset_id=config['dataset_id'],
        table_name=config['status_table'],
        data=queue_input
    )

    return upload_result

def ejournals_setup(config):

    client = bigquery.Client()
    project_id = client.project

    dataset_name = config['dataset_id']
    location = config['location']

    # prepare GCS bucket for ejournals data
    for bucket_key in ['documents_bucket_name', 'images_bucket_name']:
        full_bucket_name = f"{project_id}-{config[bucket_key]}"
        bucket = create_gcs_bucket(project_id, full_bucket_name, location=location)

    # create BigQuery dataset for ejournals data
    dataset_id = create_bigquery_dataset(project_id, dataset_name, location=location)

    # create BigQuery tables
    for table in config['tables']:
        create_bigquery_table(
            table_name=table['table_name'],
            collection_name=dataset_name,
            project_id=project_id,
            columns_spec=table['columns'],
            additional_columns=[]
        )

    # upload initial data to status table
    _ = prepare_ejournals_status_table(project_id)