from geodoc_config import get_service_config
from google.cloud import bigquery, storage
from geodoc_loader.download.queue import get_queue_items_query, delete_teryt_queue_items_query
from geodoc_loader.download.bigquery import get_query_result, run_delete_query
from geodoc_loader.download.gcp import upload_dicts_to_bigquery_table
import datetime
import importlib

def download_spatial_data_from_queue(queue_limit, service):
    """
    Downloads spatial data from the queue for a given service and processes it.
    Args:
        queue_limit (int): The maximum number of items to process from the queue.
        service (str): The name of the service to process data for.
    """
    print("Setting up BigQuery and Storage clients...")
    bq_client = bigquery.Client()
    project_id = bq_client.project
    storage_client = storage.Client(project=project_id)

    config = get_service_config(service_name=service, key='worker')
    # import the handler function dynamically
    handler_func = getattr(importlib.import_module(config['handler_func_lib']), config['handler_func'])

    print("Downloading queue items for data processing...")
    queue_query = get_queue_items_query(
        project_id=project_id,
        dataset_id=config['target_dataset_id'],
        table_id=config['queue_table'],
        limit=queue_limit
    )

    queue_items = get_query_result(client=bq_client, query=queue_query)
    print(f"Queue items to process: {len(queue_items)}")

    if len(queue_items) == 0:
        print("No queue items to process.")
        return
    
    for i, item in enumerate(queue_items):
        teryt = item['teryt']
        teryts_table_id = item['table_name']
        print('-'*30)
        print(f"Processing TERYT: {teryt}  ({i + 1}/{len(queue_items)})")

        shapes_upload, error_upload, log_upload = handler_func(
            teryt=teryt,
            teryts_table_id=teryts_table_id,
            project_id=project_id,
            bq_client=bq_client,
            storage_client=storage_client,
            config=config
        )

        # Handle upload errors
        upload_errors = []
        if not shapes_upload:
            print(f"Failed to upload shapes for TERYT {teryt}.")
            upload_errors.append('Failed to upload shapes')

        if not error_upload:
            print(f"Failed to upload errors for TERYT {teryt}.")
            upload_errors.append('Failed to upload errors')

        if not log_upload:
            print(f"Failed to upload logs for TERYT {teryt}.")
            upload_errors.append('Failed to upload logs')

        upload_errors = [
            {
                config['error_id_column']: None,
                'teryt': teryt,
                'error_message': error,
                'timestamp': str(datetime.datetime.now())
            } for error in upload_errors
        ]

        if len(upload_errors) > 0:
            error_upload = upload_dicts_to_bigquery_table(
                project_id=project_id,
                dataset_id=config['target_dataset_id'],
                table_name=config['errors_table'],
                data=upload_errors
            )
            if not error_upload:
                print(f"Failed to upload errors for TERYT {teryt} after processing.")

        # Delete processed item from the queue
        if shapes_upload:
            delete_query = delete_teryt_queue_items_query(
                project_id=project_id,
                dataset_id=config['target_dataset_id'],
                table_id=config['queue_table'],
                teryts=[teryt]
            )
            delete_result = run_delete_query(bq_client, delete_query)
            if not delete_result:
                print(f"Failed to delete queue item for TERYT {teryt}.")
            else:
                print(f"Deleted {delete_result} items from the queue for TERYT {teryt}.")


    print("\n" + "-"*30 + "\nAll queue items processed.")

    return