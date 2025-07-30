from google.cloud import bigquery, storage

def get_bq_table_schema(client, project_id, dataset_name, table_name):
    """
    Retrieves the schema of a BigQuery table.
    
    Args:
        client (bigquery.Client): The BigQuery client instance.
        project_id (str): The ID of the Google Cloud project.
        dataset_name (str): The name of the BigQuery dataset.
        table_name (str): The name of the BigQuery table.
    
    Returns:
        list: A list of dictionaries representing the schema of the table.
    """
    table_id = f"{project_id}.{dataset_name}.{table_name}"
    try:
        table = client.get_table(table_id)
        return [field.to_api_repr() for field in table.schema]
    except Exception as e:
        print(f"Error retrieving schema for {table_name}: {e}")
        return []
    
def upload_to_gcs(storage_client, bucket_name, folder_name, file_name, local_file_path):
    """
    Uploads a GeoJSON file to the specified GCS bucket
    Args:
        storage_client (google.cloud.storage.Client): The GCS client instance.
        bucket_name (str): The name of the GCS bucket.
        folder_name (str): The folder in the bucket where the file will be uploaded.
        file_name (str): The name of the file to be uploaded.
        local_file_path (str): The local path to the file to be uploaded.
    Returns:
        tuple: (str, str) - The GCS URI of the uploaded file and None if successful, or None and an error message if failed.
    """
    bucket = storage_client.bucket(bucket_name)
    blob_name = f'{folder_name}/{file_name}'
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_filename(local_file_path)
        gcs_uri = f'gs://{bucket_name}/{blob_name}'
        return gcs_uri, None
    except Exception as e:
        return None, str(e)
    
def load_geojson_to_bigquery(client, project_id, dataset_name, table_name, gcs_uri):
    """
    Loads data from a GeoJSON file in GCS into a BigQuery table. Returns True on success, False on failure.
    Args:
        client (bigquery.Client): The BigQuery client instance.
        project_id (str): The ID of the Google Cloud project.
        dataset_name (str): The name of the BigQuery dataset.
        table_name (str): The name of the BigQuery table.
        gcs_uri (str): The GCS URI of the GeoJSON file to load.
    Returns:
        tuple: (bool, str) - True if loading was successful, False and error message if it failed.
    """
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    schema = get_bq_table_schema(client, project_id, dataset_name, table_name)
    if not schema:
        print(f"Error: No schema found for table {table_name}. Cannot load data.")
        return False, "No schema found for table."
    
    schema = [bigquery.SchemaField(field['name'], field['type'], mode=field['mode']) for field in schema]  # Convert to SchemaField objects

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, # Append to existing table
        schema=schema,  # Use the schema retrieved from BigQuery
        # Add this line to tell BigQuery to interpret the JSON as GeoJSON
        json_extension='GEOJSON',
    )

    try:
        load_job = client.load_table_from_uri(
            gcs_uri, table_id, job_config=job_config
        )
        load_job.result()  # Waits for the job to complete
        print(f"Loaded {load_job.output_rows} rows into {table_id}")
        return True, None
    except Exception as e:
        print(f"Error loading data to BigQuery: {e}")
        return False, str(e)
    
def delete_gcs_temp_files(storage_client, bucket_name, folder_name, file_name):
    """
    Deletes temporary geojson files from GCS.
    Args:
        storage_client (google.cloud.storage.Client): The GCS client instance.
        bucket_name (str): The name of the GCS bucket.
        folder_name (str): The folder in the bucket where the file is located.
        file_name (str): The name of the file to be deleted.
    Returns:
        None
    """
    bucket = storage_client.bucket(bucket_name)
    blob_name = f'{folder_name}/{file_name}'
    blob = bucket.blob(blob_name)
    try:
        if blob.exists():
            blob.delete()
            print(f"Deleted GCS object: {blob_name}")
    except Exception as e:
        print(f"Error deleting GCS object {blob_name}: {e}")

def list_files_in_gcs_folder(bucket_name, folder_path):
    """
    Lists all files in a specific folder in a GCS bucket and returns a list of tuples (file_name, gcs_uri).

    Args:
        bucket_name (str): The name of the GCS bucket.
        folder_path (str): The folder path in the bucket (e.g., 'bdot_data/0214/').

    Returns:
        list: A list of tuples where each tuple contains (file_name, gcs_uri).
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    file_list = []
    for blob in blobs:
        # Extract the file name from the blob name
        file_name = blob.name.split('/')[-1]
        if file_name:  # Ensure it's not an empty folder
            gcs_uri = f"gs://{bucket_name}/{blob.name}"
            file_list.append((file_name, gcs_uri))
    
    return file_list