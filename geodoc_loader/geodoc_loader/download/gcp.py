from google.cloud import bigquery, storage
from google.api_core.exceptions import Conflict, NotFound, GoogleAPIError
import os
from geodoc_loader.download.core import delete_local_temp_files

def create_gcs_bucket(project_id, bucket_name):
    """Creates a GCS bucket if it doesn't already exist."""
    storage_client = storage.Client(project=project_id)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except NotFound:
        print(f"Bucket '{bucket_name}' does not exist. Creating it...")
        try:
            bucket = storage_client.create_bucket(bucket_name, project=project_id)
            print(f"Bucket '{bucket_name}' created successfully.")
        except Conflict:
            # This can happen if another process creates it concurrently
            print(f"Bucket '{bucket_name}' was just created by another process.")
            bucket = storage_client.get_bucket(bucket_name) # Get a reference to it
        except Exception as e:
            print(f"Error creating bucket '{bucket_name}': {e}")
            exit(1)
    except Exception as e:
        print(f"Error checking bucket '{bucket_name}': {e}")
        exit(1)
    return bucket

def create_bigquery_dataset(project_id, dataset_name, location="EU"):
    """
    Creates a BigQuery dataset with the specified name and location.

    Args:
        project_id (str): Google Cloud project ID.
        dataset_name (str): Name of the dataset to create.
        location (str): Location for the dataset (default is "EU").

    Returns:
        str: The full dataset ID of the created dataset, or None if an error occurred.
    """
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)

        # Construct the full dataset ID
        dataset_id = f"{project_id}.{dataset_name}"

        # Define the dataset
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location

        # Create the dataset
        dataset = client.create_dataset(dataset, exists_ok=False)  # API request
        print(f"Dataset '{dataset.dataset_id}' created successfully.")
        return dataset_id

    except Conflict:
        print(f"Dataset '{dataset_name}' already exists in project '{project_id}'.")
        return None
    except GoogleAPIError as e:
        print(f"Google API error while creating dataset '{dataset_name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while creating dataset '{dataset_name}': {e}")
        return None


def create_bigquery_table(table_name, collection_name, project_id, columns_spec, additional_columns):
    """
    Creates a BigQuery table with the specified schema.

    Args:
        table_name (str): Name of the BigQuery table to create.
        collection_name (str): Name of the dataset/collection in BigQuery.
        project_id (str): Google Cloud project ID.
        columns_spec (list): List of dictionaries specifying column schema (name, type, mode).
        additional_columns (list): List of additional nullable string columns to add to the schema.

    Returns:
        str: The full table ID of the created table, or None if an error occurred.
    """
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)

        # Construct the full table ID
        dataset_id = f"{project_id}.{collection_name}"
        table_id = f"{dataset_id}.{table_name}"

        # Define the schema
        schema = [
            bigquery.SchemaField(col["name"], col["type"], mode=col["mode"])
            for col in columns_spec
        ]

        # Add additional nullable string columns
        for col_name in additional_columns:
            schema.append(bigquery.SchemaField(col_name, "STRING", mode="NULLABLE"))

        # Define the table
        table = bigquery.Table(table_id, schema=schema)

        # Create the table
        table = client.create_table(table)  # API request

        print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
        return table_id

    except Conflict:
        print(f"Table '{table_name}' already exists in dataset '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error creating table '{table_name}': {e}")
        return False

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
    blob_name = f'{folder_name}/{file_name}' if folder_name else file_name
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_filename(local_file_path)
        gcs_uri = f'gs://{bucket_name}/{blob_name}'
        print(f"Uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri, None
    except Exception as e:
        print(f"Error uploading {local_file_path} to GCS: {e}")
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
    blob_name = f'{folder_name}/{file_name}' if folder_name else file_name
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


def load_single_geojson_to_bigquery(
        bucket_name, 
        bucket_folder, 
        dataset_id, 
        table_name, 
        table_schema, 
        gcs_file_name, 
        local_file_path, 
        additional_columns=[], 
        location="EU", 
        delete_local=True):
    """
    Loads a single GeoJSON file to BigQuery after uploading it to Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        bucket_folder (str): Folder in the GCS bucket.
        dataset_id (str): BigQuery dataset ID.
        table_name (str): Name of the BigQuery table.
        table_schema (list): Schema for the BigQuery table.
        gcs_file_name (str): Name of the file in GCS.
        local_file_path (str): Path to the local GeoJSON file.
        additional_columns (list, optional): Additional columns to add to the table schema. Defaults to [].
        location (str, optional): Location for the BigQuery dataset. Defaults to "EU".
        delete_local (bool, optional): Whether to delete local temporary files after upload. Defaults to True.
    Returns:
        bool: True if the operation was successful, False otherwise.
    """

    bigquery_client = bigquery.Client()
    project_id = bigquery_client.project
    storage_client = storage.Client(project=project_id)

    create_gcs_bucket(project_id, bucket_name)
    ds_id = create_bigquery_dataset(project_id, dataset_id, location=location)
    table_id = create_bigquery_table(table_name, dataset_id, project_id, table_schema, additional_columns)
    if not table_id:
        return False
    gcs_uri, err = upload_to_gcs(storage_client, bucket_name, bucket_folder, gcs_file_name, local_file_path)
    if err:
        return False
    result, err = load_geojson_to_bigquery(bigquery_client, project_id, dataset_id, table_name, gcs_uri)
    if err:
        return False
    delete_gcs_temp_files(storage_client, bucket_name, bucket_folder, gcs_file_name)
    if delete_local:
        delete_local_temp_files(os.path.dirname(local_file_path))
    
    return result