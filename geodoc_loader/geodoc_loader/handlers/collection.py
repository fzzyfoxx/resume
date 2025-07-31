from tqdm.auto import tqdm
from google.cloud import bigquery, storage
from geodoc_loader.download.validate import find_files_with_extension
from geodoc_loader.handlers.core import process_shapefile_for_columns, save_geojson, get_columns_from_config_schema, concat_gdfs
from geodoc_loader.download.gcp import upload_to_gcs, load_geojson_to_bigquery

def load_geo_collection(folder_name, config):
    """
    Loads a collection of shapefiles from a specified folder, processes them to extract specified columns.
    Saves the processed data to a GeoJSON file, uploads it to Google Cloud Storage, and loads it into BigQuery.
    Args:
        folder_name (str): The name of the folder containing the shapefiles.
        config (dict): Configuration dictionary containing schema, encoding, and other parameters. Config should include:
            - 'schema': List of dictionaries defining the schema for the GeoDataFrame.
            - 'encoding'(optional): Dictionary with 'encode' and 'decode' keys for string encoding/decoding.
            - 'bucket_name': Name of the Google Cloud Storage bucket to upload the GeoJSON file.
            - 'dataset_name': Name of the BigQuery dataset where the GeoJSON will be loaded.
            - 'table_name': Name of the BigQuery table where the GeoJSON will be loaded.
    Returns:
        bool: True if the process was successful, False otherwise.
    """
    # Get all shapefiles paths in the folder
    shp_paths = find_files_with_extension(folder_name, '.shp')
    if not shp_paths:
        print(f"No shapefiles found in {folder_name}")
        return False
    
    # Extract specified columns from config
    columns_set = get_columns_from_config_schema(config['schema'])
    
    # Get encoding specification from config
    encoding = config.get('encoding', {})
    encode = encoding.get('encode')
    decode = encoding.get('decode')

    # Load shapefiles
    gdf_list = []
    for shp_path in tqdm(shp_paths, desc=f"Processing shapefiles in {folder_name}"):
        gdf, err = process_shapefile_for_columns(shp_path, columns_set, encode, decode)
        if err:
            print(f"Error processing {shp_path}: {err}")
            return False
        gdf_list.append(gdf)

    # Concatenate all GeoDataFrames
    merged_gdf = concat_gdfs(gdf_list)

    # Save the merged GeoDataFrame to a GeoJSON file
    output_geojson_path, err = save_geojson(merged_gdf, folder_name, config['table_name'])
    if err:
        print(f"Error saving GeoJSON: {err}")
        return False
    
    print(f"GeoJSON saved successfully to {output_geojson_path}")

    # Upload the GeoJSON file to GCS
    bucket_name = config['bucket_name']
    bigquery_client = bigquery.Client()
    project_id = bigquery_client.project
    bucket_name = f"{project_id}-{bucket_name}"
    storage_client = storage.Client(project=project_id)
    gcs_uri, err = upload_to_gcs(storage_client, bucket_name, folder_name,f"{config['table_name']}.geojson", output_geojson_path)
    if err:
        print(f"Error uploading GeoJSON to GCS: {err}")
        return False
    print(f"GeoJSON uploaded to GCS: {gcs_uri}")

    # Load the GeoJSON into BigQuery
    result, err = load_geojson_to_bigquery(bigquery_client, project_id, config['dataset_name'], config['table_name'], gcs_uri)
    if err:
        print(f"Error loading GeoJSON to BigQuery: {err}")
        return False
    print(f"GeoJSON loaded into BigQuery table {config['dataset_name']}.{config['table_name']}")
    return True