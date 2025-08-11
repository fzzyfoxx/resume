from geodoc_loader.area.queries import get_query_result
from pyproj import Transformer
from shapely.geometry import Polygon
import uuid
import geopandas as gpd
import json
from google.cloud import bigquery, storage
from geodoc_loader.handlers.geom import convert_geom_from_wkt
from geodoc_loader.handlers.core import save_geojson

def get_country_bbox_query(project_id, table_id='country', dataset_id='geo'):

    query = f"""
    SELECT
    ST_BOUNDINGBOX(geometry) AS bbox
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    """

    return query

def adjust_grid_dimension(d_min, d_max, grid_size, min_side_buffer=0, decimals=-3):
    """
    Adjusts the grid dimension to ensure it aligns with the specified grid size and includes a minimum side buffer.
    Args:
        d_min (float): Minimum value of the dimension.
        d_max (float): Maximum value of the dimension.
        grid_size (float): Size of the grid cells.
        min_side_buffer (float): Minimum buffer to add to each side of the dimension.
        decimals (int): Number of decimal places to round to. Default is -3 for rounding to the nearest thousandth.
    Returns:
        tuple: Adjusted minimum and maximum values of the dimension.
    """
    grid_remainder = ((d_max - d_min) + 2*min_side_buffer) % grid_size

    side_buffer = min_side_buffer + (grid_size - grid_remainder)/2

    d_min_shifted, d_max_shifted =  d_min - side_buffer, d_max + side_buffer

    round_adjust = d_min_shifted % 10**-decimals
    d_min_adjusted = d_min_shifted - round_adjust
    d_max_adjusted = d_max_shifted + grid_size - round_adjust

    return d_min_adjusted, d_max_adjusted

def adjust_grid_range(bottom_left, top_right, grid_size, min_side_buffer=0, decimals=-3):
    """
    Adjusts the bounding box coordinates to ensure they align with the specified grid size and includes a minimum side buffer.
    Args:
        bottom_left (tuple): Coordinates of the bottom left corner (x_min, y_min).
        top_right (tuple): Coordinates of the top right corner (x_max, y_max).
        grid_size (float): Size of the grid cells.
        min_side_buffer (float): Minimum buffer to add to each side of the bounding box.
        decimals (int): Number of decimal places to round to. Default is -3 for rounding to the nearest thousandth.
    Returns:
        tuple: Adjusted bottom left and top right coordinates as tuples ((x_min, y_min), (x_max, y_max)).
    """
    x_min, y_min = bottom_left
    x_max, y_max = top_right

    x_min, x_max = adjust_grid_dimension(x_min, x_max, grid_size, min_side_buffer, decimals)
    y_min, y_max = adjust_grid_dimension(y_min, y_max, grid_size, min_side_buffer, decimals)

    return (x_min, y_min), (x_max, y_max)

def generate_flat_grid(x_min, y_min, x_max, y_max, grid_size, crs='EPSG:2180'):
    """
    Generates a grid of polygons within the specified bounding box and stores it as a GeoDataFrame.

    Args:
        x_min (float): Minimum x-coordinate of the bounding box.
        y_min (float): Minimum y-coordinate of the bounding box.
        x_max (float): Maximum x-coordinate of the bounding box.
        y_max (float): Maximum y-coordinate of the bounding box.
        grid_size (float): Size of each grid cell.
        crs (str): Coordinate Reference System in EPSG format (default is 'EPSG:2180').

    Returns:
        GeoDataFrame: GeoDataFrame containing the grid polygons with a unique "id" column.
    """
    polygons = []
    ids = []

    for x in range(int(x_min), int(x_max), grid_size):
        for y in range(int(y_min), int(y_max), grid_size):
            polygons.append(Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size)
            ]))
            ids.append(str(uuid.uuid4()))  # Generate a unique ID for each polygon

    grid_gdf = gpd.GeoDataFrame({'geometry': polygons, 'id': ids}, crs=crs)
    return grid_gdf

def delete_grid_outside_country(bigquery_client, grid_table, country_table, dataset_id, project_id):
    """
    Deletes grid cells that are outside the specified country.

    Args:
        bigquery_client (bigquery.Client): BigQuery client instance.
        grid_table (str): Fully qualified name of the grid table in the format 'project.dataset.table'.
        country_table (str): Fully qualified name of the country table in the format 'project.dataset.table'.
        dataset_id (str): BigQuery dataset ID containing the grid and country tables.
        project_id (str): Google Cloud project ID.

    Returns:
        None
    """
    query = f"""
    DELETE FROM `{project_id}.{dataset_id}.{grid_table}`
    WHERE NOT ST_INTERSECTS(
        geometry,
        (SELECT geometry FROM `{project_id}.{dataset_id}.{country_table}` LIMIT 1)
    )
    """

    job = bigquery_client.query(query)

    try:
        job.result()  # Wait for the job to complete
        print(f"Deleted grid cells outside the country from {grid_table}.")
        return True
    except Exception as e:
        print(f"Error deleting grid cells: {e}")
        return False
    
from geodoc_loader.handlers.geom import convert_geom_from_wkt
from geodoc_loader.handlers.core import save_geojson
from geodoc_loader.download.gcp import load_single_geojson_to_bigquery
import os
    
def prepare_and_load_grid(grid_size, min_side_buffer, config_path, delete_local=True):

    # load config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract configuration parameters
    source_crs = config.get('source_crs', 'EPSG:4326')
    grid_crs = config.get('grid_crs', 'EPSG:2180')
    target_crs = config.get('target_crs', 'EPSG:4326')
    temp_folder = config.get('temp_folder', './grid_temp')
    dataset_id = config.get('dataset_id', 'geo')
    bucket_folder = config.get('bucket_folder', 'grid')
    name_prefix = config.get('prefix', 'grid')
    adjust_decimals = config.get('adjust_decimals', -3)

    table_name = f"{name_prefix}_{grid_size}"
    table_schema = config['table_schema']

    bigquery_client = bigquery.Client()
    project_id = bigquery_client.project
    bucket_name = f"{project_id}-{config.get('bucket_name', 'single-load')}"

    # Get country bounding box
    query = f"select * from `{project_id}.{dataset_id}.country` limit 1"
    country_wkt = get_query_result(bigquery_client, query)
    country_geom = convert_geom_from_wkt(country_wkt[0]['geometry'], in_crs=source_crs, out_crs=grid_crs)
    country_bbox = list(country_geom.bounds)
    print(f"Country bounding box prepared")

    # Adjust the bounding box to fit the grid size and minimum side buffer
    bottom_left = country_bbox[:2]
    top_right = country_bbox[2:]
    bottom_left_adjusted, top_right_adjusted = adjust_grid_range(
            bottom_left, top_right, grid_size=grid_size, min_side_buffer=min_side_buffer, decimals=adjust_decimals
        )
    
    # Generate the grid
    try:
        grid_gdf = generate_flat_grid(*bottom_left_adjusted, *top_right_adjusted, grid_size=grid_size, crs=grid_crs)
        grid_gdf = grid_gdf.to_crs(target_crs)
        print(f"Generated grid with {len(grid_gdf)} cells.")
    except Exception as e:
        print(f"Error generating grid: {e}")
        return False

    result, err = save_geojson(grid_gdf, temp_folder, f'{name_prefix}_{grid_size}')

    if not result:
        return False

    result = load_single_geojson_to_bigquery(
            bucket_name=bucket_name,
            bucket_folder=bucket_folder,
            dataset_id=dataset_id,
            table_name=table_name,
            table_schema=table_schema,
            gcs_file_name=f'{name_prefix}_{grid_size}.geojson',
            local_file_path=os.path.join(temp_folder, f'{name_prefix}_{grid_size}.geojson'),
            additional_columns=[],
            location='EU',
            delete_local=delete_local
        )
    
    if not result:
        print(f"Error loading grid to BigQuery: {err}")
        return False
    
    result = delete_grid_outside_country(
        bigquery_client, 
        grid_table=table_name,
        country_table='country',
        dataset_id=dataset_id,
        project_id=project_id
        )
    
    return result