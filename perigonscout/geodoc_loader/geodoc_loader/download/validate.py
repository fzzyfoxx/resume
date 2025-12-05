import os
import json
import geopandas as gpd
from shapely.geometry import shape

def find_files_with_extension(directory, extension):
    """
    Finds all files with a specific extension in a directory and its subdirectories.
    
    Args:
        directory (str): The directory to search in.
        extension (str): The file extension to look for (e.g., '.shp').
    Returns:
        list: A list of file paths that match the specified extension.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def collect_columns_for_sources(sources, target_path):
    """
    Collects columns from shapefiles in the specified sources and returns a list of dictionaries with file names and columns.
    
    Args:
        sources (list): List of dictionaries with 'name' and 'url' for each source
        target_path (str): Path where the downloaded files are saved
    Returns:
        list: A list of dictionaries where each dictionary contains 'name', 'file', and 'columns'.
        list: A list of unique column names found across all shapefiles.
    """
    files_data = []
    columns_set = set()

    for source in sources:
        name = source['name']
        shp_files = find_files_with_extension(f'{target_path}/{name}', '.shp')
        if not shp_files:
            print(f"No shapefiles found for {name}")
            continue
        for file in shp_files:
            gdf = gpd.read_file(file)
            columns = gdf.columns.tolist()
            files_data.append({
                'name': name,
                'file': file,
                'columns': columns
            })
            columns_set.update(columns)
    return files_data, list(columns_set)

def load_first_n_rows_from_geojson(file_path, n):
    """
    Load the first N rows from a GeoJSON file where each row is saved on a new line.

    Args:
        file_path (str): Path to the GeoJSON file.
        n (int): Number of rows to load.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the first N rows.
    """
    rows = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                feature = json.loads(line.strip())
                geometry = shape(feature['geometry'])
                properties = feature['properties']
                properties['geometry'] = geometry
                rows.append(properties)
    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        return None

    # Convert rows to a GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame(rows, geometry='geometry')
        return gdf
    except Exception as e:
        print(f"Error creating GeoDataFrame: {e}")
        return None