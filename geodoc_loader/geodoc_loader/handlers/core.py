from shapely.ops import transform
import pandas as pd
import uuid
import geopandas as gpd
import json
import os

def get_columns_from_config_schema(schema):
    """
    Extracts column names from the schema, excluding the 'geometry' column.
    Args:
        schema (list): List of dictionaries representing the schema.
    Returns:
        list: List of column names excluding 'geometry'.
    """
    columns = []
    for field in schema:
        if 'name' in field:
            if field['name'] != 'geometry':  # Exclude geometry column
                columns.append(field['name'])
    return columns


def process_shapefile_for_columns(file_path, columns_set, encode=None, decode=None):
    """
    Processes a shapefile to extract specified columns, handle geometries, and convert CRS.
    Args:
        file_path (str): Path to the shapefile.
        columns_set (list): List of column names to extract.
        encode (str): Encoding to apply to string columns.
        decode (str): Decoding to apply to string columns.
    Returns:
        tuple: A tuple containing the processed GeoDataFrame and an error message (if any).

    If an error occurs during reading or CRS conversion, the function returns None and the error message.
    If the geometry column is not named 'geometry', it renames it to 'geometry'.
    If there are empty geometries or invalid geometries, those rows are skipped.
    If the CRS is not EPSG:4326, it converts the GeoDataFrame to that CRS.
    It also ensures that all specified columns are present, adding them with None values if they are missing.
    The function adds 'source' and 'filename' columns based on the file path and assigns a unique ID to each row.
    Finally, it reorders the columns to match the specified columns_set and returns the GeoDataFrame.
    If any critical error occurs, it returns None and the error message.        
    """
    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        # Error during reading is considered critical for this file
        return None, str(e)
    
    # Assure that geometry is stored in the 'geometry' column
    if gdf.geometry.name != 'geometry':
        gdf = gdf.set_geometry(gdf.geometry.name)

    # Remove empty geometries
    initial_rows = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty]
    if len(gdf) < initial_rows:
        print(f"Skipped {initial_rows - len(gdf)} empty geometries in {file_path}.")

    # Flatten geometries with Z-axis to 2D
    def remove_z(geometry):
        if geometry.is_empty:
            return geometry
        return transform(lambda x, y, z=None: (x, y), geometry)

    gdf.geometry = gdf.geometry.apply(remove_z)

    # Try to validate geometries inproper geometries
    if not gdf.geometry.is_valid.all():
        print(f"Invalid geometries found in {file_path}. Attempting to fix them.")
        gdf.geometry = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)


    # Make sure that every geometry is valid, if not, skip the row
    initial_rows = len(gdf)
    gdf = gdf[gdf.geometry.is_valid]
    if len(gdf) < initial_rows:
        print(f"Skipped {initial_rows - len(gdf)} invalid geometries in {file_path}.")

    # Convert CRS from 2180 to 4326
    if gdf.crs != 'EPSG:4326':
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception as e:
            # Error during CRS conversion is critical for this file
            return None, str(e)
        
    for col in (columns_set):
        if col not in gdf.columns:
            gdf[col] = None
        else:
            if encode and decode:
                # Convert to string with specified encoding and decoding
                gdf[col] = gdf[col].apply(
                    lambda x: str(x).encode(encode).decode(decode) if pd.notna(x) else None
                )
            else:
                # Convert to string, handling potential non-string types
                gdf[col] = gdf[col].apply(lambda x: str(x) if pd.notna(x) else None)


    # get folder name as source name
    if 'source' in columns_set:
        source_name = os.path.basename(os.path.dirname(file_path))
        gdf['source'] = source_name
    # get file name without extension
    if 'filename' in columns_set:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        gdf['filename'] = file_name

    # Add id column with unique values for each row
    gdf['id'] = [str(uuid.uuid4()) for _ in range(len(gdf))]

    # Reorder columns to match the columns_set
    gdf = gdf[columns_set + ['geometry']]

    return gdf, None

def concat_gdfs(gdf_list):
    """
    Concatenates a list of GeoDataFrames into a single GeoDataFrame.
    Args:
        gdf_list (list): List of GeoDataFrames to concatenate.
    Returns:
        GeoDataFrame: A single GeoDataFrame containing all geometries and properties from the list.
    """
    # Concatenate all GeoDataFrames for the current table_name
    merged_gdf = pd.concat(gdf_list, ignore_index=True)

    # Ensure CRS is set, as it can sometimes be lost during concatenation
    if merged_gdf.crs is None and gdf_list[0].crs is not None:
        merged_gdf.crs = gdf_list[0].crs
    
    return merged_gdf


def save_geojson(gdf, target_path, file_name):
    """
    Saves a GeoDataFrame to a GeoJSON file, ensuring each feature is on a new line.
    Args:
        gdf (GeoDataFrame): The GeoDataFrame to save.
        target_path (str): The directory where the GeoJSON file will be saved.
        file_name (str): The name of the GeoJSON file (without extension).
    Returns:
        tuple: A tuple containing the path to the saved GeoJSON file and an error message (if any).
    """
    os.makedirs(target_path, exist_ok=True)
    output_geojson_path = os.path.join(target_path, f"{file_name}.geojson")

    try:
        # Write each GeoJSON Feature on a new line
        with open(output_geojson_path, 'w') as f:
            for _, row in gdf.iterrows():
                feature = {
                    "type": "Feature",
                    "id": row["id"],  # Move 'id' to the top level
                    "geometry": row.geometry.__geo_interface__,
                    "properties": row.drop(labels=["geometry", "id"]).to_dict()
                }
                f.write(json.dumps(feature) + '\n')
        print(f"GeoJSON saved to {output_geojson_path}")
        return output_geojson_path, None
    except Exception as e:
        err = str(e)
        if os.path.exists(output_geojson_path):
            os.remove(output_geojson_path)  # Clean up partial file
        return None, err