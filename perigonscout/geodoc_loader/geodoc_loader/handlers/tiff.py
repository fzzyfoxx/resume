import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import open as raster_open
from shapely.geometry import box
import uuid
import os

def process_tif_files(tif_files):
    """
    Process a list of .tif files to create a GeoDataFrame with pixel data and bounding boxes.

    Args:
        tif_files (list): List of paths to .tif files.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing pixel data, file names, coordinates, and bounding boxes.
    """
    # Collect data and file names
    data_list = []
    file_names = []
    for tif_file in tif_files:
        with raster_open(tif_file) as src:
            data = src.read(1)  # Read the first band
            data_list.append(data)
            file_names.append(os.path.splitext(os.path.basename(tif_file))[0])
            transform = src.transform

    # Stack data into a single numpy array
    stacked_data = np.stack(data_list, axis=-1)  # Shape: (height, width, number_of_files)

    # Create a DataFrame with pixel data and file names
    height, width, num_files = stacked_data.shape
    pixel_indices = [(row, col) for row in range(height) for col in range(width)]
    pixel_data = stacked_data.reshape(-1, num_files)  # Flatten spatial dimensions
    df = pd.DataFrame(pixel_data, columns=file_names)
    df['row'], df['col'] = zip(*pixel_indices)

    # Add (x, y) coordinates
    df['x'], df['y'] = zip(*[transform * (col, row) for row, col in zip(df['row'], df['col'])])

    # Filter out pixels with NaN values in all file-related columns
    file_columns = file_names  # List of file-related column names
    df = df.dropna(subset=file_columns)

    # Calculate bounding boxes for each pixel
    df['geometry'] = df.apply(
            lambda row: box(
                *(transform * (row['col'], row['row'])),  # Unpack (x_min, y_min)
                *(transform * (row['col'] + 1, row['row'] + 1))  # Unpack (x_max, y_max)
            ),
            axis=1
        )
    
    df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]  # Unique ID for each pixel

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=src.crs)

    return gdf

def process_single_tif(tif_file):
    """
    Process a single .tif file to create a GeoDataFrame with pixel data and bounding boxes.

    Args:
        tif_file (str): Path to the .tif file.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing pixel data, coordinates, and bounding boxes.
    """
    with raster_open(tif_file) as src:
        # Read the first band
        data = src.read(1)  # Shape: (height, width)
        transform = src.transform

        # Get file name without extension
        file_name = os.path.splitext(os.path.basename(tif_file))[0]

        # Create a DataFrame with pixel data
        height, width = data.shape
        pixel_indices = [(row, col) for row in range(height) for col in range(width)]
        pixel_data = data.flatten()  # Flatten spatial dimensions
        df = pd.DataFrame({file_name: pixel_data})
        df['row'], df['col'] = zip(*pixel_indices)

        # Add (x, y) coordinates
        df['x'], df['y'] = zip(*[transform * (col, row) for row, col in zip(df['row'], df['col'])])

        # Filter out pixels with NaN values
        df = df.dropna(subset=[file_name])

        # Calculate bounding boxes for each pixel
        df['geometry'] = df.apply(
            lambda row: box(
                *(transform * (row['col'], row['row'])),  # Unpack (x_min, y_min)
                *(transform * (row['col'] + 1, row['row'] + 1))  # Unpack (x_max, y_max)
            ),
            axis=1
        )

        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]  # Unique ID for each pixel

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=src.crs)

    return gdf
